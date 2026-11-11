"""Research implementation of e-means (arXiv:2611.09501v1)."""

from __future__ import annotations

from math import cos, floor, log2, pi
from numbers import Integral
from typing import Iterator

import numpy as np

INITIAL_COUNT = 1e-10
REVIVAL_INTERVAL = 5
REVIVAL_THRESHOLD = 0.1
BFLOAT16_EPSILON = 2**-7
DISTANCE_BLOCK_BYTES = 8 * 1024**2
CENTROID_BLOCK_SIZE = 256
ROW_BLOCK_SIZE = 1024


def _positive_int(value: int, name: str) -> int:
    """Validate and normalize a positive integer."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def suggest_batch_size(n_samples: int, n_clusters: int, epochs: int) -> int:
    """Return the paper's knee batch size for n samples, k clusters and T.

    Use the Table 2 large-scale coefficient above 2 million samples.
    Round in log2 space and cap the resulting batch size at n_samples.
    """
    n = _positive_int(n_samples, "n_samples")
    k = _positive_int(n_clusters, "n_clusters")
    epochs = _positive_int(epochs, "epochs")
    if k > n:
        raise ValueError("n_clusters cannot exceed n_samples.")
    coefficient = 0.0044 if n > 2_000_000 else 0.035
    knee = coefficient * k**0.13 * n**0.88 * epochs**0.81
    return min(n, 2 ** max(0, floor(log2(knee) + 0.5)))


def suggest_alpha_0(
    n_samples: int, n_clusters: int, batch_size: int, epochs: float
) -> float:
    """Return the peak EMA step from Equation 6 and Table 2.

    epochs is the training budget in passes, which may be fractional
    when an explicit iteration budget is used.
    """
    n = _positive_int(n_samples, "n_samples")
    k = _positive_int(n_clusters, "n_clusters")
    batch = _positive_int(batch_size, "batch_size")
    if k > n or batch > n or not np.isfinite(epochs) or epochs <= 0:
        raise ValueError("Require k <= n, B <= n and a finite T > 0.")
    peak = 0.824 * k**-0.46 * batch**0.90 * n**-0.45 * epochs**-0.11
    return min(peak, 1.0)


class EMeans:
    """Cluster dense NumPy data with cosine-scheduled mini-batch EMA.

    Set batch_size or alpha_0 to None to apply the paper's scaling laws.
    max_epochs caps passes over freshly shuffled data. total_iterations
    can stop earlier. Random seeding is uniform without replacement.
    init also accepts "k-means++" or a (k, d) array of fixed seeds.
    finalization accepts "last_epoch_mean" or "ema". Set device to
    "cpu" or a PyTorch CUDA device string to use torch.cdist; None uses
    NumPy. NumPy state and returned distances use float64.
    """

    def __init__(
        self,
        n_clusters: int,
        batch_size: int | None = None,
        max_epochs: int = 10,
        alpha_0: float | None = None,
        epsilon: float = 0.1,
        random_state: int | None = None,
        *,
        init: str | np.ndarray = "random",
        total_iterations: int | None = None,
        finalization: str = "last_epoch_mean",
        device: str | None = None,
    ) -> None:
        """Store training parameters and validate their ranges."""
        self.n_clusters = _positive_int(n_clusters, "n_clusters")
        if batch_size is not None:
            batch_size = _positive_int(batch_size, "batch_size")
        self.batch_size = batch_size
        self.max_epochs = _positive_int(max_epochs, "max_epochs")
        if alpha_0 is not None and not 0 < alpha_0 <= 1:
            raise ValueError("alpha_0 must be in (0, 1].")
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be in [0, 1].")
        if isinstance(init, str):
            if init not in ("random", "k-means++"):
                raise ValueError("Unknown initialization method.")
        elif not isinstance(init, np.ndarray):
            raise TypeError("init must be a method name or a NumPy array.")
        if finalization not in ("last_epoch_mean", "ema"):
            raise ValueError("Unknown finalization method.")
        if total_iterations is not None:
            total_iterations = _positive_int(
                total_iterations, "total_iterations"
            )
        self.alpha_0 = alpha_0
        self.epsilon = epsilon
        self.random_state = random_state
        self.init = init
        self.total_iterations = total_iterations
        self.finalization = finalization
        self.device = device
        self._torch = None
        self._fitted = False

    def fit(self, X: np.ndarray) -> EMeans:
        """Fit on finite (n, d) data and store centers and training metrics.

        loss_history_ records pre-update batch MSE. labels_ and inertia_
        are evaluated against the returned centers after finalization.
        """
        X = self._validate_input(X)
        n_samples, n_features = X.shape
        if n_samples < self.n_clusters:
            raise ValueError("Need at least n_clusters training rows.")
        self._fitted = False
        self.n_features_in_ = n_features
        self._rng = np.random.default_rng(self.random_state)
        if self.device is not None:
            try:
                import torch
            except ImportError as error:
                raise ImportError("Install PyTorch to set device.") from error
            self._torch = torch
        self.batch_size_ = self.batch_size or suggest_batch_size(
            n_samples, self.n_clusters, self.max_epochs
        )
        if self.batch_size_ > n_samples:
            raise ValueError("batch_size cannot exceed the training size.")
        batches_per_epoch = n_samples // self.batch_size_
        budget = self.max_epochs * batches_per_epoch
        self.total_iterations_ = self.total_iterations or budget
        if self.total_iterations_ > budget:
            raise ValueError("total_iterations exceeds the max_epochs budget.")
        self.alpha_0_ = self.alpha_0 or suggest_alpha_0(
            n_samples,
            self.n_clusters,
            self.batch_size_,
            self.total_iterations_ / batches_per_epoch,
        )

        self.cluster_centers_ = self._initialize_centroids(X)
        self.ema_counts_ = np.full(self.n_clusters, INITIAL_COUNT)
        self.ema_sums_ = self.ema_counts_[:, None] * self.cluster_centers_
        self.loss_history_ = np.empty(self.total_iterations_, dtype=float)
        self.n_iter_ = 0
        self.n_revivals_ = 0
        self._train(X, batches_per_epoch)

        self.labels_, self.inertia_ = self._predict_and_score(X)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the nearest center index for each input row."""
        X = self._validate_prediction(X)
        labels = np.empty(len(X), dtype=np.intp)
        for rows in self._row_slices(len(X)):
            labels[rows], _ = self._assign_clusters(X[rows])
        return labels

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit on X and return labels using the final fitted centers."""
        return self.fit(X).labels_.copy()

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return an (n, k) array of Euclidean distances to all centers."""
        X = self._validate_prediction(X)
        distances = np.empty((len(X), self.n_clusters), dtype=float)
        for rows in self._row_slices(len(X)):
            for cols, squared in self._distance_blocks(X[rows]):
                np.sqrt(squared, out=distances[rows, cols])
        return distances

    @staticmethod
    def _row_slices(
        n_rows: int, size: int = ROW_BLOCK_SIZE
    ) -> Iterator[slice]:
        """Yield contiguous slices of up to size rows."""
        for start in range(0, n_rows, size):
            yield slice(start, min(start + size, n_rows))

    def _validate_input(self, X: np.ndarray) -> np.ndarray:
        """Check finite real data in chunks, preserving its storage."""
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] == 0 or X.dtype.kind not in "fiu":
            raise ValueError("X must be a real numeric 2D array with columns.")
        for rows in self._row_slices(len(X)):
            if not np.isfinite(X[rows]).all():
                raise ValueError("X contains NaN or infinity.")
        return X

    def _validate_prediction(self, X: np.ndarray) -> np.ndarray:
        """Check that a prediction input matches the fitted feature count."""
        if not self._fitted:
            raise RuntimeError("Call fit before prediction.")
        X = self._validate_input(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Feature count differs from fit input.")
        return X

    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """Return k initial centroids, shape (k, d), from the chosen init."""
        if isinstance(self.init, np.ndarray):
            centers = self._validate_input(self.init)
            if centers.shape != (self.n_clusters, X.shape[1]):
                raise ValueError("init must have shape (n_clusters, d).")
            return centers.astype(float, copy=True)
        if self.init == "k-means++":
            return self._kmeans_plus_plus(X)
        indices = self._rng.choice(len(X), self.n_clusters, replace=False)
        return X[indices].astype(float, copy=True)

    def _kmeans_plus_plus(self, X: np.ndarray) -> np.ndarray:
        """Sample seeds by squared distance to the nearest chosen seed."""
        indices = np.empty(self.n_clusters, dtype=np.intp)
        nearest = np.full(len(X), np.inf)
        indices[0] = self._rng.integers(len(X))
        for step in range(1, self.n_clusters):
            center = X[indices[step - 1]].astype(float)[None, :]
            for rows in self._row_slices(len(X)):
                squared = self._squared_distances(X[rows], center)[:, 0]
                nearest[rows] = np.minimum(nearest[rows], squared)
            peak = nearest.max()
            if peak > 0:
                weights = nearest / peak
                indices[step] = self._rng.choice(
                    len(X), p=weights / weights.sum()
                )
            else:
                available = np.ones(len(X), dtype=bool)
                available[indices[:step]] = False
                indices[step] = self._rng.choice(np.flatnonzero(available))
        return X[indices].astype(float, copy=True)

    def _batches(self, X: np.ndarray) -> Iterator[np.ndarray]:
        """Yield full batches from a fresh shuffle, omitting any remainder."""
        indices = self._rng.permutation(len(X))
        limit = len(X) - len(X) % self.batch_size_
        for start in range(0, limit, self.batch_size_):
            yield X[indices[start:start + self.batch_size_]].astype(
                float, copy=False
            )

    def _squared_distances(
        self, X: np.ndarray, centers: np.ndarray
    ) -> np.ndarray:
        """Compute a block of squared distances by direct subtraction."""
        X = np.ascontiguousarray(X, dtype=float)
        if self._torch is not None:
            torch = self._torch
            with torch.no_grad():
                samples = torch.tensor(X, device=self.device)
                targets = torch.as_tensor(centers, device=self.device)
                squared = torch.cdist(
                    samples, targets,
                    compute_mode="donot_use_mm_for_euclid_dist",
                ).square().cpu().numpy()
        else:
            squared = np.empty((len(X), len(centers)), dtype=float)
            bytes_per_row = 8 * len(centers) * centers.shape[1]
            row_limit = max(1, DISTANCE_BLOCK_BYTES // bytes_per_row)
            with np.errstate(over="ignore", invalid="ignore"):
                for rows in self._row_slices(len(X), row_limit):
                    differences = X[rows, None, :] - centers[None, :, :]
                    squared[rows] = np.einsum(
                        "ijk,ijk->ij", differences, differences
                    )
        if not np.isfinite(squared).all():
            raise FloatingPointError(
                "Squared distances overflowed; rescale input features."
            )
        return squared

    def _distance_blocks(
        self, X: np.ndarray
    ) -> Iterator[tuple[slice, np.ndarray]]:
        """Yield centroid column slices and squared distance blocks."""
        for cols in self._row_slices(self.n_clusters, CENTROID_BLOCK_SIZE):
            yield cols, self._squared_distances(X, self.cluster_centers_[cols])

    def _assign_clusters(
        self, X_batch: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return nearest centroid indices and their squared distances."""
        labels = np.zeros(len(X_batch), dtype=np.intp)
        distances = np.full(len(X_batch), np.inf)
        for rows in self._row_slices(len(X_batch)):
            nearest = distances[rows]
            assigned = labels[rows]
            indices = np.arange(len(nearest))
            for cols, squared in self._distance_blocks(X_batch[rows]):
                local = squared.argmin(axis=1)
                candidate = squared[indices, local]
                improved = candidate < nearest
                nearest[improved] = candidate[improved]
                assigned[improved] = cols.start + local[improved]
        return labels, distances

    def _batch_statistics(
        self, X_batch: np.ndarray, labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return per-cluster batch counts n_j and vector sums s_j."""
        counts = np.bincount(labels, minlength=self.n_clusters)
        sums = np.zeros_like(self.ema_sums_)
        with np.errstate(over="raise", invalid="raise"):
            np.add.at(sums, labels, X_batch)
        return counts, sums

    def _cosine_schedule(self, iteration: int) -> tuple[float, float]:
        """Return (alpha, gamma) for iteration t, using Eq. 5."""
        progress = iteration / self.total_iterations_
        cosine = (1 + cos(pi * progress)) / 2
        alpha = self.alpha_0_ * (self.epsilon + (1 - self.epsilon) * cosine)
        return alpha, 1 - alpha

    def _update_centroids(
        self, X_batch: np.ndarray, labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Update EMA state and centers, returning batch counts and sums."""
        counts, sums = self._batch_statistics(X_batch, labels)
        alpha, gamma = self._cosine_schedule(self.n_iter_)
        with np.errstate(over="raise", invalid="raise"):
            self.ema_counts_ = gamma * self.ema_counts_ + alpha * counts
            self.ema_sums_ = gamma * self.ema_sums_ + alpha * sums
            positive = self.ema_counts_ > 0
            np.divide(
                self.ema_sums_,
                self.ema_counts_[:, None],
                out=self.cluster_centers_,
                where=positive[:, None],
            )
        return counts, sums

    def _revive_dead_clusters(self) -> int:
        """Split distinct eligible donors and return the revival count.

        Draw donors with probability proportional to their EMA counts.
        Defer any remaining weak clusters if eligible donors run out.
        """
        weak = self.ema_counts_ < REVIVAL_THRESHOLD * self.ema_counts_.mean()
        if not weak.any():
            return 0
        viable = (
            ~weak
            & (self.ema_counts_ / 2 > 0)
            & np.any(self.cluster_centers_ != 0, axis=1)
        )
        donors = np.flatnonzero(viable)
        recipients = np.flatnonzero(weak)
        n_splits = min(len(donors), len(recipients))
        if not n_splits:
            return 0
        weights = self.ema_counts_[donors]
        donors = self._rng.choice(
            donors, size=n_splits, replace=False, p=weights / weights.sum()
        )
        recipients = recipients[:n_splits]
        delta = np.full(self.n_features_in_, BFLOAT16_EPSILON)
        delta[1::2] *= -1
        parent = self.cluster_centers_[donors].copy()
        half_counts = self.ema_counts_[donors] / 2
        with np.errstate(over="raise", invalid="raise"):
            self.cluster_centers_[recipients] = parent * (1 + delta)
            self.cluster_centers_[donors] = parent * (1 - delta)
            for indices in (donors, recipients):
                self.ema_counts_[indices] = half_counts
                self.ema_sums_[indices] = (
                    half_counts[:, None] * self.cluster_centers_[indices]
                )
        return n_splits

    def _train(self, X: np.ndarray, batches_per_epoch: int) -> None:
        """Run the update budget and optionally finalize on the last pass."""
        epochs = 1 + (self.total_iterations_ - 1) // batches_per_epoch
        close_mean = self.finalization == "last_epoch_mean"
        for epoch in range(epochs):
            final_pass = close_mean and epoch == epochs - 1
            if final_pass:
                final_counts = np.zeros(self.n_clusters, dtype=np.int64)
                final_sums = np.zeros_like(self.ema_sums_)
            for batch in self._batches(X):
                labels, squared = self._assign_clusters(batch)
                self.loss_history_[self.n_iter_] = np.sum(squared / len(batch))
                counts, sums = self._update_centroids(batch, labels)
                self.n_iter_ += 1
                if final_pass:
                    final_counts += counts
                    with np.errstate(over="raise", invalid="raise"):
                        final_sums += sums
                if self.n_iter_ % REVIVAL_INTERVAL == 0:
                    self.n_revivals_ += self._revive_dead_clusters()
                if self.n_iter_ == self.total_iterations_:
                    break
        if close_mean:
            np.divide(
                final_sums,
                final_counts[:, None],
                out=self.cluster_centers_,
                where=final_counts[:, None] > 0,
            )

    def _predict_and_score(self, X: np.ndarray) -> tuple[np.ndarray, float]:
        """Return full-data assignments and summed squared distances."""
        labels = np.empty(len(X), dtype=np.intp)
        inertia = 0.0
        with np.errstate(over="raise"):
            for rows in self._row_slices(len(X)):
                labels[rows], squared = self._assign_clusters(X[rows])
                inertia += squared.sum()
        return labels, float(inertia)
