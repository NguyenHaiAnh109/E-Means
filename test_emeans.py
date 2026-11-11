"""Regression tests for e-means formulas and edge cases."""

import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

import numpy as np

from emeans import EMeans, suggest_alpha_0, suggest_batch_size


class EMeansTests(unittest.TestCase):
    """Test individual steps and public estimator methods."""

    def test_schedule_uses_zero_based_equation(self) -> None:
        model = EMeans(1, 1, alpha_0=0.8, epsilon=0.1)
        model.alpha_0_ = 0.8
        model.total_iterations_ = 4
        np.testing.assert_allclose(model._cosine_schedule(0), (0.8, 0.2))
        np.testing.assert_allclose(model._cosine_schedule(2), (0.44, 0.56))
        np.testing.assert_allclose(model._cosine_schedule(4), (0.08, 0.92))
        self.assertGreater(model._cosine_schedule(3)[0], 0.08)

    def test_ema_matches_hand_calculation(self) -> None:
        X = np.array([[2.0], [4.0], [10.0]])
        model = EMeans(2, 3, 1, 0.25, finalization="ema").fit(X)
        model.ema_counts_[:] = [4, 2]
        model.ema_sums_[:] = [[24], [20]]
        model.n_iter_ = 0
        counts, sums = model._update_centroids(X, np.array([0, 0, 1]))
        np.testing.assert_array_equal(counts, [2, 1])
        np.testing.assert_array_equal(sums, [[6], [10]])
        np.testing.assert_allclose(model.ema_counts_, [3.5, 1.75])
        np.testing.assert_allclose(model.ema_sums_, [[19.5], [17.5]])
        np.testing.assert_allclose(
            model.cluster_centers_, [[19.5 / 3.5], [10]]
        )

    def test_alpha_one_and_empty_cluster_keep_finite_center(self) -> None:
        X = np.zeros((4, 1))
        model = EMeans(
            2, 4, 1, 1.0, init=np.array([[0.0], [9.0]])
        ).fit(X)
        np.testing.assert_array_equal(model.ema_counts_, [4, 0])
        np.testing.assert_array_equal(model.cluster_centers_, [[0], [9]])
        np.testing.assert_array_equal(model.labels_, np.zeros(4))

    def test_revival_matches_split_and_defers_extra_recipients(self) -> None:
        X = np.array([[4.0, -2.0], [2.0, -4.0], [0.0, 0.0]])
        model = EMeans(3, 3, 1, random_state=7).fit(X)
        model.cluster_centers_[:] = X
        model.ema_counts_[:] = [8, 0, 0]
        model.ema_sums_[:] = model.ema_counts_[:, None] * X
        self.assertEqual(model._revive_dead_clusters(), 1)
        np.testing.assert_allclose(model.ema_counts_, [4, 4, 0])
        expected = np.array([
            [4 * (1 - 1 / 128), -2 * (1 + 1 / 128)],
            [4 * (1 + 1 / 128), -2 * (1 - 1 / 128)],
            [0, 0],
        ])
        np.testing.assert_allclose(model.cluster_centers_, expected)
        np.testing.assert_allclose(
            model.ema_sums_, expected * np.array([[4], [4], [0]])
        )

    def test_revival_every_fifth_batch_and_no_viable_donor(self) -> None:
        X = np.zeros((12, 2))

        class RecordingEMeans(EMeans):
            """Record the completed update numbers that trigger revival."""

            def _revive_dead_clusters(self) -> int:
                calls.append(self.n_iter_)
                return super()._revive_dead_clusters()

        calls = []
        model = RecordingEMeans(3, 1, 1, 1.0, random_state=0).fit(X)
        self.assertEqual(calls, [5, 10])
        self.assertEqual(model.n_revivals_, 0)
        self.assertEqual(model.inertia_, 0.0)

    def test_shuffled_full_batches_and_iteration_override(self) -> None:
        X = np.arange(11).reshape(-1, 1)
        model = EMeans(2, 4, 2, 0.3, random_state=3, total_iterations=3)
        model.fit(X)
        batches = list(model._batches(X))
        self.assertEqual([len(batch) for batch in batches], [4, 4])
        self.assertEqual(np.unique(np.concatenate(batches)).size, 8)
        self.assertEqual(model.n_iter_, 3)
        self.assertEqual(len(model.loss_history_), 3)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            EMeans(2, 4, 1, total_iterations=3).fit(X)

    def test_last_epoch_mean_uses_training_assignments(self) -> None:
        X = np.array([[0.0], [2.0], [5.0], [19.0], [20.0], [25.0]])
        model = EMeans(
            1, 2, 1, 0.8, random_state=3, finalization="last_epoch_mean"
        ).fit(X)
        np.testing.assert_allclose(model.cluster_centers_, [[X.mean()]])
        self.assertFalse(np.isclose(
            model.ema_sums_[0, 0] / model.ema_counts_[0], X.mean()
        ))

    def test_public_methods_and_reproducibility(self) -> None:
        rng = np.random.default_rng(19)
        X = rng.normal(size=(77, 3))
        for init in ("random", "k-means++"):
            model = EMeans(4, 9, 3, 0.4, random_state=2, init=init)
            labels = model.fit_predict(X)
            centers = model.cluster_centers_.copy()
            history = model.loss_history_.copy()
            distances = model.transform(X)
            expected = np.linalg.norm(X[:, None] - centers, axis=2)
            np.testing.assert_allclose(distances, expected)
            np.testing.assert_array_equal(labels, distances.argmin(axis=1))
            np.testing.assert_array_equal(labels, model.predict(X))
            nearest = expected.min(axis=1)
            self.assertAlmostEqual(model.inertia_, nearest @ nearest)
            np.testing.assert_array_equal(labels, model.fit_predict(X))
            np.testing.assert_array_equal(centers, model.cluster_centers_)
            np.testing.assert_array_equal(history, model.loss_history_)

    def test_memmap_and_centroid_blocks(self) -> None:
        X = np.arange(300, dtype=np.float32).reshape(-1, 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.npy"
            np.save(path, X)
            memory_map = np.load(path, mmap_mode="r")
            model = EMeans(300, 300, 1, 1.0, init=X).fit(memory_map)
            np.testing.assert_array_equal(model.predict(memory_map), X[:, 0])
            self.assertEqual(model.transform(memory_map).shape, (300, 300))

    def test_invalid_inputs(self) -> None:
        model = EMeans(2, 2, 1)
        with self.assertRaises(RuntimeError):
            model.predict(np.zeros((2, 1)))
        for invalid in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaisesRegex(ValueError, "NaN or infinity"):
                model.fit(np.array([[0.0], [1.0], [invalid]]))
        with self.assertRaises(ValueError):
            model.fit(np.zeros((1, 1)))
        model.fit(np.zeros((2, 1)))
        with self.assertRaisesRegex(ValueError, "Feature count"):
            model.transform(np.zeros((2, 2)))

    def test_scaling_laws(self) -> None:
        self.assertEqual(suggest_batch_size(1, 1, 1), 1)
        self.assertEqual(suggest_batch_size(4800, 4, 10), 512)
        self.assertEqual(suggest_batch_size(10_000_000, 1024, 4), 65536)
        self.assertEqual(suggest_alpha_0(100, 1, 100, 1), 1.0)
        expected = 0.824 * 4**-0.46 * 64**0.90 * 4800**-0.45 * 10**-0.11
        self.assertAlmostEqual(suggest_alpha_0(4800, 4, 64, 10), expected)
        with self.assertRaises(ValueError):
            suggest_alpha_0(10, 2, 11, 3)

    @unittest.skipUnless(find_spec("torch"), "PyTorch is not installed")
    def test_torch_matches_numpy(self) -> None:
        X = np.random.default_rng(11).normal(size=(80, 3))
        numpy_model = EMeans(3, 10, 2, 0.4, random_state=42).fit(X)
        torch_model = EMeans(
            3, 10, 2, 0.4, random_state=42, device="cpu"
        ).fit(X)
        np.testing.assert_array_equal(torch_model.labels_, numpy_model.labels_)
        np.testing.assert_allclose(
            torch_model.transform(X), numpy_model.transform(X)
        )


if __name__ == "__main__":
    unittest.main()
