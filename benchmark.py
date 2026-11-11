"""Compare e-means with scikit-learn on a reproducible blobs benchmark."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from time import perf_counter

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from emeans import EMeans, suggest_alpha_0, suggest_batch_size

matplotlib.use("Agg")

MODELS = ("EMeans", "KMeans", "MiniBatchKMeans")
CLUSTER_COUNTS = (3, 4, 6)
SEEDS = (11, 21, 31)
DATA_SEED = 42
EPOCHS = 10
COLORS = ("#22796b", "#5955a5", "#b35325")
METRICS = (
    "train_inertia",
    "test_mse",
    "test_ari",
    "test_silhouette",
    "fit_seconds",
)


def build_models(n: int, seeds: np.ndarray, seed: int) -> dict:
    """Construct models with the same initial centers and epoch cap."""
    k = len(seeds)
    batch_size = suggest_batch_size(n, k, EPOCHS)
    alpha_0 = suggest_alpha_0(n, k, batch_size, EPOCHS)
    return {
        "EMeans": EMeans(
            k, batch_size, EPOCHS, alpha_0, 0.1, seed, init=seeds
        ),
        "KMeans": KMeans(
            k, init=seeds, n_init=1, max_iter=EPOCHS, tol=0,
            algorithm="lloyd", random_state=seed,
        ),
        "MiniBatchKMeans": MiniBatchKMeans(
            k, init=seeds, n_init=1, max_iter=EPOCHS, tol=0,
            batch_size=batch_size, max_no_improvement=None,
            random_state=seed,
        ),
    }


def evaluate(
    model: EMeans | KMeans | MiniBatchKMeans,
    X_train: np.ndarray,
    X_test: np.ndarray,
    truth_test: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    """Time fit and return training inertia and holdout quality metrics."""
    start = perf_counter()
    model.fit(X_train)
    fit_seconds = perf_counter() - start
    train_labels = model.predict(X_train)
    test_labels = model.predict(X_test)
    centers = model.cluster_centers_
    train_error = X_train - centers[train_labels]
    test_error = X_test - centers[test_labels]
    unique = len(np.unique(test_labels))
    silhouette = (
        silhouette_score(X_test, test_labels)
        if 1 < unique < len(X_test)
        else float("nan")
    )
    return {
        "train_inertia": float(np.square(train_error).sum()),
        "test_mse": float(np.square(test_error).sum(axis=1).mean()),
        "test_ari": adjusted_rand_score(truth_test, test_labels),
        "test_silhouette": silhouette,
        "fit_seconds": fit_seconds,
    }, test_labels


def write_metrics(path: Path, rows: list[dict]) -> None:
    """Write a sequence of metric records to CSV with headers."""
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(records: list[dict]) -> list[dict]:
    """Return median results by model and k, ignoring undefined metrics."""
    summary = []
    for k in CLUSTER_COUNTS:
        for name in MODELS:
            selected = [
                row for row in records
                if row["k"] == k and row["model"] == name
            ]
            row = {"k": k, "model": name}
            for metric in METRICS:
                values = np.array([run[metric] for run in selected])
                values = values[np.isfinite(values)]
                row[metric] = (
                    float(np.median(values)) if len(values) else float("nan")
                )
            summary.append(row)
    return summary


def plot_assignments(
    X: np.ndarray,
    truth: np.ndarray,
    X_test: np.ndarray,
    assignments: dict,
    path: Path,
) -> None:
    """Plot the original blobs and every fitted model on the holdout."""
    figure, axes = plt.subplots(
        1, 2, figsize=(10, 4), sharex=True, sharey=True,
        layout="constrained",
    )
    axes[0].scatter(X[:, 0], X[:, 1], c=truth, s=5, cmap="tab10",
                    alpha=0.6)
    axes[0].set_title("Original dataset • true labels")
    axes[1].scatter(X[:, 0], X[:, 1], s=4, color="#d1d7df")
    axes[1].scatter(
        X_test[:, 0], X_test[:, 1], s=7, color="#22796b", alpha=0.7
    )
    axes[1].set_title("Held-out points in green")
    for axis in axes:
        axis.set_xlabel("Feature 1")
    axes[0].set_ylabel("Feature 2")
    figure.savefig(path / "dataset.png", dpi=150)
    plt.close(figure)

    figure, axes = plt.subplots(
        len(CLUSTER_COUNTS), len(MODELS), figsize=(11, 9),
        sharex=True, sharey=True, layout="constrained",
    )
    for i, k in enumerate(CLUSTER_COUNTS):
        for j, name in enumerate(MODELS):
            axis = axes[i, j]
            labels, centers = assignments[k, name]
            axis.scatter(
                X_test[:, 0], X_test[:, 1], c=labels, s=6, cmap="tab10",
                alpha=0.65, vmin=0, vmax=9,
            )
            axis.scatter(
                centers[:, 0], centers[:, 1], marker="X", color="white",
                edgecolors="black", linewidths=1, s=80, zorder=3,
            )
            if i == 0:
                axis.set_title(name)
            if j == 0:
                axis.set_ylabel(f"k = {k}\nFeature 2")
            if i == len(CLUSTER_COUNTS) - 1:
                axis.set_xlabel("Feature 1")
    figure.suptitle(
        f"Holdout assignments and fitted centroids • seed {SEEDS[0]}"
    )
    figure.savefig(path / "clusters.png", dpi=150)
    plt.close(figure)


def plot_training(histories: dict, summary: list[dict], path: Path) -> None:
    """Plot e-means pre-update batch loss and median fit times."""
    figure, (loss_ax, time_ax) = plt.subplots(
        1, 2, figsize=(11, 4), layout="constrained"
    )
    for k, color in zip(CLUSTER_COUNTS, COLORS):
        history = histories[k]
        loss_ax.plot(
            np.arange(1, len(history) + 1), history,
            color=color, alpha=0.8, linewidth=1, label=f"k = {k}",
        )
    loss_ax.set(
        title=f"E-means pre-update batch MSE • seed {SEEDS[0]}",
        xlabel="Update number",
        ylabel="Squared distance / sample",
    )
    loss_ax.legend(frameon=False)
    positions = np.arange(len(CLUSTER_COUNTS))
    for i, (name, color) in enumerate(zip(MODELS, COLORS)):
        seconds = [
            row["fit_seconds"] for row in summary if row["model"] == name
        ]
        time_ax.bar(positions + (i - 1) * 0.25, seconds, 0.25,
                    color=color, label=name)
    time_ax.set_xticks(positions, [str(k) for k in CLUSTER_COUNTS])
    time_ax.set(
        title="Median fit time • three seeds, one CPU thread",
        xlabel="Number of clusters",
        ylabel="Seconds",
    )
    time_ax.legend(frameon=False)
    for axis in (loss_ax, time_ax):
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    figure.savefig(path / "training.png", dpi=150)
    plt.close(figure)


def run(output: Path) -> None:
    """Generate data, fit all configurations and write figures and CSVs."""
    X, truth = make_blobs(
        n_samples=6000, centers=[(-4, -3), (-3, 3), (2, 2), (5, -3)],
        cluster_std=[0.8, 1.15, 1.25, 0.9], random_state=DATA_SEED,
    )
    X_train, X_test, _, truth_test = train_test_split(
        X, truth, test_size=0.2, stratify=truth, random_state=DATA_SEED,
    )
    output.mkdir(parents=True, exist_ok=True)
    records, assignments, histories = [], {}, {}
    with threadpool_limits(limits=1):
        for k in CLUSTER_COUNTS:
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                seeds = X_train[rng.choice(len(X_train), k, replace=False)]
                models = build_models(len(X_train), seeds, seed)
                for name, model in models.items():
                    metrics, labels = evaluate(
                        model, X_train, X_test, truth_test
                    )
                    e_means = models["EMeans"]
                    records.append({
                        "k": k, "model": name, "seed": seed,
                        "batch_size": (
                            "" if name == "KMeans" else e_means.batch_size
                        ),
                        "alpha_0": (
                            e_means.alpha_0 if name == "EMeans" else ""
                        ),
                        **metrics,
                    })
                    if seed == SEEDS[0]:
                        assignments[k, name] = labels, model.cluster_centers_
                        if name == "EMeans":
                            histories[k] = model.loss_history_
    summary = summarize(records)
    write_metrics(output / "metrics_by_seed.csv", records)
    write_metrics(output / "metrics_median.csv", summary)
    plot_assignments(X, truth, X_test, assignments, output)
    plot_training(histories, summary, output)


def main() -> None:
    """Run the benchmark and print the results directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    run(args.output)
    print(f"Benchmark written to {args.output}")


if __name__ == "__main__":
    main()
