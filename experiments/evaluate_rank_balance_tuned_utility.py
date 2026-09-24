#!/usr/bin/env python3
"""Evaluate matched RankBalance variants on decisive held-out preferences."""

from __future__ import annotations
import _bootstrap  # noqa: F401

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rank_balance_ablation import (  # noqa: E402
    decisive_preference_metrics,
    fit_rank_balance,
)
from run_pollution_experiment import _load_data  # noqa: E402
from run_rank_balance_ablation import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_DATASETS,
    _load_datasets,
    _split_data,
)

DEFAULT_RESULTS = (
    PROJECT_ROOT / "experiment_results" / "rank_balance_tuned_utility"
)
DEFAULT_GAMMA_PATH = (
    PROJECT_ROOT
    / "experiment_results"
    / "rank_balance_gamma_validation"
    / "selected_gamma.json"
)
VARIANT_LABELS = {
    "matched_gamma0": "Matched baseline: gamma=0",
    "delete_only_tuned": "Delete-only",
    "full_tuned": "Full RankBalance",
}


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _configs(gamma, args):
    common = {
        "lambda_theta": args.lambda_theta,
        "lambda_u": args.lambda_u,
        "mu": args.mu,
        "max_iter": args.max_iter,
        "tol": args.tol,
    }
    return {
        "matched_gamma0": {**common, "gamma": 0.0, "regularizer": "full"},
        "delete_only_tuned": {
            **common,
            "gamma": gamma,
            "regularizer": "delete_only",
        },
        "full_tuned": {**common, "gamma": gamma, "regularizer": "full"},
    }


def run_dataset(dataset_id, settings, gamma, args):
    output_path = args.results_root / "raw" / f"{dataset_id}.csv"
    metadata_path = args.results_root / "raw" / f"{dataset_id}.json"
    data_path = Path(settings["csv"])
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    stat = data_path.stat()
    configs = _configs(gamma, args)
    identity = {
        "protocol": "decisive-heldout-utility-v1",
        "dataset_id": dataset_id,
        "path": str(data_path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "seeds": args.seeds,
        "configs": configs,
    }
    if not args.overwrite and output_path.exists() and metadata_path.exists():
        if json.loads(metadata_path.read_text(encoding="utf-8")) == identity:
            print(f"Resume utility: {dataset_id} already complete", flush=True)
            return

    frame, _ = _load_data(data_path)
    tasks = []
    for seed in args.seeds:
        train, _, test, _ = _split_data(frame, seed)
        models = sorted(
            set(train["methodA"].astype(str))
            | set(train["methodB"].astype(str))
        )
        raters = sorted(train["answerer"].astype(str).unique())
        for variant, config in configs.items():
            tasks.append((seed, train, test, models, raters, variant, config))

    def evaluate(task):
        seed, train, test, models, raters, variant, config = task
        fitted = fit_rank_balance(train, config, models=models, raters=raters)
        metrics = decisive_preference_metrics(test, fitted)
        return {
            "dataset_id": dataset_id,
            "seed": seed,
            "variant": variant,
            "configuration": VARIANT_LABELS[variant],
            "gamma": config["gamma"],
            "test_nll": metrics["nll"],
            "test_accuracy": metrics["accuracy"],
            "test_rows": metrics["rows"],
            "fit_success": fitted["success"],
            "time_seconds": fitted["elapsed"],
        }

    with ThreadPoolExecutor(
        max_workers=min(args.fit_workers, len(tasks))
    ) as executor:
        rows = list(executor.map(evaluate, tasks))
    _write_csv(output_path, pd.DataFrame(rows))
    _write_json(metadata_path, identity)


def _paired_bootstrap(dataset_matrix, seed, repetitions):
    values = dataset_matrix.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(values), size=(repetitions, len(values))
    )
    samples = values[indices].mean(axis=1)
    return (
        values.mean(axis=0),
        np.quantile(samples, 0.025, axis=0),
        np.quantile(samples, 0.975, axis=0),
    )


def aggregate(root, bootstrap_repetitions, bootstrap_seed):
    paths = sorted((root / "raw").glob("*.csv"))
    if not paths:
        raise RuntimeError(f"No utility results found in {root}")
    runs = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    if not runs["fit_success"].astype(bool).all():
        raise RuntimeError("At least one utility fit did not converge")
    _write_csv(root / "utility_runs.csv", runs)
    dataset_level = (
        runs.groupby(["dataset_id", "variant", "configuration", "gamma"], as_index=False)[
            ["test_nll", "test_accuracy"]
        ]
        .mean()
    )
    _write_csv(root / "utility_dataset_level.csv", dataset_level)
    variants = list(VARIANT_LABELS)
    rows = []
    for metric_index, metric in enumerate(["test_nll", "test_accuracy"]):
        matrix = (
            dataset_level.pivot(
                index="dataset_id", columns="variant", values=metric
            )
            .reindex(columns=variants)
            .dropna(how="any")
        )
        mean, lower, upper = _paired_bootstrap(
            matrix,
            bootstrap_seed + metric_index,
            bootstrap_repetitions,
        )
        for index, variant in enumerate(variants):
            rows.append(
                {
                    "variant": variant,
                    "configuration": VARIANT_LABELS[variant],
                    "metric": metric,
                    "mean": mean[index],
                    "ci_lower": lower[index],
                    "ci_upper": upper[index],
                    "datasets": len(matrix),
                }
            )
    statistics = pd.DataFrame(rows)
    _write_csv(root / "utility_statistics.csv", statistics)
    print(statistics.to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--gamma-path", type=Path, default=DEFAULT_GAMMA_PATH)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 2023, 3407, 7919, 15401])
    parser.add_argument("--fit-workers", type=int, default=8)
    parser.add_argument("--lambda-theta", type=float, default=1.0)
    parser.add_argument("--lambda-u", type=float, default=1.0)
    parser.add_argument("--mu", type=float, default=2.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260916)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--no-aggregate", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.results_root = args.results_root.resolve()
    args.results_root.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        aggregate(
            args.results_root,
            args.bootstrap_repetitions,
            args.bootstrap_seed,
        )
        return
    selected = json.loads(args.gamma_path.resolve().read_text(encoding="utf-8"))
    gamma = float(selected["gamma"])
    definitions = _load_datasets(args.config.resolve())
    for dataset_id in args.datasets:
        if dataset_id == "ihq_all":
            raise ValueError("IHQ-all is excluded from this experiment")
        run_dataset(dataset_id, definitions[dataset_id], gamma, args)
    if not args.no_aggregate:
        aggregate(
            args.results_root,
            args.bootstrap_repetitions,
            args.bootstrap_seed,
        )


if __name__ == "__main__":
    main()
