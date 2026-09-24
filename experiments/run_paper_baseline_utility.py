#!/usr/bin/env python3
"""Held-out decisive NLL/accuracy for the five paper baselines."""

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

from run_convergence_experiment import METHODS  # noqa: E402
from run_heldout_prediction import _fit_and_predict  # noqa: E402
from run_pollution_experiment import _load_data  # noqa: E402
from run_rank_balance_ablation import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_DATASETS,
    _load_datasets,
    _split_data,
)

DEFAULT_RESULTS = (
    PROJECT_ROOT / "experiment_results" / "paper_baseline_utility"
)
RANK_BALANCE_UTILITY = (
    PROJECT_ROOT
    / "experiment_results"
    / "rank_balance_tuned_utility"
    / "utility_dataset_level.csv"
)
PAPER_METHODS = ["m_elo", "crowd_bt", "bayes_bt", "bbq", "am_elo"]
METHOD_LABELS = {
    "m_elo": "ELO",
    "crowd_bt": "Crowd-BT",
    "bayes_bt": "Bayes-BT",
    "bbq": "BBQ",
    "am_elo": "am-ELO",
    "rank_balance": "RankBalance",
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


def run_dataset(dataset_id, settings, args):
    output_path = args.results_root / "raw" / f"{dataset_id}.csv"
    metadata_path = args.results_root / "raw" / f"{dataset_id}.json"
    data_path = Path(settings["csv"])
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    stat = data_path.stat()
    identity = {
        "protocol": "paper-baseline-decisive-utility-v1",
        "dataset_id": dataset_id,
        "path": str(data_path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "seeds": args.seeds,
        "methods": PAPER_METHODS,
    }
    if not args.overwrite and output_path.exists() and metadata_path.exists():
        if json.loads(metadata_path.read_text(encoding="utf-8")) == identity:
            print(f"Resume baseline utility: {dataset_id} complete", flush=True)
            return

    frame, _ = _load_data(data_path)
    tasks = []
    for seed in args.seeds:
        train, _, test, _ = _split_data(frame, seed)
        test = test.copy()
        test["__comparison_group"] = "__row_" + test["__row_id"].astype(str)
        test["__fold"] = int(seed)
        test["__subgroup"] = "all"
        for method_id in PAPER_METHODS:
            tasks.append((seed, method_id, train, test))

    def evaluate(task):
        seed, method_id, train, test = task
        # Estimators may normalize or add columns internally. Isolate each
        # task because Pandas DataFrames are not thread-safe under mutation.
        train = train.copy(deep=True)
        test = test.copy(deep=True)
        predictions, fit, _ = _fit_and_predict(
            train=train,
            test=test,
            method_config=METHODS[method_id],
            device=args.device,
        )
        decisive = predictions["target"].isin([0.0, 1.0])
        selected = predictions.loc[decisive]
        target = selected["target"].to_numpy(dtype=float)
        probability = selected["p_rater"].to_numpy(dtype=float)
        nll = float(selected["rater_nll_loss"].mean())
        accuracy = float(np.mean((probability >= 0.5) == target.astype(bool)))
        return {
            "dataset_id": dataset_id,
            "seed": seed,
            "method_id": method_id,
            "method": METHOD_LABELS[method_id],
            "test_nll": nll,
            "test_accuracy": accuracy,
            "test_rows": len(selected),
            "fit_status": fit["fit_status"],
            "time_seconds": fit["time_seconds"],
        }

    with ThreadPoolExecutor(
        max_workers=min(args.fit_workers, len(tasks))
    ) as executor:
        rows = list(executor.map(evaluate, tasks))
    _write_csv(output_path, pd.DataFrame(rows))
    _write_json(metadata_path, identity)


def paired_bootstrap(matrix, seed, repetitions):
    values = matrix.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(repetitions, len(values)))
    samples = values[indices].mean(axis=1)
    return (
        values.mean(axis=0),
        np.quantile(samples, 0.025, axis=0),
        np.quantile(samples, 0.975, axis=0),
    )


def aggregate(root, repetitions, seed):
    paths = sorted((root / "raw").glob("*.csv"))
    if not paths:
        raise RuntimeError(f"No baseline utility results in {root}")
    runs = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    _write_csv(root / "baseline_utility_runs.csv", runs)
    baseline = (
        runs.groupby(["dataset_id", "method_id", "method"], as_index=False)[
            ["test_nll", "test_accuracy"]
        ]
        .mean()
    )

    rank_balance = pd.read_csv(RANK_BALANCE_UTILITY)
    rank_balance = rank_balance.loc[rank_balance["variant"].eq("full_tuned")].copy()
    rank_balance["method_id"] = "rank_balance"
    rank_balance["method"] = "RankBalance"
    rank_balance = rank_balance[
        ["dataset_id", "method_id", "method", "test_nll", "test_accuracy"]
    ]
    dataset_level = pd.concat([baseline, rank_balance], ignore_index=True)
    _write_csv(root / "paper_utility_dataset_level.csv", dataset_level)

    method_order = PAPER_METHODS + ["rank_balance"]
    rows = []
    for metric_index, metric in enumerate(["test_nll", "test_accuracy"]):
        matrix = (
            dataset_level.pivot(
                index="dataset_id", columns="method_id", values=metric
            )
            .reindex(columns=method_order)
            .dropna(how="any")
        )
        mean, lower, upper = paired_bootstrap(
            matrix, seed + metric_index, repetitions
        )
        for index, method_id in enumerate(method_order):
            rows.append(
                {
                    "method_id": method_id,
                    "method": METHOD_LABELS[method_id],
                    "metric": metric,
                    "mean": mean[index],
                    "ci_lower": lower[index],
                    "ci_upper": upper[index],
                    "datasets": len(matrix),
                }
            )
    statistics = pd.DataFrame(rows)
    _write_csv(root / "paper_utility_statistics.csv", statistics)

    table = pd.DataFrame({"method_id": method_order})
    table["Method"] = table["method_id"].map(METHOD_LABELS)
    for metric, label, percentage in [
        ("test_nll", "Test NLL", False),
        ("test_accuracy", "Accuracy", True),
    ]:
        values = statistics.loc[statistics["metric"].eq(metric)].set_index(
            "method_id"
        ).reindex(method_order)
        if percentage:
            table[label] = [
                f"{m:.1%} [{l:.1%}, {u:.1%}]"
                for m, l, u in zip(values["mean"], values["ci_lower"], values["ci_upper"])
            ]
        else:
            table[label] = [
                f"{m:.4f} [{l:.4f}, {u:.4f}]"
                for m, l, u in zip(values["mean"], values["ci_lower"], values["ci_upper"])
            ]
    table = table.drop(columns="method_id")
    _write_csv(root / "paper_utility_table.csv", table)
    (root / "paper_utility_table.html").write_text(
        table.to_html(index=False), encoding="utf-8"
    )
    print(table.to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 2023, 3407, 7919, 15401])
    parser.add_argument("--fit-workers", type=int, default=8)
    parser.add_argument("--device", choices=["cpu", "auto", "cuda"], default="cpu")
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
    definitions = _load_datasets(args.config.resolve())
    for dataset_id in args.datasets:
        if dataset_id == "ihq_all":
            raise ValueError("IHQ-all is excluded")
        run_dataset(dataset_id, definitions[dataset_id], args)
    if not args.no_aggregate:
        aggregate(
            args.results_root,
            args.bootstrap_repetitions,
            args.bootstrap_seed,
        )


if __name__ == "__main__":
    main()
