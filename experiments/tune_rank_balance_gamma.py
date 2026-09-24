#!/usr/bin/env python3
"""Select one global RankBalance gamma using decisive validation NLL."""

from __future__ import annotations
import _bootstrap  # noqa: F401

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

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
    PROJECT_ROOT / "experiment_results" / "rank_balance_gamma_validation"
)


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
        "protocol": "decisive-validation-nll-one-se-v1",
        "dataset_id": dataset_id,
        "path": str(data_path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "seeds": args.seeds,
        "gammas": args.gammas,
        "lambda_theta": args.lambda_theta,
        "lambda_u": args.lambda_u,
        "mu": args.mu,
        "max_iter": args.max_iter,
        "tol": args.tol,
    }
    if not args.overwrite and output_path.exists() and metadata_path.exists():
        if json.loads(metadata_path.read_text(encoding="utf-8")) == identity:
            print(f"Resume validation: {dataset_id} already complete", flush=True)
            return

    frame, _ = _load_data(data_path)
    rows = []
    for seed in args.seeds:
        train, validation, _, _ = _split_data(frame, seed)
        models = sorted(
            set(train["methodA"].astype(str))
            | set(train["methodB"].astype(str))
        )
        raters = sorted(train["answerer"].astype(str).unique())

        def evaluate(gamma):
            config = {
                "gamma": gamma,
                "regularizer": "full",
                "lambda_theta": args.lambda_theta,
                "lambda_u": args.lambda_u,
                "mu": args.mu,
                "max_iter": args.max_iter,
                "tol": args.tol,
            }
            fitted = fit_rank_balance(
                train, config, models=models, raters=raters
            )
            metrics = decisive_preference_metrics(validation, fitted)
            return {
                "dataset_id": dataset_id,
                "seed": seed,
                "gamma": gamma,
                "validation_nll": metrics["nll"],
                "validation_accuracy": metrics["accuracy"],
                "validation_rows": metrics["rows"],
                "fit_success": fitted["success"],
                "time_seconds": fitted["elapsed"],
            }

        with ThreadPoolExecutor(
            max_workers=min(args.gamma_workers, len(args.gammas))
        ) as executor:
            rows.extend(executor.map(evaluate, args.gammas))

    _write_csv(output_path, pd.DataFrame(rows))
    _write_json(metadata_path, identity)


def aggregate(root: Path):
    paths = sorted((root / "raw").glob("*.csv"))
    if not paths:
        raise RuntimeError(f"No validation results found in {root}")
    runs = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    if not runs["fit_success"].astype(bool).all():
        raise RuntimeError("At least one validation fit did not converge")
    _write_csv(root / "validation_runs.csv", runs)

    # Seeds are repeated measurements; datasets are the macro/SE units.
    dataset_level = (
        runs.groupby(["dataset_id", "gamma"], as_index=False)[
            ["validation_nll", "validation_accuracy"]
        ]
        .mean()
    )
    summary = (
        dataset_level.groupby("gamma")
        .agg(
            validation_nll_mean=("validation_nll", "mean"),
            validation_nll_std=("validation_nll", "std"),
            validation_accuracy_mean=("validation_accuracy", "mean"),
            datasets=("dataset_id", "nunique"),
        )
        .reset_index()
        .sort_values("gamma", kind="stable")
    )
    summary["validation_nll_se"] = (
        summary["validation_nll_std"] / summary["datasets"].pow(0.5)
    )
    minimum_row = summary.loc[summary["validation_nll_mean"].idxmin()]
    gamma_min = float(minimum_row["gamma"])
    validation_matrix = dataset_level.pivot(
        index="dataset_id", columns="gamma", values="validation_nll"
    )
    paired_rows = []
    for gamma in summary["gamma"]:
        differences = validation_matrix[gamma] - validation_matrix[gamma_min]
        difference_se = float(differences.std(ddof=1)) / math.sqrt(
            len(differences)
        )
        paired_rows.append(
            {
                "gamma": gamma,
                "paired_difference_mean": float(differences.mean()),
                "paired_difference_se": difference_se,
            }
        )
    summary = summary.merge(
        pd.DataFrame(paired_rows), on="gamma", validate="one_to_one"
    )
    summary["one_se_eligible"] = (
        summary["paired_difference_mean"]
        <= summary["paired_difference_se"]
    )
    selected_gamma = float(
        summary.loc[summary["one_se_eligible"], "gamma"].max()
    )
    summary["selected"] = summary["gamma"].eq(selected_gamma)
    _write_csv(root / "gamma_validation_summary.csv", summary)
    selected = {
        "gamma": selected_gamma,
        "gamma_min": gamma_min,
        "minimum_validation_nll": float(minimum_row["validation_nll_mean"]),
        "datasets": int(minimum_row["datasets"]),
        "selection_rule": (
            "largest gamma whose paired dataset-macro decisive validation "
            "NLL increase over gamma_min does not exceed one SE of that "
            "paired increase"
        ),
    }
    _write_json(root / "selected_gamma.json", selected)
    print(summary.to_string(index=False))
    print(f"\nSelected gamma*: {selected_gamma}")
    return selected


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 2023, 3407, 7919, 15401])
    parser.add_argument(
        "--gammas",
        nargs="+",
        type=float,
        default=[0.0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
    )
    parser.add_argument("--gamma-workers", type=int, default=8)
    parser.add_argument("--lambda-theta", type=float, default=1.0)
    parser.add_argument("--lambda-u", type=float, default=1.0)
    parser.add_argument("--mu", type=float, default=2.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--no-aggregate", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.results_root = args.results_root.resolve()
    args.results_root.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        aggregate(args.results_root)
        return
    definitions = _load_datasets(args.config.resolve())
    for dataset_id in args.datasets:
        if dataset_id == "ihq_all":
            raise ValueError("IHQ-all is excluded from this experiment")
        run_dataset(dataset_id, definitions[dataset_id], args)
    if not args.no_aggregate:
        aggregate(args.results_root)


if __name__ == "__main__":
    main()
