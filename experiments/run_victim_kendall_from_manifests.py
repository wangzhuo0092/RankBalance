#!/usr/bin/env python3
"""Refit saved victim-specific edits to measure full-ranking Kendall tau."""

import _bootstrap  # noqa: F401
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import pickle

import pandas as pd

from run_cross_attack_transfer import apply_attack, ranking_metrics
from run_pollution_experiment import _fit_method, _load_data
from run_rank_balance_ablation import DEFAULT_CONFIG, _load_datasets
from run_victim_specific_robustness import DEFAULT_DATASETS


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "experiment_results/victim_specific_matched_budgets"
OUTPUT = SOURCE / "kendall"
METHODS = (
    "m_elo", "crowd_bt", "bayes_bt", "bbq", "am_elo", "matched_gamma0",
    "rank_balance_tuned_full",
)


def atomic_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_json(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def fit_task(dataset_id, method_id, csv_path, source_root, output_root):
    source = source_root / dataset_id / method_id
    runs_path = source / "runs.csv"
    manifest_path = source / "selection_manifest.csv"
    checkpoint_path = source / "checkpoints" / "clean.pkl"
    if not all(path.exists() for path in (runs_path, manifest_path, checkpoint_path)):
        return f"Missing original self-attack results: {dataset_id}/{method_id}"

    output = output_root / "raw" / dataset_id / f"{method_id}.csv"
    metadata = output.with_suffix(".json")
    identity = {
        "protocol": "victim_kendall_from_saved_edits_v1",
        "dataset_id": dataset_id,
        "method_id": method_id,
        "data_mtime_ns": csv_path.stat().st_mtime_ns,
        "manifest_mtime_ns": manifest_path.stat().st_mtime_ns,
        "checkpoint_mtime_ns": checkpoint_path.stat().st_mtime_ns,
    }
    if metadata.exists() and json.loads(metadata.read_text(encoding="utf-8")) != identity:
        raise ValueError(f"Input changed since partial Kendall refits: {output}")

    runs = pd.read_csv(runs_path)
    manifest = pd.read_csv(manifest_path)
    with checkpoint_path.open("rb") as handle:
        checkpoint = pickle.load(handle)
    method_config = checkpoint["identity"]["method_config"]
    clean = checkpoint["fit"]["ranking"]
    expected = runs.loc[runs.top_k.eq(1), ["attack_type", "budget", "selected_rows"]]
    if expected.duplicated(["attack_type", "budget"]).any():
        raise ValueError(f"Duplicate top-1 attack conditions: {source}")
    expected = expected.sort_values(["attack_type", "budget"])

    existing = pd.read_csv(output) if output.exists() else pd.DataFrame()
    if not existing.empty and not metadata.exists():
        raise ValueError(f"Kendall output has no protocol identity: {output}")
    done = set(zip(existing.get("attack_type", []), existing.get("budget", [])))
    rows = existing.to_dict("records")
    pending = expected.loc[
        ~pd.Series(list(zip(expected.attack_type, expected.budget)), index=expected.index).isin(done)
    ]
    if pending.empty:
        return f"Reused {dataset_id}/{method_id}: {len(rows)} conditions"

    frame, _ = _load_data(csv_path)
    for condition in pending.itertuples(index=False):
        selected = manifest.loc[manifest.attack_type.eq(condition.attack_type)].sort_values("selection_order")
        row_ids = selected.row_id.astype(int).head(int(condition.selected_rows)).tolist()
        if len(row_ids) != condition.selected_rows or len(set(row_ids)) != len(row_ids):
            raise ValueError(f"Incomplete edit sequence for {dataset_id}/{method_id}/{condition.attack_type}/{condition.budget}")
        attacked = apply_attack(frame, row_ids, condition.attack_type)
        fit = _fit_method(attacked, sorted(attacked.answerer.astype(str).unique()), method_config, "cpu")
        if set(clean.Method.astype(str)) != set(fit["ranking"].Method.astype(str)):
            raise ValueError(f"Model coverage changed for {dataset_id}/{method_id}/{condition.budget}")
        tau, rho, maximum_change = ranking_metrics(clean, fit["ranking"])
        rows.append({
            "dataset_id": dataset_id, "method_id": method_id,
            "attack_type": condition.attack_type, "budget": int(condition.budget),
            "selected_rows": len(row_ids), "budget_fraction": condition.budget / len(frame),
            "kendall_tau": tau, "spearman_rho": rho, "max_rank_change": maximum_change,
            "fit_status": fit["fit_status"],
        })
        atomic_csv(pd.DataFrame(rows), output)
        atomic_json(identity, metadata)
        print(f"{dataset_id}/{method_id}/{condition.attack_type}: {condition.budget} edits", flush=True)
    return f"Completed {dataset_id}/{method_id}: {len(rows)} conditions"


def aggregate(root):
    paths = sorted((root / "raw").glob("*/*.csv"))
    if not paths:
        raise FileNotFoundError(f"No Kendall refits found under {root / 'raw'}")
    values = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    if values.duplicated(["dataset_id", "method_id", "attack_type", "budget"]).any():
        raise ValueError("Duplicate Kendall attack conditions")
    atomic_csv(values, root / "kendall_runs.csv")
    print(f"Kendall: {len(values)} refits across {values.dataset_id.nunique()} datasets")
    print(values.groupby(["method_id", "fit_status"], dropna=False).size().to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--source-root", type=Path, default=SOURCE)
    parser.add_argument("--results-root", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    if args.summarize_only:
        aggregate(args.results_root)
        return

    definitions = _load_datasets(args.config.resolve())
    jobs = []
    for dataset_id in args.datasets:
        path = Path(definitions[dataset_id]["csv"])
        csv_path = path if path.is_absolute() else PROJECT / path
        jobs.extend((dataset_id, method_id, csv_path, args.source_root, args.results_root)
                    for method_id in args.methods)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(fit_task, *job) for job in jobs]
        for future in as_completed(futures):
            print(future.result(), flush=True)
    aggregate(args.results_root)


if __name__ == "__main__":
    main()
