#!/usr/bin/env python3
"""Matched RankBalance ablation with adaptive attacks and held-out NLL."""

from __future__ import annotations
import _bootstrap  # noqa: F401

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Dict, List, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rank_balance_ablation import (  # noqa: E402
    apply_attack,
    fit_rank_balance,
    preference_nll,
    select_adaptive_attack,
)
from run_pollution_experiment import _load_data  # noqa: E402


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "paper_datasets.json"
DEFAULT_RESULTS = PROJECT_ROOT / "experiment_results" / "rank_balance_ablation"
DEFAULT_GAMMA_PATH = PROJECT_ROOT / "experiment_results/rank_balance_gamma_validation/selected_gamma.json"
DEFAULT_DATASETS = [
    "chatbot_arena_33k",
    "vision_arena",
    "search_arena",
    "computer_agent_arena",
    "mt_bench",
    "humaine",
    "multipref_all",
    "llm_judge_holdout",
    "hific",
    "conha",
    "wd",
]
VARIANT_LABELS = {
    "matched_gamma0": "Matched baseline: gamma=0",
    "delete_only": "Delete-only regularizer",
    "full": "Full RankBalance",
}


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _load_datasets(path: Path) -> Dict[str, Dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    defaults = raw.get("defaults", {})
    return {key: {**defaults, **value} for key, value in raw["datasets"].items()}


def _split_data(frame: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    assignment = np.full(len(frame), "train", dtype=object)
    order = rng.permutation(len(frame))
    test_count = max(1, int(round(0.2 * len(frame))))
    validation_count = max(1, int(round(0.2 * len(frame))))
    assignment[order[:test_count]] = "test"
    assignment[order[test_count:test_count + validation_count]] = "validation"

    # Preserve model coverage in training without looking at preference labels.
    all_models = sorted(set(frame["methodA"].astype(str)) | set(frame["methodB"].astype(str)))
    while True:
        train = assignment == "train"
        train_models = set(frame.loc[train, "methodA"].astype(str)) | set(frame.loc[train, "methodB"].astype(str))
        missing = [model for model in all_models if model not in train_models]
        if not missing:
            break
        # Recompute after each move because one row can add two missing models.
        model = missing[0]
        candidates = np.flatnonzero(
            (assignment != "train")
            & (
                frame["methodA"].astype(str).eq(model).to_numpy()
                | frame["methodB"].astype(str).eq(model).to_numpy()
            )
        )
        if not len(candidates):
            raise RuntimeError(f"Cannot preserve training coverage for {model}")
        assignment[candidates[0]] = "train"

    manifest = pd.DataFrame({"row_id": frame["__row_id"], "split": assignment})
    return (
        frame.loc[assignment == "train"].reset_index(drop=True),
        frame.loc[assignment == "validation"].reset_index(drop=True),
        frame.loc[assignment == "test"].reset_index(drop=True),
        manifest,
    )


def _topk_changed(clean: pd.DataFrame, attacked: pd.DataFrame, top_k: int) -> float:
    clean_set = set(clean["Method"].astype(str).head(top_k))
    attacked_set = set(attacked["Method"].astype(str).head(top_k))
    return float(clean_set != attacked_set)


def _variant_configs(args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
    common = {
        "lambda_theta": args.lambda_theta,
        "lambda_u": args.lambda_u,
        "mu": args.mu,
        "max_iter": args.max_iter,
        "tol": args.tol,
    }
    return {
        "matched_gamma0": {**common, "gamma": 0.0, "regularizer": "full"},
        "delete_only": {**common, "gamma": args.gamma, "regularizer": "delete_only"},
        "full": {**common, "gamma": args.gamma, "regularizer": "full"},
    }


def _run_variant(
    dataset_id: str,
    seed: int,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    variant: str,
    config: Dict[str, Any],
    top_ks: Sequence[int],
    budget_fraction: float,
    condition_workers: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    models = sorted(set(train["methodA"].astype(str)) | set(train["methodB"].astype(str)))
    raters = sorted(train["answerer"].astype(str).unique())
    clean = fit_rank_balance(train, config, models=models, raters=raters)
    clean_nll, clean_count = preference_nll(test, clean)
    validation_nll, validation_count = preference_nll(validation, clean)
    budget = max(1, int(math.floor(budget_fraction * len(train))))
    run_rows = [{
        "dataset_id": dataset_id,
        "seed": seed,
        "variant": variant,
        "variant_label": VARIANT_LABELS[variant],
        "attack_type": "clean",
        "top_k": 0,
        "attack_success": np.nan,
        "test_nll": clean_nll,
        "test_rows": clean_count,
        "validation_nll": validation_nll,
        "validation_rows": validation_count,
        "budget": 0,
        "selected_rows": 0,
        "fit_success": clean["success"],
        "time_seconds": clean["elapsed"],
        "error": "",
    }]
    manifest_rows: List[Dict[str, Any]] = []
    conditions = [
        (attack_type, top_k)
        for attack_type in ("delete", "flip")
        for top_k in top_ks
        if top_k < len(models)
    ]

    def run_condition(condition):
        attack_type, top_k = condition
        selected = select_adaptive_attack(train, clean, attack_type, top_k, budget)
        if selected is None:
            raise RuntimeError(f"No eligible {attack_type} records for Top-{top_k}")
        attacked_train = apply_attack(train, attack_type, selected["positions"])
        attacked = fit_rank_balance(attacked_train, config, models=models, raters=raters)
        test_nll, test_count = preference_nll(test, attacked)
        row = {
            "dataset_id": dataset_id,
            "seed": seed,
            "variant": variant,
            "variant_label": VARIANT_LABELS[variant],
            "attack_type": attack_type,
            "top_k": top_k,
            "attack_success": _topk_changed(clean["ranking"], attacked["ranking"], top_k),
            "test_nll": test_nll,
            "test_rows": test_count,
            "validation_nll": validation_nll,
            "validation_rows": validation_count,
            "budget": budget,
            "selected_rows": len(selected["positions"]),
            "target_a": selected["target_a"],
            "target_b": selected["target_b"],
            "predicted_gap": selected["predicted_gap"],
            "fit_success": attacked["success"],
            "time_seconds": attacked["elapsed"],
            "error": "",
        }
        manifests = [
            {
                "dataset_id": dataset_id,
                "seed": seed,
                "variant": variant,
                "attack_type": attack_type,
                "top_k": top_k,
                "attack_order": order + 1,
                "train_position": int(position),
                "source_row_id": int(train.iloc[position]["__row_id"]),
            }
            for order, position in enumerate(selected["positions"])
        ]
        return row, manifests

    with ThreadPoolExecutor(max_workers=min(condition_workers, len(conditions) or 1)) as executor:
        for row, manifests in executor.map(run_condition, conditions):
            run_rows.append(row)
            manifest_rows.extend(manifests)
    return run_rows, manifest_rows


def run_dataset(dataset_id: str, settings: Dict[str, Any], args: argparse.Namespace) -> None:
    output_dir = args.results_root / "raw" / dataset_id
    path = Path(settings["csv"])
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    frame, _ = _load_data(path)
    configs = _variant_configs(args)
    stat = path.stat()
    identity = {
        "dataset_id": dataset_id,
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "rows": len(frame),
        "seeds": args.seeds,
        "top_ks": args.top_ks,
        "budget_fraction": args.budget_fraction,
        "configs": configs,
    }
    metadata_path = output_dir / "metadata.json"
    runs_path = output_dir / "runs.csv"
    if not args.overwrite and metadata_path.exists() and runs_path.exists():
        if json.loads(metadata_path.read_text(encoding="utf-8")) == identity:
            print(f"Resume: {dataset_id} already complete", flush=True)
            return
    all_runs, all_manifests = [], []
    for seed in args.seeds:
        train, validation, test, split = _split_data(frame, seed)
        split_path = args.results_root / "splits" / dataset_id / f"seed_{seed}.csv"
        _write_csv(split_path, split)
        for variant, config in configs.items():
            try:
                rows, manifests = _run_variant(
                    dataset_id, seed, train, validation, test, variant, config,
                    args.top_ks, args.budget_fraction, args.condition_workers,
                )
                all_runs.extend(rows)
                all_manifests.extend(manifests)
            except Exception as exc:
                all_runs.append({
                    "dataset_id": dataset_id,
                    "seed": seed,
                    "variant": variant,
                    "variant_label": VARIANT_LABELS[variant],
                    "attack_type": "error",
                    "top_k": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                error_path = output_dir / f"seed_{seed}_{variant}_traceback.txt"
                error_path.parent.mkdir(parents=True, exist_ok=True)
                error_path.write_text(traceback.format_exc(), encoding="utf-8")
    _write_csv(runs_path, pd.DataFrame(all_runs))
    _write_csv(output_dir / "attack_manifests.csv", pd.DataFrame(all_manifests))
    _write_json(metadata_path, identity)


def aggregate(root: Path) -> None:
    frames = [pd.read_csv(path) for path in sorted((root / "raw").glob("*/runs.csv"))]
    if not frames:
        raise RuntimeError(f"No ablation runs found in {root}")
    runs = pd.concat(frames, ignore_index=True)
    _write_csv(root / "runs.csv", runs)
    success = runs.loc[runs["attack_type"].isin(["clean", "delete", "flip"])].copy()
    clean = success.loc[success["attack_type"].eq("clean"), ["dataset_id", "seed", "variant", "test_nll"]]
    clean = clean.rename(columns={"test_nll": "clean_test_nll"})
    attacked = (
        success.loc[success["attack_type"].isin(["delete", "flip"])]
        .groupby(["dataset_id", "seed", "variant", "attack_type"], as_index=False)[["attack_success", "test_nll"]]
        .mean()
    )
    wide = attacked.pivot(index=["dataset_id", "seed", "variant"], columns="attack_type", values=["attack_success", "test_nll"])
    wide.columns = [f"{attack}_{metric}" for metric, attack in wide.columns]
    units = clean.merge(wide.reset_index(), on=["dataset_id", "seed", "variant"], how="inner")
    _write_csv(root / "dataset_seed_statistics.csv", units)
    metrics = ["delete_attack_success", "flip_attack_success", "clean_test_nll", "delete_test_nll", "flip_test_nll"]
    rows = []
    for variant, group in units.groupby("variant", sort=False):
        row = {"variant": variant, "configuration": VARIANT_LABELS[variant], "units": len(group)}
        for metric in metrics:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
        rows.append(row)
    statistics = pd.DataFrame(rows).set_index("variant").reindex(VARIANT_LABELS).reset_index()
    _write_csv(root / "ablation_statistics.csv", statistics)
    display = statistics[["configuration"]].copy()
    labels = {
        "delete_attack_success": "Delete attack success",
        "flip_attack_success": "Flip attack success",
        "clean_test_nll": "Clean test NLL",
        "delete_test_nll": "Delete test NLL",
        "flip_test_nll": "Flip test NLL",
    }
    for metric, label in labels.items():
        display[label] = statistics.apply(
            lambda row: f"{row[f'{metric}_mean']:.4f} +/- {row[f'{metric}_std']:.4f}", axis=1
        )
    _write_csv(root / "ablation_table.csv", display)
    (root / "ablation_table.html").write_text(display.to_html(index=False), encoding="utf-8")
    print(display.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 2023, 3407, 7919, 15401])
    parser.add_argument("--top-ks", nargs="+", type=int, default=[1, 3, 5, 10, 20])
    parser.add_argument("--budget-fraction", type=float, default=0.01)
    parser.add_argument("--condition-workers", type=int, default=8)
    parser.add_argument("--gamma-path", type=Path, default=DEFAULT_GAMMA_PATH)
    parser.add_argument("--lambda-theta", type=float, default=1.0)
    parser.add_argument("--lambda-u", type=float, default=1.0)
    parser.add_argument("--mu", type=float, default=2.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--no-aggregate", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.results_root = args.results_root.resolve()
    args.results_root.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        aggregate(args.results_root)
        return
    args.gamma = float(json.loads(args.gamma_path.resolve().read_text(encoding="utf-8"))["gamma"])
    definitions = _load_datasets(args.config.resolve())
    for dataset_id in args.datasets:
        if dataset_id == "ihq_all":
            raise ValueError("IHQ-all is excluded from this ablation")
        if dataset_id not in definitions:
            raise ValueError(f"Unknown dataset: {dataset_id}")
        print(f"Running {dataset_id}", flush=True)
        run_dataset(dataset_id, definitions[dataset_id], args)
    if not args.no_aggregate:
        _write_json(args.results_root / "settings.json", {
            "datasets": args.datasets,
            "seeds": args.seeds,
            "top_ks": args.top_ks,
            "budget_fraction": args.budget_fraction,
            "variants": _variant_configs(args),
        })
        aggregate(args.results_root)


if __name__ == "__main__":
    main()
