#!/usr/bin/env python3
"""Victim-conditioned adaptive Delete/Flip robustness experiment."""

from __future__ import annotations
import _bootstrap  # noqa: F401

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import math
import os
from multiprocessing import Pool
from pathlib import Path
import pickle
import sys
import traceback
from typing import Any, Dict, List, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.special import expit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from run_convergence_experiment import METHODS  # noqa: E402
from run_pollution_experiment import _fit_method, _load_data  # noqa: E402


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiment_7a_rank_robustness.json"
DEFAULT_RESULTS = (
    PROJECT_ROOT / "experiment_results" / "victim_specific_robustness"
)
DEFAULT_GAMMA_PATH = (
    PROJECT_ROOT
    / "experiment_results"
    / "rank_balance_gamma_validation"
    / "selected_gamma.json"
)
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
DEFAULT_METHODS = [
    "m_elo",
    "crowd_bt",
    "bayes_bt",
    "bbq",
    "am_elo",
    "rank_balance_tuned_full",
]
METHOD_LABELS = {
    "m_elo": "ELO",
    "crowd_bt": "Crowd-BT",
    "bayes_bt": "Bayes-BT",
    "bbq": "BBQ",
    "am_elo": "am-ELO",
    "rank_balance_tuned_full": "RankBalance",
    "matched_gamma0": "Matched (gamma=0)",
}
PROTOCOL_VERSION = "victim_conditioned_adaptive_v1"
_WORKER: Dict[str, Any] = {}


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



def _write_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def _read_pickle(path: Path):
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except Exception:
        return None

def _load_definitions(path: Path) -> Dict[str, Dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    defaults = raw.get("defaults", {})
    return {key: {**defaults, **value} for key, value in raw["datasets"].items()}


def _budget_counts(
    rows: int,
    counts: Sequence[int],
    fractions: Sequence[float],
    rounding: str = "floor",
) -> List[int]:
    round_budget = math.ceil if rounding == "ceil" else math.floor
    values = {int(value) for value in counts if 0 < int(value) < rows}
    values.update(
        max(1, int(round_budget(rows * float(value))))
        for value in fractions
        if 0.0 < float(value) <= 1.0
    )
    return sorted(value for value in values if value < rows)


def _score_map(ranking: pd.DataFrame) -> Dict[str, float]:
    scale = math.log(10.0) / 400.0
    values = ranking.set_index("Method")["ELO Score"].astype(float)
    centered = values - values.mean()
    return {str(key): float(value * scale) for key, value in centered.items()}


def _topk_state(ranking: pd.DataFrame, top_ks: Sequence[int]) -> Dict[int, set]:
    order = ranking["Method"].astype(str).tolist()
    return {int(k): set(order[:k]) for k in top_ks if int(k) < len(order)}


def _topk_changed(
    ranking: pd.DataFrame,
    clean_sets: Dict[int, set],
) -> Dict[int, float]:
    order = ranking["Method"].astype(str).tolist()
    return {
        k: float(set(order[:k]) != clean_set)
        for k, clean_set in clean_sets.items()
    }


def _candidate_scores(
    frame: pd.DataFrame,
    fit: Dict[str, Any],
    clean_order: Sequence[str],
    clean_scores: Dict[str, float],
    top_ks: Sequence[int],
    attack_type: str,
    ridge: float,
) -> np.ndarray:
    models = list(clean_order)
    model_index = {model: index for index, model in enumerate(models)}
    a = frame["methodA"].astype(str).map(model_index)
    b = frame["methodB"].astype(str).map(model_index)
    labels = frame["answerValue"].astype(str).str.lower()
    valid = a.notna() & b.notna()
    if attack_type == "flip":
        valid &= labels.isin({"a", "b"})
    positions = np.flatnonzero(valid.to_numpy())
    output = np.full(len(frame), -np.inf, dtype=float)
    if not len(positions):
        return output

    ia = a.iloc[positions].to_numpy(dtype=int)
    ib = b.iloc[positions].to_numpy(dtype=int)
    theta_map = _score_map(fit["ranking"])
    theta = np.array([theta_map.get(model, clean_scores[model]) for model in models])
    probability = expit(theta[ia] - theta[ib])
    target = np.select(
        [labels.iloc[positions].eq("a"), labels.iloc[positions].eq("b")],
        [1.0, 0.0],
        default=0.5,
    )

    confidence = np.ones(len(positions), dtype=float)
    rater_values = fit.get("rater_values", {})
    if rater_values:
        mapped = frame.iloc[positions]["answerer"].astype(str).map(rater_values)
        finite = mapped.notna().to_numpy()
        confidence[finite] = np.clip(
            np.abs(mapped.loc[finite].to_numpy(dtype=float)), 0.05, 1.0
        )

    model_count = len(models)
    hessian = np.eye(model_count) * float(ridge)
    curvature = confidence * probability * (1.0 - probability)
    for left, right, weight in zip(ia, ib, curvature):
        hessian[left, left] += weight
        hessian[right, right] += weight
        hessian[left, right] -= weight
        hessian[right, left] -= weight
    inverse = np.linalg.pinv(hessian, rcond=1e-10)

    if attack_type == "delete":
        coefficient = confidence * (probability - target)
    else:
        coefficient = -confidence * (2.0 * target - 1.0)

    score = np.full(len(positions), -np.inf, dtype=float)
    valid_top_ks = [int(k) for k in top_ks if int(k) < model_count]
    for top_k in valid_top_ks:
        inside = clean_order[:top_k]
        outside = clean_order[top_k:]
        for inside_model in inside:
            for outside_model in outside:
                initial_gap = (
                    clean_scores[inside_model] - clean_scores[outside_model]
                )
                if initial_gap <= 0.0:
                    continue
                contrast_direction = (
                    inverse[model_index[inside_model]]
                    - inverse[model_index[outside_model]]
                )
                gap_change = coefficient * (
                    contrast_direction[ia] - contrast_direction[ib]
                )
                harmful = -gap_change / max(initial_gap, 1e-6)
                score = np.maximum(score, harmful)
    output[positions] = score
    return output


def _select_batch(
    original: pd.DataFrame,
    current: pd.DataFrame,
    scores: np.ndarray,
    selected_ids: set,
    count: int,
    attack_type: str,
) -> List[int]:
    candidates = current.copy()
    candidates["_score"] = scores
    candidates = candidates.loc[
        np.isfinite(candidates["_score"])
        & ~candidates["__row_id"].isin(selected_ids)
    ].sort_values(["_score", "__row_id"], ascending=[False, True])

    if attack_type == "flip":
        return candidates["__row_id"].head(count).astype(int).tolist()

    incident = pd.concat(
        [
            original.loc[~original["__row_id"].isin(selected_ids), "methodA"],
            original.loc[~original["__row_id"].isin(selected_ids), "methodB"],
        ]
    ).astype(str).value_counts().to_dict()
    chosen = []
    for _, row in candidates.iterrows():
        model_a, model_b = str(row["methodA"]), str(row["methodB"])
        if incident.get(model_a, 0) <= 1 or incident.get(model_b, 0) <= 1:
            continue
        chosen.append(int(row["__row_id"]))
        incident[model_a] -= 1
        incident[model_b] -= 1
        if len(chosen) == count:
            break
    return chosen


def _apply_selected(
    original: pd.DataFrame,
    selected_ids: set,
    attack_type: str,
) -> pd.DataFrame:
    selected = original["__row_id"].isin(selected_ids)
    if attack_type == "delete":
        return original.loc[~selected].reset_index(drop=True)
    result = original.copy()
    labels = result.loc[selected, "answerValue"]
    result.loc[selected, "answerValue"] = labels.map({"A": "B", "B": "A"})
    return result.reset_index(drop=True)


def _run_attack_path(
    original: pd.DataFrame,
    clean_fit: Dict[str, Any],
    method_config: Dict[str, Any],
    attack_type: str,
    budgets: Sequence[int],
    top_ks: Sequence[int],
    device: str,
    ridge: float,
    checkpoint_path: Path,
    checkpoint_identity: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    clean_order = clean_fit["ranking"]["Method"].astype(str).tolist()
    clean_scores = _score_map(clean_fit["ranking"])
    clean_sets = _topk_state(clean_fit["ranking"], top_ks)
    checkpoint = _read_pickle(checkpoint_path)
    if isinstance(checkpoint, dict) and checkpoint.get("identity") == checkpoint_identity:
        selected_ids = set(checkpoint["selected_ids"])
        current_frame = _apply_selected(original, selected_ids, attack_type)
        current_fit = checkpoint["current_fit"]
        rows = list(checkpoint["rows"])
        manifest = list(checkpoint["manifest"])
        completed_budgets = set(checkpoint["completed_budgets"])
        print(
            f"Resume {attack_type} checkpoint at budget "
            f"{max(completed_budgets) if completed_budgets else 0}",
            flush=True,
        )
    else:
        selected_ids = set()
        current_frame = original.copy()
        current_fit = clean_fit
        rows, manifest = [], []
        completed_budgets = set()

    for budget in budgets:
        if int(budget) in completed_budgets:
            continue
        needed = int(budget) - len(selected_ids)
        if needed <= 0:
            continue
        scores = _candidate_scores(
            current_frame, current_fit, clean_order, clean_scores,
            top_ks, attack_type, ridge,
        )
        chosen = _select_batch(
            original, current_frame, scores, selected_ids, needed, attack_type,
        )
        if not chosen:
            break
        selected_ids.update(chosen)
        for row_id in chosen:
            manifest.append(
                {
                    "attack_type": attack_type,
                    "budget": budget,
                    "selection_order": len(manifest) + 1,
                    "row_id": row_id,
                }
            )
        current_frame = _apply_selected(original, selected_ids, attack_type)
        current_raters = sorted(current_frame["answerer"].astype(str).unique())
        current_fit = _fit_method(
            current_frame, current_raters, method_config, device
        )
        changes = _topk_changed(current_fit["ranking"], clean_sets)
        for top_k, changed in changes.items():
            rows.append(
                {
                    "attack_type": attack_type,
                    "budget": budget,
                    "budget_fraction": budget / len(original),
                    "top_k": top_k,
                    "topk_changed": changed,
                    "selected_rows": len(selected_ids),
                    "fit_status": current_fit["fit_status"],
                    "time_seconds": current_fit["elapsed"],
                }
            )
        completed_budgets.add(int(budget))
        _write_pickle(
            checkpoint_path,
            {
                "identity": checkpoint_identity,
                "selected_ids": sorted(selected_ids),
                "current_fit": current_fit,
                "rows": rows,
                "manifest": manifest,
                "completed_budgets": sorted(completed_budgets),
            },
        )
    return rows, manifest

def _worker_init(context):
    global _WORKER
    _WORKER = context


def _run_method(method_id: str) -> Dict[str, Any]:
    context = _WORKER
    output = context["output_root"] / context["dataset_id"] / method_id
    metadata_path = output / "metadata.json"
    runs_path = output / "runs.csv"
    manifest_path = output / "selection_manifest.csv"
    identity = {
        "protocol": PROTOCOL_VERSION,
        "dataset_id": context["dataset_id"],
        "method_id": method_id,
        "method_config": METHODS[method_id],
        "budgets": context["budgets"],
        "top_ks": context["top_ks"],
        "ridge": context["ridge"],
    }
    if not context["overwrite"] and metadata_path.exists() and runs_path.exists():
        if json.loads(metadata_path.read_text(encoding="utf-8")) == identity:
            print(f"Resume victim attack: {context['dataset_id']}/{method_id}", flush=True)
            return {"status": "resumed", "method_id": method_id}

    try:
        frame = context["frame"]
        raters = sorted(frame["answerer"].astype(str).unique())
        method_config = METHODS[method_id]
        clean_checkpoint_path = output / "checkpoints" / "clean.pkl"
        clean_checkpoint = _read_pickle(clean_checkpoint_path)
        if isinstance(clean_checkpoint, dict) and clean_checkpoint.get("identity") == identity:
            clean_fit = clean_checkpoint["fit"]
            print(f"Resume clean fit: {context['dataset_id']}/{method_id}", flush=True)
        else:
            clean_fit = _fit_method(frame, raters, method_config, context["device"])
            _write_pickle(clean_checkpoint_path, {"identity": identity, "fit": clean_fit})

        def run_attack(attack_type):
            return _run_attack_path(
                frame,
                clean_fit,
                method_config,
                attack_type,
                context["budgets"],
                context["top_ks"],
                context["device"],
                context["ridge"],
                output / "checkpoints" / f"{attack_type}.pkl",
                {**identity, "attack_type": attack_type},
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(run_attack, ["delete", "flip"]))
        rows, manifests = [], []
        for attack_type, (attack_rows, attack_manifest) in zip(
            ["delete", "flip"], results
        ):
            for row in attack_rows:
                rows.append(
                    {
                        "dataset_id": context["dataset_id"],
                        "method_id": method_id,
                        "method": METHOD_LABELS[method_id],
                        **row,
                        "status": "success",
                        "error": "",
                    }
                )
            for row in attack_manifest:
                manifests.append(
                    {
                        "dataset_id": context["dataset_id"],
                        "method_id": method_id,
                        **row,
                    }
                )
        _write_csv(runs_path, pd.DataFrame(rows))
        _write_csv(manifest_path, pd.DataFrame(manifests))
        _write_json(metadata_path, identity)
        return {"status": "success", "method_id": method_id}
    except Exception as exc:
        output.mkdir(parents=True, exist_ok=True)
        (output / "traceback.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        return {
            "status": "error",
            "method_id": method_id,
            "error": f"{type(exc).__name__}: {exc}",
        }


def aggregate(root: Path) -> None:
    paths = sorted(root.glob("*/*/runs.csv"))
    if not paths:
        raise RuntimeError(f"No victim-specific results found in {root}")
    runs = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    _write_csv(root / "runs.csv", runs)
    successful = runs.loc[runs["status"].eq("success")].copy()
    macro = (
        successful.groupby(
            ["method_id", "method", "attack_type", "budget", "top_k"]
        )["topk_changed"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "asr", "std": "asr_std", "count": "datasets"})
    )
    _write_csv(root / "macro_asr.csv", macro)

    first = successful.loc[successful["topk_changed"].eq(1)].groupby(
        ["dataset_id", "method_id", "method", "attack_type", "top_k"],
        as_index=False,
    )["budget"].min()
    _write_csv(root / "first_success_cost.csv", first)
    print(macro.to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--gamma-path", type=Path, default=DEFAULT_GAMMA_PATH)
    parser.add_argument("--top-ks", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--budget-counts", nargs="*", type=int, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--budget-rounding", choices=["floor", "ceil"], default="floor")
    parser.add_argument(
        "--budget-fractions",
        nargs="+",
        type=float,
        default=[0.0001, 0.0005, 0.001, 0.005, 0.01],
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--device", choices=["cpu", "auto", "cuda"], default="cpu")
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--no-aggregate", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.results_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        aggregate(root)
        return
    selected_gamma = float(
        json.loads(args.gamma_path.resolve().read_text(encoding="utf-8"))["gamma"]
    )
    METHODS["rank_balance_tuned_full"]["rank_balance_config"]["gamma"] = selected_gamma
    METHODS["matched_gamma0"] = deepcopy(METHODS["rank_balance_tuned_full"])
    METHODS["matched_gamma0"]["rank_balance_config"]["gamma"] = 0.0
    definitions = _load_definitions(args.config.resolve())
    for dataset_id in args.datasets:
        if dataset_id == "ihq_all":
            raise ValueError("IHQ-all is excluded")
        settings = definitions[dataset_id]
        data_path = Path(settings["csv"])
        if not data_path.is_absolute():
            data_path = PROJECT_ROOT / data_path
        frame, _ = _load_data(data_path)
        budgets = _budget_counts(
            len(frame), args.budget_counts, args.budget_fractions, args.budget_rounding
        )
        context = {
            "dataset_id": dataset_id,
            "frame": frame,
            "budgets": budgets,
            "top_ks": args.top_ks,
            "device": args.device,
            "ridge": args.ridge,
            "output_root": root,
            "overwrite": args.overwrite,
        }
        print(
            f"Victim attacks {dataset_id}: rows={len(frame)}, budgets={budgets}",
            flush=True,
        )
        with Pool(
            processes=min(args.workers, len(args.methods)),
            initializer=_worker_init,
            initargs=(context,),
        ) as pool:
            results = list(pool.map(_run_method, args.methods))
        failures = [result for result in results if result["status"] == "error"]
        if failures:
            raise RuntimeError(f"Victim-specific failures: {failures}")
    if not args.no_aggregate:
        aggregate(root)


if __name__ == "__main__":
    main()
