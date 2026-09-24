#!/usr/bin/env python3
"""Experiment 5: artificial rater pollution and ranking robustness.

This is not a Bootstrap experiment. Each method is fitted once on clean data,
then fitted on shared Random/Equal/Flip/Mixed perturbations of selected raters.
The runner saves every fitted ranking, supports resume, and writes both the
complete 0--45% result table and a post-hoc table whose conservative estimated
total pollution does not exceed 50%.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from multiprocessing import Pool
from pathlib import Path
import sys
import traceback
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Do not let every worker create its own BLAS thread pool.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata, spearmanr
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from elo_processor import EloProcessor  # noqa: E402
from models import (  # noqa: E402
    BETA_PRIOR_ALPHA,
    BETA_PRIOR_BETA,
    ELO_SCALE_FACTOR,
)
from run_cross_dataset_stability import METHODS  # noqa: E402
from run_data_scale_experiment import (  # noqa: E402
    _optimization_status,
    _resolve_methods,
)
from torch_bayesian_backend import resolve_device  # noqa: E402


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiment_5_pollution.json"
DEFAULT_RESULTS = (
    PROJECT_ROOT / "experiment_results" / "experiment_5_pollution"
)
PERTURBATIONS = {"random", "equal", "flip", "mixed"}
RATER_AWARE_METHODS = {
    "crowd_bt",
    "bbq",
    "am_elo",
    "correctness_downweight",
    "correctness_reverse",
    "adaptive_clip",
    "adaptive_flip",
}
RUN_FIELDS = [
    "dataset_id",
    "dataset",
    "detection_unit",
    "method_id",
    "method",
    "model",
    "shared_rater",
    "condition_id",
    "perturbation",
    "seed",
    "level_id",
    "requested_rater_fraction",
    "requested_rater_count",
    "selected_raters",
    "total_raters",
    "actual_rater_fraction",
    "selected_rows",
    "selected_row_fraction",
    "changed_rows",
    "changed_row_fraction",
    "ties_before",
    "ties_after",
    "tie_fraction_before",
    "tie_fraction_after",
    "estimated_original_error_rate",
    "conservative_total_pollution",
    "status",
    "fit_status",
    "backend",
    "error_type",
    "error_message",
    "time_seconds",
    "top1_agreement",
    "top3_overlap",
    "ordered_top3_agreement",
    "pearson",
    "spearman",
    "kendall_tau",
    "rank_mae",
    "pairwise_consistency",
    "elo_rmse",
    "elo_max_abs_change",
    "method_coverage",
    "ranked_methods",
    "reference_methods",
    "rater_score_supported",
    "rater_score_coverage",
    "predicted_anomalous_raters",
    "anomaly_precision",
    "anomaly_recall",
    "anomaly_f1",
    "anomaly_auroc",
    "anomaly_auprc",
    "adaptive_contamination_rate",
]
RANKING_FIELDS = [
    "dataset_id",
    "dataset",
    "method_id",
    "method",
    "condition_id",
    "perturbation",
    "seed",
    "level_id",
    "requested_rater_fraction",
    "actual_rater_fraction",
    "model_name",
    "elo_score",
    "rank",
]
SUMMARY_METRICS = [
    "actual_rater_fraction",
    "selected_row_fraction",
    "changed_row_fraction",
    "tie_fraction_before",
    "tie_fraction_after",
    "estimated_original_error_rate",
    "conservative_total_pollution",
    "time_seconds",
    "top1_agreement",
    "top3_overlap",
    "ordered_top3_agreement",
    "pearson",
    "spearman",
    "kendall_tau",
    "rank_mae",
    "pairwise_consistency",
    "elo_rmse",
    "elo_max_abs_change",
    "method_coverage",
    "rater_score_coverage",
    "anomaly_precision",
    "anomaly_recall",
    "anomaly_f1",
    "anomaly_auroc",
    "anomaly_auprc",
    "adaptive_contamination_rate",
]

_WORKER_CONTEXT: Dict[str, Any] = {}
_RUN_LOCK_HANDLE = None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            _json_safe(value),
            handle,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _acquire_run_lock(root: Path) -> None:
    global _RUN_LOCK_HANDLE
    root.mkdir(parents=True, exist_ok=True)
    handle = (root / ".run.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(
            f"Another pollution experiment is writing to {root}"
        ) from exc
    handle.seek(0)
    handle.truncate()
    handle.write(f"{os.getpid()}\n")
    handle.flush()
    _RUN_LOCK_HANDLE = handle


def _load_config(path: Path) -> Dict[str, Dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    defaults = raw.get("defaults", {})
    datasets = raw.get("datasets", {})
    if not datasets:
        raise ValueError("Experiment 5 config contains no datasets")

    resolved: Dict[str, Dict[str, Any]] = {}
    for dataset_id, settings in datasets.items():
        value = {**defaults, **settings}
        missing = {"label", "csv", "seeds", "perturbations"} - set(value)
        if missing:
            raise ValueError(
                f"Dataset {dataset_id!r} is missing {sorted(missing)}"
            )
        value["seeds"] = [int(seed) for seed in value["seeds"]]
        value["pollution_ratios"] = [
            float(ratio) for ratio in value.get("pollution_ratios", [])
        ]
        value["perturbations"] = [
            str(name).lower() for name in value["perturbations"]
        ]
        unknown = set(value["perturbations"]) - PERTURBATIONS
        if unknown:
            raise ValueError(
                f"Unknown perturbations for {dataset_id}: {sorted(unknown)}"
            )
        if any(not 0.0 < ratio < 0.5 for ratio in value["pollution_ratios"]):
            raise ValueError(
                f"Pollution ratios for {dataset_id} must be in (0, 0.5)"
            )
        if "polluted_rater_counts" in value:
            value["polluted_rater_counts"] = [
                int(count) for count in value["polluted_rater_counts"]
            ]
        resolved[dataset_id] = value
    return resolved


def _resolve_datasets(
    requested: Sequence[str],
    config: Dict[str, Dict[str, Any]],
) -> List[str]:
    if not requested or "default" in requested:
        selected = [
            key for key, value in config.items() if value.get("enabled", True)
        ]
    elif "all" in requested:
        selected = list(config)
    else:
        unknown = sorted(set(requested) - set(config))
        if unknown:
            raise ValueError(f"Unknown datasets: {unknown}")
        selected = list(requested)
    if not selected:
        raise ValueError("No datasets selected")
    return list(dict.fromkeys(selected))


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _normalize_answer(value: Any) -> str:
    text = str(value).strip().lower()
    if text == "a":
        return "A"
    if text == "b":
        return "B"
    if text in {"tie", "draw"}:
        return "draw"
    return ""


def _drop_golden(frame: pd.DataFrame) -> pd.DataFrame:
    """Exclude attention-check rows from pollution and evaluation data."""
    if "isGolden" not in frame.columns:
        return frame
    golden = (
        frame["isGolden"].notna()
        & frame["isGolden"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("true")
    )
    return frame.loc[~golden]


def _load_data(path: Path) -> Tuple[pd.DataFrame, List[str]]:
    from elo_processor import DataProcessor

    processor = DataProcessor(str(path))
    if not processor.load_data() or processor.df is None:
        raise RuntimeError(f"Failed to load {path}")
    frame = _drop_golden(processor.df).copy()

    frame["methodA"] = frame["methodA"].astype(str).str.strip()
    frame["methodB"] = frame["methodB"].astype(str).str.strip()
    frame["answerValue"] = frame["answerValue"].map(_normalize_answer)
    frame["answerer"] = frame["answerer"].astype(str).str.strip()

    invalid_name = {"", "nan", "none", "n/a", "na"}
    valid = (
        ~frame["methodA"].str.lower().isin(invalid_name)
        & ~frame["methodB"].str.lower().isin(invalid_name)
        & frame["methodA"].ne(frame["methodB"])
        & frame["answerValue"].isin({"A", "B", "draw"})
        & frame["answerer"].ne("")
    )
    frame = frame.loc[valid].reset_index(drop=True)
    frame["__row_id"] = np.arange(len(frame), dtype=np.int64)
    if frame.empty:
        raise ValueError(f"No valid pairwise rows remain in {path}")

    raters = sorted(frame["answerer"].unique().tolist())
    if not raters:
        raise ValueError(f"No raters remain in {path}")
    return frame, raters


def _data_signature(path: Path, frame: pd.DataFrame) -> str:
    stat = path.stat()
    payload = {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "rows": len(frame),
        "raters": int(frame["answerer"].nunique()),
        "models": int(
            pd.concat([frame["methodA"], frame["methodB"]]).nunique()
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _stable_seed(*parts: Any) -> int:
    payload = ":".join(map(str, parts)).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16) % (2**32)


def _level_id_from_ratio(ratio: float) -> str:
    return f"p{int(round(100 * ratio)):02d}"


def _build_manifest(
    root: Path,
    dataset_id: str,
    raters: Sequence[str],
    settings: Dict[str, Any],
) -> pd.DataFrame:
    total_raters = len(raters)
    maximum = total_raters // 2
    rows: List[Dict[str, Any]] = []

    explicit_counts = settings.get("polluted_rater_counts")
    if explicit_counts is not None:
        levels = [
            (f"k{count}", count / total_raters, int(count))
            for count in explicit_counts
            if 0 < int(count) <= maximum
        ]
    else:
        levels = []
        for ratio in settings["pollution_ratios"]:
            count = int(math.floor(ratio * total_raters + 0.5))
            count = min(maximum, max(1, count))
            if count > 0:
                levels.append((_level_id_from_ratio(ratio), ratio, count))

    for seed in settings["seeds"]:
        rng = np.random.default_rng(_stable_seed(dataset_id, seed, "raters"))
        permutation = np.asarray(raters, dtype=object)[
            rng.permutation(total_raters)
        ].tolist()
        for level_id, requested_fraction, requested_count in levels:
            selected = [str(value) for value in permutation[:requested_count]]
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "seed": int(seed),
                    "level_id": level_id,
                    "requested_rater_fraction": float(requested_fraction),
                    "requested_rater_count": int(requested_count),
                    "selected_raters": len(selected),
                    "total_raters": total_raters,
                    "actual_rater_fraction": len(selected) / total_raters,
                    "selected_raters_json": json.dumps(
                        selected, ensure_ascii=True
                    ),
                }
            )

    expected = pd.DataFrame(rows)
    path = root / "manifests" / f"{dataset_id}.csv"
    if path.exists():
        existing = pd.read_csv(path)
        comparable = existing.copy()
        for column in expected.columns:
            if column not in comparable.columns:
                raise ValueError(
                    f"Manifest {path} is missing column {column}; "
                    "use a new results root"
                )
        comparable = comparable[expected.columns]
        try:
            pd.testing.assert_frame_equal(
                comparable,
                expected,
                check_dtype=False,
                check_exact=False,
                rtol=0.0,
                atol=1e-12,
            )
        except AssertionError as exc:
            raise ValueError(
                f"Manifest {path} conflicts with the current protocol; "
                "use a new results root"
            ) from exc
            
        return existing
    _atomic_csv(path, expected)
    return expected


def _apply_perturbation(
    frame: pd.DataFrame,
    selected_raters: Sequence[str],
    perturbation: str,
    seed: int,
    condition_id: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    perturbed = frame.copy()
    selected = set(map(str, selected_raters))
    mask = perturbed["answerer"].isin(selected)
    indices = np.flatnonzero(mask.to_numpy())
    answers = perturbed["answerValue"].to_numpy(copy=True)
    before = answers.copy()
    rng = np.random.default_rng(
        _stable_seed(seed, condition_id, perturbation, "labels")
    )

    def apply_one(index: int, kind: str) -> None:
        answer = answers[index]
        if kind == "equal":
            answers[index] = "draw"
        elif kind == "flip":
            if answer == "A":
                answers[index] = "B"
            elif answer == "B":
                answers[index] = "A"
        elif kind == "random":
            if answer == "A":
                answers[index] = "draw" if rng.random() < 0.5 else "B"
            elif answer == "B":
                answers[index] = "draw" if rng.random() < 0.5 else "A"
        else:
            raise ValueError(f"Unknown perturbation {kind}")

    if perturbation == "mixed":
        choices = rng.choice(
            np.asarray(["random", "equal", "flip"], dtype=object),
            size=len(indices),
        )
        for index, kind in zip(indices, choices):
            apply_one(int(index), str(kind))
    else:
        for index in indices:
            apply_one(int(index), perturbation)

    perturbed["answerValue"] = answers
    changed = answers != before
    ties_before = int(np.sum(before == "draw"))
    ties_after = int(np.sum(answers == "draw"))
    total = len(frame)
    diagnostics = {
        "selected_rows": int(mask.sum()),
        "selected_row_fraction": float(mask.mean()),
        "changed_rows": int(changed.sum()),
        "changed_row_fraction": float(changed.mean()),
        "ties_before": ties_before,
        "ties_after": ties_after,
        "tie_fraction_before": ties_before / total,
        "tie_fraction_after": ties_after / total,
    }
    return perturbed, diagnostics


def _extract_rater_values(processor: EloProcessor) -> Dict[str, float]:
    if processor.metric is None:
        return {}
    values: Dict[str, float] = {}
    for rater, raw in processor.metric.state.get("qualities", {}).items():
        value = raw.get("value") if isinstance(raw, dict) else raw
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            values[str(rater)] = number
    return values


def _fit_method(
    frame: pd.DataFrame,
    raters: Sequence[str],
    method_config: Dict[str, Any],
    device: str,
) -> Dict[str, Any]:
    if method_config["model"] == "rank_balance_ablation":
        from rank_balance_ablation import fit_rank_balance

        fitted = fit_rank_balance(
            frame,
            method_config["rank_balance_config"],
        )
        qualities = 1.0 / (1.0 + np.exp(-fitted["u"]))
        return {
            "ranking": fitted["ranking"],
            "elapsed": float(fitted["elapsed"]),
            "fit_status": (
                "converged" if fitted["success"] else "optimizer_warning"
            ),
            "backend": "cpu",
            "rater_values": dict(
                zip(fitted["design"]["raters"], qualities.astype(float))
            ),
            "quality_parameter": "agreement_probability",
            "adaptive_contamination_rate": math.nan,
        }

    processor = EloProcessor(frame, list(raters))
    ranking, elapsed = processor.process(
        df=frame,
        valid_users=list(raters),
        model=method_config["model"],
        shared_rater=method_config["shared_rater"],
        device=device,
        rank_balance_config=method_config.get("rank_balance_config"),
    )
    if ranking is None or ranking.empty:
        raise RuntimeError("The method returned an empty ranking")
    required = {"Method", "ELO Score"}
    if not required.issubset(ranking.columns):
        raise RuntimeError(
            f"Ranking is missing {sorted(required - set(ranking.columns))}"
        )
    ranking = ranking.sort_values(
        "ELO Score", ascending=False
    ).reset_index(drop=True)
    state = processor.metric.state if processor.metric is not None else {}
    adaptive = state.get("adaptive_clipping", {})
    return {
        "ranking": ranking,
        "elapsed": float(elapsed),
        "fit_status": _optimization_status(processor),
        "backend": state.get("computation_backend", "cpu"),
        "rater_values": _extract_rater_values(processor),
        "quality_parameter": state.get("quality_parameter", ""),
        "adaptive_contamination_rate": adaptive.get(
            "contamination_rate", math.nan
        ),
    }


def _fit_or_load_clean(
    output_dir: Path,
    frame: pd.DataFrame,
    raters: Sequence[str],
    method_id: str,
    method_config: Dict[str, Any],
    signature: str,
    device: str,
    recompute: bool,
) -> Dict[str, Any]:
    ranking_path = output_dir / "clean_ranking.csv"
    rater_path = output_dir / "clean_rater_scores.csv"
    metadata_path = output_dir / "clean_metadata.json"
    if (
        not recompute
        and ranking_path.exists()
        and rater_path.exists()
        and metadata_path.exists()
    ):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("data_signature") == signature:
            rater_frame = pd.read_csv(rater_path)
            return {
                "ranking": pd.read_csv(ranking_path),
                "rater_values": dict(
                    zip(
                        rater_frame.get("rater", pd.Series(dtype=str)).astype(
                            str
                        ),
                        pd.to_numeric(
                            rater_frame.get(
                                "value", pd.Series(dtype=float)
                            ),
                            errors="coerce",
                        ),
                    )
                ),
                **metadata,
            }

    result = _fit_method(frame, raters, method_config, device)
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_csv(ranking_path, result["ranking"])
    rater_frame = pd.DataFrame(
        sorted(result["rater_values"].items()), columns=["rater", "value"]
    )
    _atomic_csv(rater_path, rater_frame)
    metadata = {
        "data_signature": signature,
        "method_id": method_id,
        "method": method_config["label"],
        "model": method_config["model"],
        "shared_rater": method_config["shared_rater"],
        "elapsed": result["elapsed"],
        "fit_status": result["fit_status"],
        "backend": result["backend"],
        "quality_parameter": result["quality_parameter"],
        "adaptive_contamination_rate": result[
            "adaptive_contamination_rate"
        ],
    }
    _json_dump(metadata_path, metadata)
    return {
        "ranking": result["ranking"],
        "rater_values": result["rater_values"],
        **metadata,
    }


def _prior_mode(alpha: float, beta: float) -> float:
    if alpha > 1.0 and beta > 1.0:
        return (alpha - 1.0) / (alpha + beta - 2.0)
    return alpha / (alpha + beta)


def _label_correctness_posteriors(
    frame: pd.DataFrame,
    ranking: pd.DataFrame,
    rater_values: Dict[str, float],
) -> Tuple[pd.DataFrame, float]:
    scores = ranking.set_index(
        ranking["Method"].astype(str)
    )["ELO Score"].astype(float)
    decisive = frame["answerValue"].isin({"A", "B"})
    usable = (
        decisive
        & frame["methodA"].isin(scores.index)
        & frame["methodB"].isin(scores.index)
    )
    work = frame.loc[
        usable,
        ["__row_id", "answerer", "methodA", "methodB", "answerValue"],
    ].copy()
    if work.empty:
        return work.assign(correctness_posterior=pd.Series(dtype=float)), math.nan

    elo_a = work["methodA"].map(scores).to_numpy(float)
    elo_b = work["methodB"].map(scores).to_numpy(float)
    probability_a = 1.0 / (
        1.0 + np.power(10.0, (elo_b - elo_a) / ELO_SCALE_FACTOR)
    )
    observed_preference_probability = np.where(
        work["answerValue"].eq("A").to_numpy(), probability_a, 1 - probability_a
    )
    default_eta = _prior_mode(BETA_PRIOR_ALPHA, BETA_PRIOR_BETA)
    eta = (
        work["answerer"]
        .map({str(k): float(v) for k, v in rater_values.items()})
        .fillna(default_eta)
        .to_numpy(float)
    )
    numerator = eta * observed_preference_probability
    denominator = numerator + (1 - eta) * (
        1 - observed_preference_probability
    )
    posterior = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, 0.5),
        where=denominator > 0,
    )
    work["model_observed_preference_probability"] = (
        observed_preference_probability
    )
    work["rater_eta"] = eta
    work["correctness_posterior"] = posterior
    return work, float(np.mean(1 - posterior))


def _ranking_metrics(
    ranking: pd.DataFrame,
    reference: pd.DataFrame,
) -> Dict[str, float]:
    ranking_scores = (
        ranking.assign(_name=ranking["Method"].astype(str))
        .drop_duplicates("_name")
        .set_index("_name")["ELO Score"]
        .astype(float)
    )
    reference_scores = (
        reference.assign(_name=reference["Method"].astype(str))
        .drop_duplicates("_name")
        .set_index("_name")["ELO Score"]
        .astype(float)
    )
    common = reference_scores.index.intersection(ranking_scores.index)
    coverage = len(common) / len(reference_scores) if len(reference_scores) else math.nan
    reference_order = reference_scores.sort_values(ascending=False).index.tolist()
    ranking_order = ranking_scores.sort_values(ascending=False).index.tolist()
    top_k = min(3, len(reference_order))
    metrics = {
        "top1_agreement": float(
            bool(reference_order)
            and bool(ranking_order)
            and reference_order[0] == ranking_order[0]
        ),
        "top3_overlap": (
            len(set(reference_order[:top_k]) & set(ranking_order[:top_k]))
            / top_k
            if top_k
            else math.nan
        ),
        "ordered_top3_agreement": float(
            reference_order[:top_k] == ranking_order[:top_k]
        )
        if top_k
        else math.nan,
        "pearson": math.nan,
        "spearman": math.nan,
        "kendall_tau": math.nan,
        "rank_mae": math.nan,
        "pairwise_consistency": math.nan,
        "elo_rmse": math.nan,
        "elo_max_abs_change": math.nan,
        "method_coverage": coverage,
        "ranked_methods": len(ranking_scores),
        "reference_methods": len(reference_scores),
    }
    if coverage < 1.0 - 1e-12 or len(common) < 2:
        return metrics

    candidate = ranking_scores.loc[common]
    baseline = reference_scores.loc[common]
    metrics["pearson"] = float(np.corrcoef(candidate, baseline)[0, 1])
    metrics["spearman"] = float(spearmanr(candidate, baseline).correlation)
    metrics["kendall_tau"] = float(kendalltau(candidate, baseline).correlation)

    candidate_rank = candidate.rank(ascending=False, method="average")
    baseline_rank = baseline.rank(ascending=False, method="average")
    metrics["rank_mae"] = float(np.mean(np.abs(candidate_rank - baseline_rank)))

    candidate_array = candidate.to_numpy(float)
    baseline_array = baseline.to_numpy(float)
    upper = np.triu_indices(len(common), k=1)
    candidate_pairs = np.sign(
        candidate_array[:, None] - candidate_array[None, :]
    )[upper]
    baseline_pairs = np.sign(
        baseline_array[:, None] - baseline_array[None, :]
    )[upper]
    metrics["pairwise_consistency"] = float(
        np.mean(candidate_pairs == baseline_pairs)
    )

    candidate_centered = candidate_array - candidate_array.mean()
    baseline_centered = baseline_array - baseline_array.mean()
    difference = candidate_centered - baseline_centered
    metrics["elo_rmse"] = float(np.sqrt(np.mean(difference**2)))
    metrics["elo_max_abs_change"] = float(np.max(np.abs(difference)))
    return metrics


def _binary_detection_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray,
) -> Dict[str, float]:
    positives = int(labels.sum())
    negatives = len(labels) - positives
    true_positive = int(np.sum(predictions & labels))
    false_positive = int(np.sum(predictions & ~labels))
    false_negative = int(np.sum(~predictions & labels))
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else math.nan
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    if positives and negatives:
        ranks = rankdata(scores, method="average")
        auroc = (
            ranks[labels].sum() - positives * (positives + 1) / 2
        ) / (positives * negatives)

        order = np.argsort(-scores, kind="mergesort")
        sorted_scores = scores[order]
        sorted_labels = labels[order].astype(int)
        group_ends = np.r_[
            np.flatnonzero(np.diff(sorted_scores) != 0),
            len(sorted_scores) - 1,
        ]
        cumulative_tp = np.cumsum(sorted_labels)
        cumulative_fp = np.cumsum(1 - sorted_labels)
        tp = cumulative_tp[group_ends]
        fp = cumulative_fp[group_ends]
        recall_curve = tp / positives
        precision_curve = tp / np.maximum(tp + fp, 1)
        auprc = float(
            np.sum(
                np.diff(np.r_[0.0, recall_curve]) * precision_curve
            )
        )
    else:
        auroc = math.nan
        auprc = math.nan

    return {
        "predicted_anomalous_raters": int(predictions.sum()),
        "anomaly_precision": precision,
        "anomaly_recall": recall,
        "anomaly_f1": f1,
        "anomaly_auroc": float(auroc),
        "anomaly_auprc": float(auprc),
    }


def _anomaly_metrics(
    method_id: str,
    method_config: Dict[str, Any],
    all_raters: Sequence[str],
    selected_raters: Sequence[str],
    rater_values: Dict[str, float],
) -> Dict[str, Any]:
    unsupported = {
        "rater_score_supported": False,
        "rater_score_coverage": math.nan,
        "predicted_anomalous_raters": math.nan,
        "anomaly_precision": math.nan,
        "anomaly_recall": math.nan,
        "anomaly_f1": math.nan,
        "anomaly_auroc": math.nan,
        "anomaly_auprc": math.nan,
    }
    if (
        method_id not in RATER_AWARE_METHODS
        or method_config["shared_rater"]
        or not rater_values
    ):
        return unsupported

    available = [rater for rater in all_raters if rater in rater_values]
    if not available:
        return unsupported
    selected = set(selected_raters)
    labels = np.asarray([rater in selected for rater in available], dtype=bool)
    values = np.asarray([rater_values[rater] for rater in available], dtype=float)
    if method_id == "am_elo":
        anomaly_scores = -values
        predictions = values < 0.0
    else:
        anomaly_scores = 1.0 - values
        predictions = values < 0.5

    metrics = _binary_detection_metrics(labels, anomaly_scores, predictions)
    metrics.update(
        {
            "rater_score_supported": True,
            "rater_score_coverage": len(available) / len(all_raters),
        }
    )
    return metrics


def _base_run_record(
    dataset_id: str,
    dataset_settings: Dict[str, Any],
    method_id: str,
    method_config: Dict[str, Any],
    condition_id: str,
    perturbation: str,
    epsilon_zero: float,
) -> Dict[str, Any]:
    record = {field: math.nan for field in RUN_FIELDS}
    record.update(
        {
            "dataset_id": dataset_id,
            "dataset": dataset_settings["label"],
            "detection_unit": dataset_settings.get(
                "detection_unit", "rater"
            ),
            "method_id": method_id,
            "method": method_config["label"],
            "model": method_config["model"],
            "shared_rater": method_config["shared_rater"],
            "condition_id": condition_id,
            "perturbation": perturbation,
            "estimated_original_error_rate": epsilon_zero,
            "status": "failed",
            "fit_status": "not_run",
            "backend": "",
            "error_type": "",
            "error_message": "",
            "ranked_methods": 0,
            "reference_methods": 0,
            "rater_score_supported": False,
        }
    )
    return record


def _worker_init(
    frame: pd.DataFrame,
    raters: Sequence[str],
    dataset_id: str,
    dataset_settings: Dict[str, Any],
    method_configs: Dict[str, Dict[str, Any]],
    clean_rankings: Dict[str, pd.DataFrame],
    epsilon_zero: float,
    device: str,
) -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = {
        "frame": frame,
        "raters": list(raters),
        "dataset_id": dataset_id,
        "dataset_settings": dataset_settings,
        "method_configs": method_configs,
        "clean_rankings": clean_rankings,
        "epsilon_zero": epsilon_zero,
        "device": device,
    }


def _run_worker(task: Dict[str, Any]) -> Dict[str, Any]:
    context = _WORKER_CONTEXT
    method_id = task["method_id"]
    method_config = context["method_configs"][method_id]
    perturbation = task["perturbation"]
    condition_id = f"{perturbation}:{task['level_id']}:s{task['seed']}"
    record = _base_run_record(
        context["dataset_id"],
        context["dataset_settings"],
        method_id,
        method_config,
        condition_id,
        perturbation,
        context["epsilon_zero"],
    )
    selected_raters = task["selected_raters_list"]
    record.update(
        {
            "seed": task["seed"],
            "level_id": task["level_id"],
            "requested_rater_fraction": task[
                "requested_rater_fraction"
            ],
            "requested_rater_count": task["requested_rater_count"],
            "selected_raters": len(selected_raters),
            "total_raters": len(context["raters"]),
            "actual_rater_fraction": task["actual_rater_fraction"],
        }
    )
    try:
        perturbed, diagnostics = _apply_perturbation(
            context["frame"],
            selected_raters,
            perturbation,
            int(task["seed"]),
            condition_id,
        )
        record.update(diagnostics)
        epsilon_zero = context["epsilon_zero"]
        record["conservative_total_pollution"] = (
            epsilon_zero + diagnostics["changed_row_fraction"]
            if math.isfinite(epsilon_zero)
            else math.nan
        )
        fit = _fit_method(
            perturbed,
            context["raters"],
            method_config,
            context["device"],
        )
        record.update(
            _ranking_metrics(
                fit["ranking"], context["clean_rankings"][method_id]
            )
        )
        record.update(
            _anomaly_metrics(
                method_id,
                method_config,
                context["raters"],
                selected_raters,
                fit["rater_values"],
            )
        )
        record.update(
            {
                "status": "success",
                "fit_status": fit["fit_status"],
                "backend": fit["backend"],
                "time_seconds": fit["elapsed"],
                "adaptive_contamination_rate": fit[
                    "adaptive_contamination_rate"
                ],
            }
        )
        ranking_rows = []
        for rank, row in enumerate(
            fit["ranking"].itertuples(index=False), start=1
        ):
            ranking_rows.append(
                {
                    "dataset_id": context["dataset_id"],
                    "dataset": context["dataset_settings"]["label"],
                    "method_id": method_id,
                    "method": method_config["label"],
                    "condition_id": condition_id,
                    "perturbation": perturbation,
                    "seed": task["seed"],
                    "level_id": task["level_id"],
                    "requested_rater_fraction": task[
                        "requested_rater_fraction"
                    ],
                    "actual_rater_fraction": task[
                        "actual_rater_fraction"
                    ],
                    "model_name": str(getattr(row, "Method")),
                    "elo_score": float(getattr(row, "_1")),
                    "rank": rank,
                }
            )
        return {"record": record, "rankings": ranking_rows}
    except Exception as exc:
        record["error_type"] = type(exc).__name__
        record["error_message"] = " | ".join(
            line.strip() for line in traceback.format_exc(limit=5).splitlines()
        )
        return {"record": record, "rankings": []}


def _read_runs(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=RUN_FIELDS)
    frame = pd.read_csv(path, on_bad_lines="skip")
    for field in RUN_FIELDS:
        if field not in frame.columns:
            frame[field] = math.nan
    return frame[RUN_FIELDS].drop_duplicates("condition_id", keep="last")


def _read_rankings(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=RANKING_FIELDS)
    frame = pd.read_csv(path, on_bad_lines="skip")
    for field in RANKING_FIELDS:
        if field not in frame.columns:
            frame[field] = math.nan
    return frame[RANKING_FIELDS].drop_duplicates(
        ["condition_id", "model_name"], keep="last"
    )


def _append_dict_rows(
    path: Path,
    rows: Iterable[Dict[str, Any]],
    fields: Sequence[str],
) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
        handle.flush()


def _clean_run_record(
    dataset_id: str,
    dataset_settings: Dict[str, Any],
    method_id: str,
    method_config: Dict[str, Any],
    clean: Dict[str, Any],
    total_rows: int,
    total_raters: int,
    ties: int,
    epsilon_zero: float,
) -> Dict[str, Any]:
    record = _base_run_record(
        dataset_id,
        dataset_settings,
        method_id,
        method_config,
        "clean",
        "clean",
        epsilon_zero,
    )
    record.update(
        {
            "seed": -1,
            "level_id": "clean",
            "requested_rater_fraction": 0.0,
            "requested_rater_count": 0,
            "selected_raters": 0,
            "total_raters": total_raters,
            "actual_rater_fraction": 0.0,
            "selected_rows": 0,
            "selected_row_fraction": 0.0,
            "changed_rows": 0,
            "changed_row_fraction": 0.0,
            "ties_before": ties,
            "ties_after": ties,
            "tie_fraction_before": ties / total_rows,
            "tie_fraction_after": ties / total_rows,
            "conservative_total_pollution": epsilon_zero,
            "status": "success",
            "fit_status": clean["fit_status"],
            "backend": clean["backend"],
            "time_seconds": clean["elapsed"],
            "top1_agreement": 1.0,
            "top3_overlap": 1.0,
            "ordered_top3_agreement": 1.0,
            "pearson": 1.0,
            "spearman": 1.0,
            "kendall_tau": 1.0,
            "rank_mae": 0.0,
            "pairwise_consistency": 1.0,
            "elo_rmse": 0.0,
            "elo_max_abs_change": 0.0,
            "method_coverage": 1.0,
            "ranked_methods": len(clean["ranking"]),
            "reference_methods": len(clean["ranking"]),
            "rater_score_supported": (
                method_id in RATER_AWARE_METHODS
                and not method_config["shared_rater"]
            ),
            "rater_score_coverage": (
                len(clean["rater_values"]) / total_raters
                if clean["rater_values"]
                else math.nan
            ),
            "adaptive_contamination_rate": clean.get(
                "adaptive_contamination_rate", math.nan
            ),
        }
    )
    return record


def _clean_ranking_rows(
    dataset_id: str,
    dataset_settings: Dict[str, Any],
    method_id: str,
    method_config: Dict[str, Any],
    ranking: pd.DataFrame,
) -> List[Dict[str, Any]]:
    rows = []
    for rank, (_, item) in enumerate(ranking.iterrows(), start=1):
        rows.append(
            {
                "dataset_id": dataset_id,
                "dataset": dataset_settings["label"],
                "method_id": method_id,
                "method": method_config["label"],
                "condition_id": "clean",
                "perturbation": "clean",
                "seed": -1,
                "level_id": "clean",
                "requested_rater_fraction": 0.0,
                "actual_rater_fraction": 0.0,
                "model_name": str(item["Method"]),
                "elo_score": float(item["ELO Score"]),
                "rank": rank,
            }
        )
    return rows


def _prepare_method_output(
    output_dir: Path,
    clean_record: Dict[str, Any],
    clean_ranking_rows: List[Dict[str, Any]],
) -> None:
    runs_path = output_dir / "runs.csv"
    rankings_path = output_dir / "rankings.csv"
    runs = _read_runs(runs_path)
    if "clean" not in set(runs["condition_id"].astype(str)):
        _append_dict_rows(runs_path, [clean_record], RUN_FIELDS)
    rankings = _read_rankings(rankings_path)
    if "clean" not in set(rankings["condition_id"].astype(str)):
        _append_dict_rows(
            rankings_path, clean_ranking_rows, RANKING_FIELDS
        )


def _summarize_frame(runs: pd.DataFrame) -> pd.DataFrame:
    if runs.empty:
        return pd.DataFrame()
    group_columns = [
        "dataset_id",
        "dataset",
        "detection_unit",
        "method_id",
        "method",
        "model",
        "shared_rater",
        "perturbation",
        "level_id",
        "requested_rater_fraction",
        "requested_rater_count",
    ]
    summaries: List[Dict[str, Any]] = []
    for keys, group in runs.groupby(group_columns, dropna=False, sort=True):
        row = dict(zip(group_columns, keys))
        successful = group[group["status"].eq("success")]
        row.update(
            {
                "runs": len(group),
                "successful_runs": len(successful),
                "failed_runs": int((group["status"] != "success").sum()),
                "failure_rate": float(
                    (group["status"] != "success").mean()
                ),
                "optimizer_warning_runs": int(
                    successful["fit_status"].isin(
                        {"optimizer_warning", "iteration_limit"}
                    ).sum()
                ),
            }
        )
        for metric in SUMMARY_METRICS:
            values = pd.to_numeric(
                successful[metric], errors="coerce"
            ).dropna()
            row[f"{metric}_mean"] = (
                float(values.mean()) if len(values) else math.nan
            )
            row[f"{metric}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
            row[f"{metric}_valid_runs"] = len(values)
        summaries.append(row)
    return pd.DataFrame(summaries).sort_values(
        [
            "dataset_id",
            "method_id",
            "perturbation",
            "requested_rater_fraction",
        ]
    )


def summarize_results(root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for path in sorted(root.glob("*/*/runs.csv")):
        frame = _read_runs(path)
        if not frame.empty:
            frames.append(frame)
    runs = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    full = _summarize_frame(runs)
    _atomic_csv(root / "summary.csv", full)

    if runs.empty:
        restricted_runs = runs
    else:
        total = pd.to_numeric(
            runs["conservative_total_pollution"], errors="coerce"
        )
        restricted_runs = runs[
            runs["perturbation"].eq("clean") | total.le(0.5)
        ].copy()
    restricted = _summarize_frame(restricted_runs)
    _atomic_csv(root / "summary_under_50pct.csv", restricted)
    return full, restricted


def _protocol_identity(
    path: Path,
    signature: str,
    settings: Dict[str, Any],
    methods: Sequence[str],
) -> Dict[str, Any]:
    return {
        "input_csv": str(path.resolve()),
        "data_signature": signature,
        "seeds": settings["seeds"],
        "pollution_ratios": settings["pollution_ratios"],
        "polluted_rater_counts": settings.get("polluted_rater_counts"),
        "perturbations": settings["perturbations"],
        "methods": list(methods),
        "beta_prior_alpha": BETA_PRIOR_ALPHA,
        "beta_prior_beta": BETA_PRIOR_BETA,
        "tie_protocol": {
            "random": "A/B -> 50% tie, 50% opposite; tie unchanged",
            "equal": "all selected rows -> tie",
            "flip": "A/B reversed; tie unchanged",
            "mixed": "row-wise random choice of random/equal/flip",
        },
    }


def _ensure_protocol_identity(
    output_dir: Path,
    identity: Dict[str, Any],
) -> None:
    path = output_dir / "protocol_identity.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != identity and any(output_dir.glob("*/runs.csv")):
            raise ValueError(
                f"Protocol changed for {output_dir.name}; use a new "
                "--results-root instead of mixing incompatible runs"
            )
    _json_dump(path, identity)


def _tasks_for_dataset(
    manifest: pd.DataFrame,
    perturbations: Sequence[str],
    method_ids: Sequence[str],
    results_root: Path,
    dataset_id: str,
    retry_failed: bool,
) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    completed: Dict[str, set] = {}
    for method_id in method_ids:
        runs = _read_runs(
            results_root / dataset_id / method_id / "runs.csv"
        )
        if retry_failed:
            runs = runs[runs["status"].eq("success")]
        completed[method_id] = set(runs["condition_id"].astype(str))

    for item in manifest.to_dict(orient="records"):
        selected = json.loads(item["selected_raters_json"])
        for perturbation in perturbations:
            condition_id = (
                f"{perturbation}:{item['level_id']}:s{int(item['seed'])}"
            )
            for method_id in method_ids:
                if condition_id in completed[method_id]:
                    continue
                tasks.append(
                    {
                        **item,
                        "seed": int(item["seed"]),
                        "requested_rater_count": int(
                            item["requested_rater_count"]
                        ),
                        "selected_raters_list": selected,
                        "method_id": method_id,
                        "perturbation": perturbation,
                    }
                )
    return tasks


def _run_dataset(
    dataset_id: str,
    settings: Dict[str, Any],
    method_ids: Sequence[str],
    args: argparse.Namespace,
) -> None:
    csv_path = _project_path(settings["csv"])
    frame, raters = _load_data(csv_path)
    signature = _data_signature(csv_path, frame)
    dataset_root = args.results_root / dataset_id
    identity = _protocol_identity(
        csv_path, signature, settings, method_ids
    )
    _ensure_protocol_identity(dataset_root, identity)
    manifest = _build_manifest(
        args.results_root, dataset_id, raters, settings
    )

    print(
        f"\nDataset {settings['label']}: {len(frame)} rows, "
        f"{len(raters)} raters, {len(manifest)} rater selections"
    )

    clean_results: Dict[str, Dict[str, Any]] = {}
    method_configs = {method_id: METHODS[method_id] for method_id in method_ids}

    # Correctness Reverse is also the fixed clean-data error estimator.
    clean_method_ids = list(method_ids)
    if "correctness_reverse" not in clean_method_ids:
        clean_method_ids.append("correctness_reverse")
    for method_id in tqdm(clean_method_ids, desc=f"{dataset_id}/clean fits"):
        method_config = METHODS[method_id]
        clean_results[method_id] = _fit_or_load_clean(
            dataset_root / method_id,
            frame,
            raters,
            method_id,
            method_config,
            signature,
            args.device,
            args.recompute_clean,
        )

    correctness_clean = clean_results["correctness_reverse"]
    posteriors, epsilon_zero = _label_correctness_posteriors(
        frame,
        correctness_clean["ranking"],
        correctness_clean["rater_values"],
    )
    diagnostics_dir = dataset_root / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    posterior_path = diagnostics_dir / "clean_label_correctness.csv.gz"
    temporary = diagnostics_dir / "clean_label_correctness.csv.gz.tmp"
    posteriors.to_csv(temporary, index=False, compression="gzip")
    os.replace(temporary, posterior_path)
    _json_dump(
        diagnostics_dir / "clean_error_estimate.json",
        {
            "estimated_original_error_rate": epsilon_zero,
            "non_tie_rows": len(posteriors),
            "tie_rows": int(frame["answerValue"].eq("draw").sum()),
            "estimator": "correctness_reverse",
            "beta_prior_alpha": BETA_PRIOR_ALPHA,
            "beta_prior_beta": BETA_PRIOR_BETA,
        },
    )

    clean_rankings = {
        method_id: clean_results[method_id]["ranking"]
        for method_id in method_ids
    }
    ties = int(frame["answerValue"].eq("draw").sum())
    for method_id in method_ids:
        method_config = METHODS[method_id]
        output_dir = dataset_root / method_id
        clean = clean_results[method_id]
        _prepare_method_output(
            output_dir,
            _clean_run_record(
                dataset_id,
                settings,
                method_id,
                method_config,
                clean,
                len(frame),
                len(raters),
                ties,
                epsilon_zero,
            ),
            _clean_ranking_rows(
                dataset_id,
                settings,
                method_id,
                method_config,
                clean["ranking"],
            ),
        )

    tasks = _tasks_for_dataset(
        manifest,
        settings["perturbations"],
        method_ids,
        args.results_root,
        dataset_id,
        args.retry_failed,
    )
    if not tasks:
        print(f"[resume] {dataset_id}: already complete")
        summarize_results(args.results_root)
        return

    initializer_args = (
        frame,
        raters,
        dataset_id,
        settings,
        method_configs,
        clean_rankings,
        epsilon_zero,
        args.device,
    )
    use_cuda = resolve_device(args.device).startswith("cuda")
    workers = 1 if use_cuda else args.workers
    description = f"{dataset_id}/pollution fits"

    if workers == 1:
        _worker_init(*initializer_args)
        iterator = map(_run_worker, tasks)
        results = tqdm(iterator, total=len(tasks), desc=description)
        for result in results:
            method_id = result["record"]["method_id"]
            output_dir = dataset_root / method_id
            _append_dict_rows(
                output_dir / "rankings.csv",
                result["rankings"],
                RANKING_FIELDS,
            )
            _append_dict_rows(
                output_dir / "runs.csv",
                [result["record"]],
                RUN_FIELDS,
            )
    else:
        with Pool(
            processes=workers,
            initializer=_worker_init,
            initargs=initializer_args,
            maxtasksperchild=args.max_tasks_per_child,
        ) as pool:
            iterator = pool.imap_unordered(
                _run_worker, tasks, chunksize=1
            )
            for result in tqdm(
                iterator, total=len(tasks), desc=description
            ):
                method_id = result["record"]["method_id"]
                output_dir = dataset_root / method_id
                _append_dict_rows(
                    output_dir / "rankings.csv",
                    result["rankings"],
                    RANKING_FIELDS,
                )
                _append_dict_rows(
                    output_dir / "runs.csv",
                    [result["record"]],
                    RUN_FIELDS,
                )

    full, restricted = summarize_results(args.results_root)
    print(
        f"[done] {dataset_id}: summary={len(full)} rows, "
        f"under_50pct={len(restricted)} rows"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG
    )
    parser.add_argument(
        "--results-root", type=Path, default=DEFAULT_RESULTS
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["all"]
    )
    parser.add_argument(
        "--methods", nargs="+", default=["all"]
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="Override config seeds for selected datasets.",
    )
    parser.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        help="Override pollution ratios for non-WD datasets.",
    )
    parser.add_argument(
        "--perturbations",
        nargs="+",
        choices=sorted(PERTURBATIONS),
        help="Override perturbation types.",
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda"]
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-tasks-per-child", type=int, default=100)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--recompute-clean", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    args.results_root = args.results_root.resolve()
    _acquire_run_lock(args.results_root)

    config = _load_config(args.config.resolve())
    dataset_ids = _resolve_datasets(args.datasets, config)
    method_ids = _resolve_methods(args.methods)
    for dataset_id in dataset_ids:
        settings = config[dataset_id]
        if args.seeds:
            settings["seeds"] = list(args.seeds)
        if args.ratios and "polluted_rater_counts" not in settings:
            settings["pollution_ratios"] = list(args.ratios)
        if args.perturbations:
            settings["perturbations"] = list(args.perturbations)

    _json_dump(
        args.results_root / "experiment_config.json",
        {
            "config": str(args.config.resolve()),
            "datasets": dataset_ids,
            "methods": method_ids,
            "requested_device": args.device,
            "resolved_device": resolve_device(args.device),
            "workers": args.workers,
            "max_tasks_per_child": args.max_tasks_per_child,
            "seeds_override": args.seeds,
            "ratios_override": args.ratios,
            "perturbations_override": args.perturbations,
            "beta_prior_alpha": BETA_PRIOR_ALPHA,
            "beta_prior_beta": BETA_PRIOR_BETA,
        },
    )

    for dataset_id in dataset_ids:
        _run_dataset(
            dataset_id,
            config[dataset_id],
            method_ids,
            args,
        )

    full, restricted = summarize_results(args.results_root)
    print(f"\nFinished. Full summary: {args.results_root / 'summary.csv'}")
    print(
        "Restricted summary: "
        f"{args.results_root / 'summary_under_50pct.csv'}"
    )
    print(f"Rows: full={len(full)}, restricted={len(restricted)}")


if __name__ == "__main__":
    raise SystemExit(
        "This Experiment 5 runner is deprecated because its old summaries "
        "can be biased. Use run_pollution_experiment_v2.py."
    )
