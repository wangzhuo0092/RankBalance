#!/usr/bin/env python3
"""Experiment 4 v4: out-of-fold pairwise-preference prediction.

Every method uses the same fixed five-fold partition. Each eligible comparison
group is evaluated exactly once out of fold; groups that must remain in train
to preserve model coverage are explicitly marked train-only. Golden attention
checks are training-only calibration data for methods that model them and are
never evaluated.

Model-only and rater-conditioned probabilities are evaluated separately on all
test, seen-rater, unseen-rater, and dense-seen-rater scopes. Dense raters are
defined only by their training-fold annotation count. Per-row paired losses are
retained so conditioning improvements and confidence intervals do not require
model refitting.
"""

from __future__ import annotations

import argparse
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

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from elo_processor import DataProcessor, EloProcessor  # noqa: E402
from models import (  # noqa: E402
    BETA_PRIOR_ALPHA,
    BETA_PRIOR_BETA,
    ELO_INITIAL_SCORE,
    ELO_SCALE_FACTOR,
)
from run_cross_dataset_stability import METHODS  # noqa: E402
from run_data_scale_experiment import (  # noqa: E402
    _optimization_status,
    _resolve_methods,
)
from torch_bayesian_backend import cuda_metadata, resolve_device  # noqa: E402


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiment_4_datasets.json"
DEFAULT_RESULTS = (
    PROJECT_ROOT / "experiment_results" / "experiment_4_heldout_prediction_v4"
)
METRIC_COLUMNS = [
    "brier",
    "nll",
    "accuracy",
    "auc",
    "soft_brier_all",
    "soft_nll_all",
    "tie_brier",
    "brier_improvement",
    "nll_improvement",
]
DEFAULT_DENSE_RATER_MIN_ANNOTATIONS = 20
DEFAULT_FILTER_RATER_MIN_ANNOTATIONS = 20
DATA_VARIANTS = {"full", "filtered"}
DEFAULT_FOLDS = 5
DEFAULT_FOLD_SEED = 2023
CACHE_SCHEMA_VERSION = 5
CROWD_BT_MAX_MEAN_RANDOM_PROBABILITY = 0.80
CROWD_BT_MAX_SCORE_RANGE = 5000.0
CODE_SIGNATURE_PATHS = (
    Path(__file__).resolve(),
    PROJECT_ROOT / "experiments" / "run_data_scale_experiment.py",
    SRC_DIR / "models.py",
    SRC_DIR / "elo_processor.py",
    SRC_DIR / "bayesian_elo.py",
    SRC_DIR / "bayesian_elo_noise_vectorized.py",
    SRC_DIR / "bayesian_elo_calibrated.py",
    SRC_DIR / "am_elo.py",
    SRC_DIR / "google_elo_processor.py",
    SRC_DIR / "google_elo_wrapper.py",
    SRC_DIR / "google_elo" / "build" / "elo_main",
)
_WORKER_CONTEXTS: Dict[str, Dict[str, Any]] = {}
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
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            _json_safe(value),
            handle,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame, **kwargs: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, **kwargs)
    os.replace(temporary, path)


def _code_signature() -> str:
    """Hash every implementation artifact that can change fitted results."""
    digest = hashlib.sha256()
    digest.update(f"cache-schema:{CACHE_SCHEMA_VERSION}\n".encode("utf-8"))
    digest.update(
        json.dumps(METHODS, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    for path in CODE_SIGNATURE_PATHS:
        digest.update(
            f"\nfile:{path.relative_to(PROJECT_ROOT)}\n".encode("utf-8")
        )
        if not path.is_file():
            digest.update(b"<missing>")
            continue
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    return digest.hexdigest()


def _acquire_run_lock(root: Path) -> None:
    global _RUN_LOCK_HANDLE
    root.mkdir(parents=True, exist_ok=True)
    handle = (root / ".run.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(
            f"Another held-out experiment is writing to {root}"
        ) from exc
    handle.seek(0)
    handle.truncate()
    handle.write(f"{os.getpid()}\n")
    handle.flush()
    _RUN_LOCK_HANDLE = handle


def _load_config(path: Path) -> Dict[str, Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    defaults = raw.get("defaults", {})
    datasets = raw.get("datasets", {})
    if not datasets:
        raise ValueError("The experiment config contains no datasets")
    resolved: Dict[str, Dict[str, Any]] = {}
    for dataset_id, settings in datasets.items():
        value = {**defaults, **settings}
        missing = {"label", "csv"} - set(value)
        if missing:
            raise ValueError(
                f"Dataset {dataset_id!r} is missing {sorted(missing)}"
            )
        value["folds"] = int(value.get("folds", DEFAULT_FOLDS))
        value["fold_seed"] = int(
            value.get("fold_seed", DEFAULT_FOLD_SEED)
        )
        if value["folds"] < 2:
            raise ValueError(f"Dataset {dataset_id!r} needs at least 2 folds")
        resolved[dataset_id] = value
    return resolved


def _resolve_datasets(
    requested: Sequence[str], config: Dict[str, Dict[str, Any]]
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
            raise ValueError(
                f"Unknown datasets {unknown}; add them to {DEFAULT_CONFIG.name}"
            )
        selected = list(requested)
    return list(dict.fromkeys(selected))


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _golden_mask(frame: pd.DataFrame) -> pd.Series:
    if "isGolden" not in frame.columns:
        return pd.Series(False, index=frame.index)
    return (
        frame["isGolden"].notna()
        & frame["isGolden"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("true")
    )


def _load_data(path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    processor = DataProcessor(str(path))
    if not processor.load_data() or processor.df is None:
        raise RuntimeError(f"Failed to load {path}")
    raw = processor.df.reset_index(drop=True)
    is_golden = _golden_mask(raw)
    frame = raw.loc[~is_golden].reset_index(drop=True)
    golden = raw.loc[is_golden].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"No non-Golden rows remain in {path}")
    frame["__row_id"] = np.arange(len(frame), dtype=np.int64)
    return frame, golden


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


def _prior_mode(alpha: float, beta: float) -> float:
    if alpha > 1.0 and beta > 1.0:
        return (alpha - 1.0) / (alpha + beta - 2.0)
    return alpha / (alpha + beta)


def _build_data_variants(
    frame: pd.DataFrame,
    requested: Sequence[str],
    filter_min_annotations: int,
) -> Dict[str, Tuple[pd.DataFrame, Dict[str, Any]]]:
    """Create the full and optional dense-panel dataset versions."""
    counts = frame.groupby("answerer", dropna=False).size()
    original_rows = len(frame)
    original_raters = int(frame["answerer"].nunique())
    variants: Dict[str, Tuple[pd.DataFrame, Dict[str, Any]]] = {}

    if "full" in requested:
        variants["full"] = (
            frame.copy(),
            {
                "filter_min_annotations": 0,
                "original_rows": original_rows,
                "variant_rows": original_rows,
                "original_raters": original_raters,
                "variant_raters": original_raters,
                "retained_row_ratio": 1.0,
                "retained_rater_ratio": 1.0,
            },
        )

    if "filtered" in requested:
        eligible = set(counts[counts >= filter_min_annotations].index)
        filtered = frame[frame["answerer"].isin(eligible)].copy()
        variant_name = f"filtered_rater_{filter_min_annotations}"
        variant_rows = len(filtered)
        variant_raters = int(filtered["answerer"].nunique())
        variants[variant_name] = (
            filtered,
            {
                "filter_min_annotations": filter_min_annotations,
                "original_rows": original_rows,
                "variant_rows": variant_rows,
                "original_raters": original_raters,
                "variant_raters": variant_raters,
                "retained_row_ratio": (
                    variant_rows / original_rows if original_rows else math.nan
                ),
                "retained_rater_ratio": (
                    variant_raters / original_raters
                    if original_raters else math.nan
                ),
            },
        )

    return variants


def _split_group_values(
    frame: pd.DataFrame, comparison_id_column: Optional[str]
) -> pd.Series:
    if comparison_id_column and comparison_id_column in frame.columns:
        raw = frame[comparison_id_column].astype(object)
        missing = raw.isna() | raw.astype(str).str.strip().eq("")
        values = raw.astype(str)
        values.loc[missing] = (
            "__row_" + frame.loc[missing, "__row_id"].astype(str)
        )
        return values
    return "__row_" + frame["__row_id"].astype(str)


def _make_fold_assignment(
    frame: pd.DataFrame,
    folds: int,
    fold_seed: int,
    comparison_id_column: Optional[str],
) -> pd.DataFrame:
    groups = _split_group_values(frame, comparison_id_column)
    group_sizes = groups.value_counts(sort=False)
    if len(group_sizes) < folds:
        raise ValueError(
            f"Need at least {folds} comparison groups, found {len(group_sizes)}"
        )

    # Randomized greedy bin packing balances row counts even when a comparison
    # group contains several annotations.
    rng = np.random.default_rng(fold_seed)
    shuffled = rng.permutation(group_sizes.index.to_numpy())
    fold_rows = np.zeros(folds, dtype=np.int64)
    group_fold: Dict[str, int] = {}
    for group in shuffled:
        smallest = np.flatnonzero(fold_rows == fold_rows.min())
        fold = int(rng.choice(smallest))
        group_fold[str(group)] = fold
        fold_rows[fold] += int(group_sizes.loc[group])

    assigned = groups.astype(str).map(group_fold).to_numpy(dtype=np.int64)

    # A model that occurs only inside its assigned test fold cannot be scored.
    # Mark the minimum required groups train-only (-1). This preserves model
    # coverage and makes the achieved out-of-fold coverage explicit.
    while True:
        changed = False
        for fold in range(folds):
            is_test = assigned == fold
            train_models = set(frame.loc[~is_test, "methodA"].astype(str))
            train_models.update(frame.loc[~is_test, "methodB"].astype(str))
            test_models = set(frame.loc[is_test, "methodA"].astype(str))
            test_models.update(frame.loc[is_test, "methodB"].astype(str))
            for model in sorted(test_models - train_models):
                candidate = is_test & (
                    frame["methodA"].astype(str).eq(model).to_numpy()
                    | frame["methodB"].astype(str).eq(model).to_numpy()
                )
                group_to_keep = groups.loc[candidate].iloc[0]
                assigned[groups.eq(group_to_keep).to_numpy()] = -1
                changed = True
        if not changed:
            break

    empty_folds = [fold for fold in range(folds) if not np.any(assigned == fold)]
    if empty_folds:
        raise ValueError(
            f"Model-coverage repair emptied test folds {empty_folds}"
        )

    return pd.DataFrame({
        "row_id": frame["__row_id"].to_numpy(),
        "comparison_group": groups.astype(str).to_numpy(),
        "fold": assigned,
    })


def _ensure_fold_manifest(
    root: Path,
    dataset_id: str,
    frame: pd.DataFrame,
    folds: int,
    fold_seed: int,
    comparison_id_column: Optional[str],
    signature: str,
) -> pd.DataFrame:
    directory = root / "manifests" / dataset_id
    path = directory / "fold_assignments.csv"
    metadata_path = directory / "fold_assignments.json"
    protocol = {
        "data_signature": signature,
        "folds": folds,
        "fold_seed": fold_seed,
        "comparison_id_column": comparison_id_column,
    }
    if path.exists() and metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing != protocol:
            raise ValueError(
                f"Split protocol changed at {path}; use a new results root"
            )
        manifest = pd.read_csv(path)
        if len(manifest) != len(frame):
            raise ValueError(f"Split manifest length mismatch at {path}")
        return manifest
    manifest = _make_fold_assignment(
        frame, folds, fold_seed, comparison_id_column
    )
    _atomic_csv(path, manifest)
    _json_dump(metadata_path, protocol)
    return manifest


def _label_target(value: Any) -> float:
    label = str(value).strip().lower()
    if label in {"a", "model_a", "left", "winner_a"}:
        return 1.0
    if label in {"b", "model_b", "right", "winner_b"}:
        return 0.0
    if label in {"draw", "tie", "equal"}:
        return 0.5
    return math.nan


def _score_probability(
    method_a: pd.Series,
    method_b: pd.Series,
    scores: Dict[str, float],
) -> np.ndarray:
    score_a = method_a.astype(str).map(scores).to_numpy(dtype=float)
    score_b = method_b.astype(str).map(scores).to_numpy(dtype=float)
    return expit(
        math.log(10.0) * (score_a - score_b) / ELO_SCALE_FACTOR
    )


def _quality_values(state: Dict[str, Any]) -> Dict[str, float]:
    raw = state.get("qualities") or state.get("rater_qualities") or {}
    values: Dict[str, float] = {}
    for rater, value in raw.items():
        if isinstance(value, dict):
            value = value.get("value")
        try:
            values[str(rater)] = float(value)
        except (TypeError, ValueError):
            continue
    return values


def _rater_conditioned_probability(
    test: pd.DataFrame,
    model_probability: np.ndarray,
    method_config: Dict[str, Any],
    state: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Predict annotations, using the method prior for unseen raters."""
    model = method_config["model"]
    answerers = test["answerer"].astype(str)
    fallback = np.zeros(len(test), dtype=bool)

    # Methods without a rater model reduce to their model-only probability.
    if model in {"m_elo", "bayesian_elo"}:
        return model_probability.copy(), fallback, False

    if model == "google_elo":
        random_probabilities = {
            str(rater): float(value)
            for rater, value in state.get(
                "rater_random_probabilities", {}
            ).items()
        }
        random_probability = answerers.map(random_probabilities)
        fallback = random_probability.isna().to_numpy()
        # The Crowd-BT regularizer has its mode at zero random answering.
        random_probability = random_probability.fillna(0.0).to_numpy(float)
        probability = (
            (1.0 - random_probability) * model_probability
            + random_probability * 0.5
        )
        return probability, fallback, True

    qualities = state.get("qualities", {})
    if model == "am_elo":
        relative = {
            str(rater): float(value["relative_value"])
            for rater, value in qualities.items()
            if isinstance(value, dict) and "relative_value" in value
        }
        ability = answerers.map(relative)
        fallback = ability.isna().to_numpy()
        ability = ability.fillna(1.0).to_numpy(float)
        clean_margin = np.log(
            np.clip(model_probability, 1e-12, 1.0 - 1e-12)
            / np.clip(1.0 - model_probability, 1e-12, 1.0)
        )
        return expit(ability * clean_margin), fallback, True

    quality = _quality_values(state)
    prior_quality = _prior_mode(BETA_PRIOR_ALPHA, BETA_PRIOR_BETA)
    if method_config.get("shared_rater") and quality:
        eta = np.full(len(test), next(iter(quality.values())), dtype=float)
    else:
        mapped_quality = answerers.map(quality)
        fallback = mapped_quality.isna().to_numpy()
        eta = mapped_quality.fillna(prior_quality).to_numpy(float)

    if model == "bayesian_elo_noise":
        probability = eta * model_probability + (1.0 - eta) * 0.5
        return probability, fallback, True

    if model in {
        "bayesian_elo_correctness",
        "bayesian_elo_correctness_reverse",
        "bayesian_elo_adaptive_clip",
        "bayesian_elo_adaptive_flip",
    }:
        probability = (
            eta * model_probability
            + (1.0 - eta) * (1.0 - model_probability)
        )
        return probability, fallback, True

    return model_probability.copy(), fallback, False


def _binary_auc(target: np.ndarray, probability: np.ndarray) -> float:
    positive = target == 1.0
    negative = target == 0.0
    n_positive = int(positive.sum())
    n_negative = int(negative.sum())
    if n_positive == 0 or n_negative == 0:
        return math.nan
    ranks = rankdata(probability, method="average")
    rank_sum = float(ranks[positive].sum())
    return (
        rank_sum - n_positive * (n_positive + 1) / 2.0
    ) / (n_positive * n_negative)


def _evaluation_mask(
    predictions: pd.DataFrame,
    evaluation_scope: str,
    dense_rater_min_annotations: int,
) -> pd.Series:
    if evaluation_scope == "all_test":
        return pd.Series(True, index=predictions.index)
    if evaluation_scope == "seen_rater":
        return predictions["rater_seen_in_train"].astype(bool)
    if evaluation_scope == "unseen_rater":
        return ~predictions["rater_seen_in_train"].astype(bool)
    if evaluation_scope == "dense_seen_rater":
        return (
            predictions["rater_seen_in_train"].astype(bool)
            & predictions["rater_train_annotations"].ge(
                dense_rater_min_annotations
            )
        )
    raise ValueError(f"Unknown evaluation scope: {evaluation_scope}")


def _metric_record(
    predictions: pd.DataFrame,
    probability_column: str,
    prediction_scope: str,
    evaluation_scope: str,
    subgroup: str,
    dense_rater_min_annotations: int,
) -> Dict[str, Any]:
    selected = predictions.loc[
        _evaluation_mask(
            predictions, evaluation_scope, dense_rater_min_annotations
        )
    ]
    if subgroup != "overall":
        selected = selected[selected["subgroup"].eq(subgroup)]

    probability = selected[probability_column].to_numpy(dtype=float)
    model_probability = selected["p_model"].to_numpy(dtype=float)
    rater_probability = selected["p_rater"].to_numpy(dtype=float)
    target = selected["target"].to_numpy(dtype=float)
    valid = np.isfinite(probability) & np.isfinite(target)
    paired_valid = (
        valid
        & np.isfinite(model_probability)
        & np.isfinite(rater_probability)
    )
    probability = np.clip(probability[valid], 1e-12, 1.0 - 1e-12)
    target = target[valid]
    decisive = target != 0.5
    ties = target == 0.5
    conditioning_supported = bool(
        selected["rater_conditioned_supported"].any()
    ) if len(selected) else False
    fallback = (
        selected["rater_prior_fallback"].to_numpy(dtype=bool)[valid]
        if probability_column == "p_rater" and conditioning_supported
        else np.array([], dtype=bool)
    )
    record: Dict[str, Any] = {
        "prediction_scope": prediction_scope,
        "evaluation_scope": evaluation_scope,
        "subgroup": subgroup,
        "dense_rater_min_annotations": dense_rater_min_annotations,
        "evaluation_rows": len(selected),
        "evaluation_raters": int(selected["answerer"].nunique()),
        "predicted_rows": int(valid.sum()),
        "prediction_coverage": (
            float(valid.mean()) if len(selected) else math.nan
        ),
        "prior_fallback_rate": (
            float(fallback.mean()) if len(fallback) else math.nan
        ),
        "rater_conditioned_supported": conditioning_supported,
        "decisive_rows": int(decisive.sum()),
        "tie_rows": int(ties.sum()),
        "brier": math.nan,
        "nll": math.nan,
        "accuracy": math.nan,
        "auc": math.nan,
        "soft_brier_all": math.nan,
        "soft_nll_all": math.nan,
        "tie_brier": math.nan,
        "brier_improvement": math.nan,
        "nll_improvement": math.nan,
    }
    if not len(target):
        return record

    squared_error = (probability - target) ** 2
    soft_nll = -(
        target * np.log(probability)
        + (1.0 - target) * np.log(1.0 - probability)
    )
    record["soft_brier_all"] = float(squared_error.mean())
    record["soft_nll_all"] = float(soft_nll.mean())
    if decisive.any():
        record["brier"] = float(squared_error[decisive].mean())
        record["nll"] = float(soft_nll[decisive].mean())
        record["accuracy"] = float(
            ((probability[decisive] >= 0.5) == target[decisive]).mean()
        )
        record["auc"] = _binary_auc(
            target[decisive], probability[decisive]
        )
    if ties.any():
        record["tie_brier"] = float(squared_error[ties].mean())

    if prediction_scope == "rater_conditioned" and paired_valid.any():
        paired_target = selected["target"].to_numpy(dtype=float)[paired_valid]
        paired_decisive = paired_target != 0.5
        paired_model = np.clip(
            model_probability[paired_valid], 1e-12, 1.0 - 1e-12
        )
        paired_rater = np.clip(
            rater_probability[paired_valid], 1e-12, 1.0 - 1e-12
        )
        if paired_decisive.any():
            y = paired_target[paired_decisive]
            p_model = paired_model[paired_decisive]
            p_rater = paired_rater[paired_decisive]
            model_nll = -(
                y * np.log(p_model) + (1.0 - y) * np.log(1.0 - p_model)
            )
            rater_nll = -(
                y * np.log(p_rater) + (1.0 - y) * np.log(1.0 - p_rater)
            )
            record["nll_improvement"] = float(
                np.mean(model_nll - rater_nll)
            )
            record["brier_improvement"] = float(
                np.mean(
                    (p_model - y) ** 2 - (p_rater - y) ** 2
                )
            )
    return record


def _prediction_metrics(
    predictions: pd.DataFrame,
    dense_rater_min_annotations: int,
) -> List[Dict[str, Any]]:
    subgroups = ["overall"]
    distinct = sorted(
        value
        for value in predictions["subgroup"].dropna().astype(str).unique()
        if value != "overall"
    )
    if len(distinct) > 1:
        subgroups.extend(distinct)

    output: List[Dict[str, Any]] = []
    for evaluation_scope in (
        "all_test",
        "seen_rater",
        "unseen_rater",
        "dense_seen_rater",
    ):
        for subgroup in subgroups:
            output.append(
                _metric_record(
                    predictions,
                    "p_model",
                    "model_only",
                    evaluation_scope,
                    subgroup,
                    dense_rater_min_annotations,
                )
            )
            output.append(
                _metric_record(
                    predictions,
                    "p_rater",
                    "rater_conditioned",
                    evaluation_scope,
                    subgroup,
                    dense_rater_min_annotations,
                )
            )
    return output


def _cache_identity(
    context: Dict[str, Any], method_id: str, fold: int
) -> Dict[str, Any]:
    return {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "code_signature": context["code_signature"],
        "data_signature": context["data_signature"],
        "dataset_id": context["dataset_id"],
        "data_variant": context["data_variant"],
        "variant_metadata": context["variant_metadata"],
        "method_id": method_id,
        "method_config": METHODS[method_id],
        "fold": int(fold),
        "fold_protocol": context["fold_protocol"],
        "device": context["device"],
    }


def _validate_fit_and_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    predictions: pd.DataFrame,
    fit: Dict[str, Any],
    method_config: Dict[str, Any],
) -> None:
    """Reject stale, incomplete, non-finite, or degenerate fitted outputs."""
    required_columns = {
        "row_id",
        "comparison_group",
        "fold",
        "target",
        "p_model",
        "p_rater",
        "train_prevalence",
        "answerer",
        "rater_seen_in_train",
        "rater_prior_fallback",
        "rater_conditioned_supported",
        "model_nll_loss",
        "rater_nll_loss",
        "nll_improvement",
        "model_brier_loss",
        "rater_brier_loss",
        "brier_improvement",
    }
    missing_columns = sorted(required_columns - set(predictions.columns))
    if missing_columns:
        raise RuntimeError(
            f"Predictions are missing required columns: {missing_columns}"
        )
    if len(predictions) != len(test):
        raise RuntimeError(
            f"Prediction row mismatch: {len(predictions)} != {len(test)}"
        )
    predicted_ids = predictions["row_id"].astype(int).to_numpy()
    expected_ids = test["__row_id"].astype(int).to_numpy()
    if not np.array_equal(predicted_ids, expected_ids):
        raise RuntimeError("Prediction row IDs do not match the test split")

    expected_models = set(train["methodA"].dropna().astype(str))
    expected_models.update(train["methodB"].dropna().astype(str))
    expected_models.update(test["methodA"].dropna().astype(str))
    expected_models.update(test["methodB"].dropna().astype(str))
    raw_scores = fit.get("scores", {})
    score_keys = {str(key) for key in raw_scores}
    missing_models = sorted(expected_models - score_keys)
    if missing_models:
        preview = ", ".join(missing_models[:10])
        suffix = " ..." if len(missing_models) > 10 else ""
        raise RuntimeError(
            f"Fitted scores are missing {len(missing_models)} models: "
            f"{preview}{suffix}"
        )
    try:
        score_values = np.asarray(
            [float(raw_scores[name]) for name in sorted(expected_models)],
            dtype=float,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Fitted scores are not numeric") from exc
    if not np.isfinite(score_values).all():
        raise RuntimeError("Fitted scores contain NaN or infinity")

    for column in ("target", "p_model", "p_rater", "train_prevalence"):
        values = pd.to_numeric(
            predictions[column], errors="coerce"
        ).to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise RuntimeError(f"{column} contains NaN or infinity")
        if column != "target" and (
            np.any(values < 0.0) or np.any(values > 1.0)
        ):
            raise RuntimeError(f"{column} contains values outside [0, 1]")

    expected_target = (
        test["answerValue"].map(_label_target).to_numpy(dtype=float)
    )
    actual_target = predictions["target"].to_numpy(dtype=float)
    if (
        not np.isfinite(expected_target).all()
        or not np.array_equal(actual_target, expected_target)
    ):
        raise RuntimeError("Prediction targets do not match test labels")

    expected_groups = test["__comparison_group"].astype(str).to_numpy()
    actual_groups = predictions["comparison_group"].astype(str).to_numpy()
    if not np.array_equal(actual_groups, expected_groups):
        raise RuntimeError("Prediction comparison groups do not match test rows")
    expected_folds = test["__fold"].to_numpy(dtype=int)
    actual_folds = predictions["fold"].to_numpy(dtype=int)
    if not np.array_equal(actual_folds, expected_folds):
        raise RuntimeError("Prediction fold identifiers do not match test rows")

    decisive = np.isin(actual_target, [0.0, 1.0])
    p_model = np.clip(
        predictions["p_model"].to_numpy(dtype=float), 1e-12, 1.0 - 1e-12
    )
    p_rater = np.clip(
        predictions["p_rater"].to_numpy(dtype=float), 1e-12, 1.0 - 1e-12
    )
    expected_model_nll = -(
        actual_target * np.log(p_model)
        + (1.0 - actual_target) * np.log(1.0 - p_model)
    )
    expected_rater_nll = -(
        actual_target * np.log(p_rater)
        + (1.0 - actual_target) * np.log(1.0 - p_rater)
    )
    checks = {
        "model_nll_loss": expected_model_nll,
        "rater_nll_loss": expected_rater_nll,
        "nll_improvement": expected_model_nll - expected_rater_nll,
        "model_brier_loss": (p_model - actual_target) ** 2,
        "rater_brier_loss": (p_rater - actual_target) ** 2,
        "brier_improvement": (
            (p_model - actual_target) ** 2
            - (p_rater - actual_target) ** 2
        ),
    }
    for column, expected in checks.items():
        actual = predictions[column].to_numpy(dtype=float)
        if not np.isfinite(actual[decisive]).all() or not np.allclose(
            actual[decisive], expected[decisive], rtol=1e-10, atol=1e-12
        ):
            raise RuntimeError(f"{column} does not match stored probabilities")
        if np.isfinite(actual[~decisive]).any():
            raise RuntimeError(f"{column} must be missing for tie rows")

    validation: Dict[str, Any] = {
        "prediction_coverage": 1.0,
        "score_range": float(np.ptp(score_values)),
    }
    if method_config["model"] == "google_elo":
        raw_random = fit.get("rater_random_probabilities", {})
        try:
            random_values = np.asarray(
                [float(value) for value in raw_random.values()], dtype=float
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "Crowd-BT rater random probabilities are not numeric"
            ) from exc
        if len(random_values) and (
            not np.isfinite(random_values).all()
            or np.any(random_values < 0.0)
            or np.any(random_values > 1.0)
        ):
            raise RuntimeError(
                "Crowd-BT rater random probabilities are invalid"
            )
        random_mean = (
            float(random_values.mean()) if len(random_values) else math.nan
        )
        validation["crowd_bt_mean_random_probability"] = random_mean
        if (
            math.isfinite(random_mean)
            and random_mean >= CROWD_BT_MAX_MEAN_RANDOM_PROBABILITY
            and validation["score_range"] > CROWD_BT_MAX_SCORE_RANGE
        ):
            raise RuntimeError(
                "Degenerate Crowd-BT fit: mean random probability "
                f"{random_mean:.4f}, ELO range "
                f"{validation['score_range']:.1f}"
            )
    fit["validation"] = validation


def _fit_and_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    method_config: Dict[str, Any],
    device: str,
    golden_train: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    fit_frame = train
    golden_rows = 0
    if (
        method_config["model"] == "google_elo"
        and golden_train is not None
        and not golden_train.empty
    ):
        golden_rows = len(golden_train)
        fit_frame = pd.concat(
            [train, golden_train], ignore_index=True, sort=False
        )
    valid_users = fit_frame["answerer"].drop_duplicates().tolist()
    processor = EloProcessor(fit_frame, valid_users)
    ranking, elapsed = processor.process(
        df=fit_frame,
        valid_users=valid_users,
        model=method_config["model"],
        shared_rater=method_config["shared_rater"],
        device=device,
    )
    if ranking is None or ranking.empty:
        raise RuntimeError("The method returned an empty ranking")

    duplicated = ranking.loc[
        ranking["Method"].astype(str).duplicated(keep=False), "Method"
    ].astype(str)
    if not duplicated.empty:
        raise RuntimeError(
            "The method returned duplicate model identifiers: "
            + ", ".join(sorted(duplicated.unique()))
        )

    scores = dict(
        zip(
            ranking["Method"].astype(str),
            ranking["ELO Score"].astype(float),
        )
    )
    expected_models = set(train["methodA"].dropna().astype(str))
    expected_models.update(train["methodB"].dropna().astype(str))
    missing_models = sorted(expected_models - set(scores))
    if missing_models:
        preview = ", ".join(missing_models[:10])
        suffix = " ..." if len(missing_models) > 10 else ""
        raise RuntimeError(
            f"The fitted ranking is missing {len(missing_models)} training "
            f"models: {preview}{suffix}"
        )

    state = processor.metric.state if processor.metric is not None else {}
    model_probability = _score_probability(
        test["methodA"], test["methodB"], scores
    )
    (
        rater_probability,
        rater_prior_fallback,
        rater_conditioned_supported,
    ) = _rater_conditioned_probability(
        test, model_probability, method_config, state
    )

    train_answerers = train["answerer"].astype(str)
    test_answerers = test["answerer"].astype(str)
    all_answerers = pd.concat(
        [train_answerers, test_answerers], ignore_index=True
    )
    train_counts = train_answerers.value_counts()
    total_counts = all_answerers.value_counts()
    train_raters = set(train_answerers)
    train_targets = train["answerValue"].map(_label_target).to_numpy(dtype=float)
    train_decisive = train_targets != 0.5
    train_prevalence = (
        float(train_targets[train_decisive].mean())
        if train_decisive.any() else 0.5
    )

    predictions = pd.DataFrame(
        {
            "row_id": test["__row_id"].to_numpy(),
            "comparison_group": test["__comparison_group"].astype(str).to_numpy(),
            "fold": test["__fold"].to_numpy(dtype=int),
            "methodA": test["methodA"].astype(str).to_numpy(),
            "methodB": test["methodB"].astype(str).to_numpy(),
            "answerer": test_answerers.to_numpy(),
            "answerValue": test["answerValue"].astype(str).to_numpy(),
            "target": test["answerValue"].map(_label_target).to_numpy(),
            "subgroup": test["__subgroup"].astype(str).to_numpy(),
            "rater_seen_in_train": test_answerers.isin(train_raters).to_numpy(),
            "rater_train_annotations": (
                test_answerers.map(train_counts).fillna(0).to_numpy(int)
            ),
            "rater_total_annotations": (
                test_answerers.map(total_counts).fillna(0).to_numpy(int)
            ),
            "rater_prior_fallback": rater_prior_fallback,
            "rater_conditioned_supported": (
                rater_conditioned_supported
            ),
            "p_model": model_probability,
            "p_rater": rater_probability,
            "train_prevalence": train_prevalence,
        }
    )
    decisive = predictions["target"].isin([0.0, 1.0]).to_numpy()
    target = predictions["target"].to_numpy(dtype=float)
    p_model = np.clip(model_probability, 1e-12, 1.0 - 1e-12)
    p_rater = np.clip(rater_probability, 1e-12, 1.0 - 1e-12)
    model_nll = -(
        target * np.log(p_model) + (1.0 - target) * np.log(1.0 - p_model)
    )
    rater_nll = -(
        target * np.log(p_rater) + (1.0 - target) * np.log(1.0 - p_rater)
    )
    predictions["model_nll_loss"] = np.where(decisive, model_nll, np.nan)
    predictions["rater_nll_loss"] = np.where(decisive, rater_nll, np.nan)
    predictions["nll_improvement"] = np.where(
        decisive, model_nll - rater_nll, np.nan
    )
    predictions["model_brier_loss"] = np.where(
        decisive, (p_model - target) ** 2, np.nan
    )
    predictions["rater_brier_loss"] = np.where(
        decisive, (p_rater - target) ** 2, np.nan
    )
    predictions["brier_improvement"] = (
        predictions["model_brier_loss"]
        - predictions["rater_brier_loss"]
    )
    backend = state.get("computation_backend", "cpu")
    fit = {
        "status": "success",
        "fit_status": _optimization_status(processor),
        "backend": backend,
        "time_seconds": float(elapsed),
        "train_rows": len(train),
        "golden_calibration_rows": golden_rows,
        "fit_rows": len(fit_frame),
        "test_rows": len(test),
        "train_raters": int(train["answerer"].nunique()),
        "test_raters": int(test["answerer"].nunique()),
        "train_models": len(scores),
        "rater_conditioned_supported": rater_conditioned_supported,
        "scores": scores,
        "qualities": state.get("qualities", {}),
        "rater_random_probabilities": state.get(
            "rater_random_probabilities", {}
        ),
        "diagnostics": {
            key: state[key]
            for key in (
                "adaptive_clipping",
                "quality_parameter",
                "rater_mode",
                "update_mode",
                "weighting_mode",
                "iterations",
                "converged",
                "m_elo_optimization",
                "am_elo_optimization",
            )
            if key in state
        },
    }
    return predictions, fit, state


def _global_worker_init(
    contexts: Dict[str, Dict[str, Any]],
) -> None:
    global _WORKER_CONTEXTS
    _WORKER_CONTEXTS = contexts


def _worker_in_context(
    context: Dict[str, Any], task: Tuple[str, int]
) -> Dict[str, Any]:
    method_id, fold = task
    method_config = METHODS[method_id]
    output = (
        Path(context["results_root"])
        / context["dataset_id"]
        / context["data_variant"]
        / method_id
        / f"fold_{fold}"
    )
    prediction_path = output / "predictions.csv.gz"
    fit_path = output / "fit.json"
    cache_identity = _cache_identity(context, method_id, fold)
    try:
        manifest = context["manifest"]
        test_ids = set(
            manifest.loc[manifest["fold"].eq(fold), "row_id"].astype(int)
        )
        frame = context["frame"]
        is_test = frame["__row_id"].isin(test_ids)
        train = frame.loc[~is_test].copy()
        test = frame.loc[is_test].copy()
        group_map = manifest.set_index("row_id")["comparison_group"]
        test["__comparison_group"] = test["__row_id"].map(group_map)
        test["__fold"] = int(fold)

        predictions: Optional[pd.DataFrame] = None
        fit: Optional[Dict[str, Any]] = None
        cache_status = "fresh_fit"
        cache_rejection_reason = ""
        if (
            prediction_path.exists()
            and fit_path.exists()
            and not context["recompute"]
        ):
            try:
                cached_fit = json.loads(
                    fit_path.read_text(encoding="utf-8")
                )
                if cached_fit.get("cache_identity") != cache_identity:
                    raise RuntimeError(
                        "cached result identity does not match current code"
                    )
                cached_predictions = pd.read_csv(prediction_path)
                _validate_fit_and_predictions(
                    train,
                    test,
                    cached_predictions,
                    cached_fit,
                    method_config,
                )
                predictions = cached_predictions
                fit = cached_fit
                cache_status = "reused"
            except Exception as exc:
                cache_rejection_reason = (
                    f"{type(exc).__name__}: {exc}"
                )

        if predictions is None or fit is None:
            predictions, fit, _ = _fit_and_predict(
                train,
                test,
                method_config,
                context["device"],
                golden_train=context["golden_frame"],
            )
            fit["cache_identity"] = cache_identity
            fit["cache_status"] = "fresh_fit"
            if cache_rejection_reason:
                fit["cache_rejection_reason"] = cache_rejection_reason
            _validate_fit_and_predictions(
                train, test, predictions, fit, method_config
            )
            output.mkdir(parents=True, exist_ok=True)
            _atomic_csv(
                prediction_path,
                predictions,
                compression="gzip",
            )
            _json_dump(fit_path, fit)

        metric_rows = _prediction_metrics(
            predictions, context["dense_rater_min_annotations"]
        )
        base = {
            "dataset_id": context["dataset_id"],
            "dataset": context["dataset_label"],
            "data_variant": context["data_variant"],
            **context["variant_metadata"],
            "method_id": method_id,
            "method": method_config["label"],
            "model": method_config["model"],
            "shared_rater": method_config["shared_rater"],
            "fold": fold,
            "expected_runs": context["expected_runs"],
            "code_signature": context["code_signature"],
            "cache_status": cache_status,
            "status": fit.get("status", "success"),
            "fit_status": fit.get("fit_status", "completed"),
            "backend": fit.get("backend", "cpu"),
            "time_seconds": fit.get("time_seconds", math.nan),
            "train_rows": fit.get("train_rows", math.nan),
            "test_rows": fit.get("test_rows", math.nan),
            "train_raters": fit.get("train_raters", math.nan),
            "test_raters": fit.get("test_raters", math.nan),
            "train_models": fit.get("train_models", math.nan),
            "error_type": "",
            "error_message": "",
        }
        return {
            "method_id": method_id,
            "fold": fold,
            "rows": [{**base, **row} for row in metric_rows],
        }
    except Exception as exc:
        traceback.print_exc()
        return {
            "method_id": method_id,
            "fold": fold,
            "rows": [
                {
                    "dataset_id": context["dataset_id"],
                    "dataset": context["dataset_label"],
                    "data_variant": context["data_variant"],
                    **context["variant_metadata"],
                    "method_id": method_id,
                    "method": method_config["label"],
                    "model": method_config["model"],
                    "shared_rater": method_config["shared_rater"],
                    "fold": fold,
                    "expected_runs": context["expected_runs"],
                    "code_signature": context["code_signature"],
                    "cache_status": "failed",
                    "status": "failed",
                    "fit_status": "failed",
                    "backend": "unknown",
                    "prediction_scope": "model_only",
                    "evaluation_scope": "all_test",
                    "subgroup": "overall",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            ],
        }


def _global_worker(
    task: Tuple[str, str, int],
) -> Dict[str, Any]:
    context_key, method_id, fold = task
    result = _worker_in_context(
        _WORKER_CONTEXTS[context_key], (method_id, fold)
    )
    result["context_key"] = context_key
    return result

def _update_runs(
    path: Path, existing: pd.DataFrame, result: Dict[str, Any]
) -> pd.DataFrame:
    if not existing.empty:
        keep = ~(
            existing["method_id"].eq(result["method_id"])
            & existing["fold"].astype(int).eq(int(result["fold"]))
        )
        existing = existing.loc[keep]
    updated = pd.concat(
        [existing, pd.DataFrame(result["rows"])], ignore_index=True
    )
    _atomic_csv(path, updated)
    return updated


def _summarize(root: Path) -> pd.DataFrame:
    paths = sorted(
        path
        for path in root.glob("*/*/runs.csv")
        if path.is_file()
    )
    if not paths:
        return pd.DataFrame()
    runs = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    protocol_path = root / "protocol.json"
    if not protocol_path.is_file():
        raise RuntimeError(
            f"Missing protocol metadata at {protocol_path}"
        )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    active_signature = protocol.get("code_signature")
    if not active_signature:
        raise RuntimeError(
            "Legacy results have no code signature and cannot be summarized; "
            "rerun Experiment 4 into a new results directory"
        )
    if "code_signature" not in runs.columns:
        runs["code_signature"] = ""
    runs = runs[runs["code_signature"].eq(active_signature)].copy()
    successful = runs[runs["status"].eq("success")].copy()
    group_columns = [
        "dataset_id",
        "dataset",
        "data_variant",
        "method_id",
        "method",
        "prediction_scope",
        "evaluation_scope",
        "subgroup",
    ]
    summaries: List[Dict[str, Any]] = []
    for evaluation_set, selected in (
        ("all_successful", successful),
        (
            "converged_only",
            successful[
                successful["fit_status"].isin({"converged", "completed"})
            ],
        ),
    ):
        for keys, group in selected.groupby(group_columns, dropna=False):
            row = dict(zip(group_columns, keys))
            row["evaluation_set"] = evaluation_set
            row["code_signature"] = active_signature
            row["runs"] = int(group["fold"].nunique())
            row["optimizer_warning_rate"] = float(
                group["fit_status"].eq("optimizer_warning").mean()
            )
            for column in [
                "filter_min_annotations",
                "original_rows",
                "variant_rows",
                "original_raters",
                "variant_raters",
                "retained_row_ratio",
                "retained_rater_ratio",
                "dense_rater_min_annotations",
                "rater_conditioned_supported",
                "expected_runs",
            ]:
                row[column] = group[column].iloc[0]
            row["complete_run_set"] = (
                row["runs"] == int(row["expected_runs"])
            )
            for column in [
                "evaluation_rows",
                "evaluation_raters",
                "predicted_rows",
                "prediction_coverage",
                "prior_fallback_rate",
                *METRIC_COLUMNS,
                "time_seconds",
            ]:
                values = pd.to_numeric(group.get(column), errors="coerce")
                row[f"{column}_mean"] = float(values.mean())
                row[f"{column}_std"] = float(values.std(ddof=1))
            summaries.append(row)
    summary = pd.DataFrame(summaries)
    if not summary.empty:
        summary = summary.sort_values(
            [
                "dataset_id",
                "data_variant",
                "evaluation_scope",
                "prediction_scope",
                "subgroup",
                "method_id",
            ]
        )
    _atomic_csv(root / "summary.csv", summary)
    _write_report_tables(root, summary)
    return summary


def _paired_prediction_table(
    summary: pd.DataFrame,
    evaluation_scope: str,
    variant_prefix: str,
) -> pd.DataFrame:
    selected = summary[
        summary["evaluation_set"].eq("converged_only")
        & summary["complete_run_set"].astype(bool)
        & summary["subgroup"].eq("overall")
        & summary["evaluation_scope"].eq(evaluation_scope)
        & summary["data_variant"].str.startswith(variant_prefix)
    ].copy()
    if selected.empty:
        return pd.DataFrame()

    keys = [
        "dataset_id",
        "dataset",
        "data_variant",
        "method_id",
        "method",
    ]
    value_columns = [
        "runs",
        "evaluation_rows_mean",
        "evaluation_raters_mean",
        "prediction_coverage_mean",
        "prior_fallback_rate_mean",
        "brier_mean",
        "nll_mean",
        "accuracy_mean",
        "auc_mean",
        "soft_brier_all_mean",
        "soft_nll_all_mean",
        "tie_brier_mean",
        "brier_improvement_mean",
        "brier_improvement_std",
        "nll_improvement_mean",
        "nll_improvement_std",
        "time_seconds_mean",
    ]
    model_only = selected[
        selected["prediction_scope"].eq("model_only")
    ][keys + value_columns].copy()
    conditioned = selected[
        selected["prediction_scope"].eq("rater_conditioned")
    ][keys + value_columns + ["rater_conditioned_supported"]].copy()
    model_only = model_only.rename(
        columns={column: f"model_only_{column}" for column in value_columns}
    )
    conditioned = conditioned.rename(
        columns={
            column: f"rater_conditioned_{column}"
            for column in value_columns
        }
    )
    paired = model_only.merge(conditioned, on=keys, how="outer")
    paired["brier_improvement"] = paired[
        "rater_conditioned_brier_improvement_mean"
    ]
    paired["nll_improvement"] = paired[
        "rater_conditioned_nll_improvement_mean"
    ]
    paired["brier_improvement_fold_std"] = paired[
        "rater_conditioned_brier_improvement_std"
    ]
    paired["nll_improvement_fold_std"] = paired[
        "rater_conditioned_nll_improvement_std"
    ]
    paired["brier_difference_check"] = (
        paired["model_only_brier_mean"]
        - paired["rater_conditioned_brier_mean"]
    )
    paired["nll_difference_check"] = (
        paired["model_only_nll_mean"]
        - paired["rater_conditioned_nll_mean"]
    )
    paired["auc_improvement"] = (
        paired["rater_conditioned_auc_mean"]
        - paired["model_only_auc_mean"]
    )
    return paired.sort_values(keys).reset_index(drop=True)


def _write_report_tables(root: Path, summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    base = summary[
        summary["evaluation_set"].eq("converged_only")
        & summary["complete_run_set"].astype(bool)
        & summary["subgroup"].eq("overall")
    ].copy()
    table_1 = base[
        base["data_variant"].eq("full")
        & base["evaluation_scope"].eq("all_test")
        & base["prediction_scope"].eq("model_only")
    ]
    table_2 = _paired_prediction_table(summary, "seen_rater", "full")
    table_3 = _paired_prediction_table(
        summary, "dense_seen_rater", "full"
    )
    table_4 = _paired_prediction_table(summary, "unseen_rater", "full")
    _atomic_csv(root / "table_1_model_only_all_test.csv", table_1)
    _atomic_csv(root / "table_2_seen_rater_prediction.csv", table_2)
    _atomic_csv(root / "table_3_dense_seen_rater_20.csv", table_3)
    _atomic_csv(root / "table_4_unseen_rater_fallback.csv", table_4)


def _cluster_mean_ci(
    values: np.ndarray, groups: np.ndarray
) -> Tuple[float, float, float, float]:
    """Return a mean and a comparison-clustered 95% normal interval."""
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups).astype(str)
    valid = np.isfinite(values)
    values = values[valid]
    groups = groups[valid]
    if not len(values):
        return math.nan, math.nan, math.nan, math.nan
    mean = float(values.mean())
    grouped = pd.DataFrame({"value": values, "group": groups}).groupby(
        "group", sort=False
    )["value"].agg(["sum", "count"])
    cluster_count = len(grouped)
    if cluster_count < 2:
        return mean, math.nan, math.nan, math.nan
    centered_sums = (
        grouped["sum"].to_numpy(dtype=float)
        - mean * grouped["count"].to_numpy(dtype=float)
    )
    variance = (
        cluster_count
        / (cluster_count - 1.0)
        * float(np.square(centered_sums).sum())
        / float(len(values) ** 2)
    )
    standard_error = math.sqrt(max(variance, 0.0))
    return (
        mean,
        standard_error,
        mean - 1.96 * standard_error,
        mean + 1.96 * standard_error,
    )


def _summarize_oof(
    root: Path, dense_rater_min_annotations: int
) -> pd.DataFrame:
    """Pool disjoint fold predictions and compute paired official metrics."""
    protocol_path = root / "protocol.json"
    if not protocol_path.is_file():
        return pd.DataFrame()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    active_signature = protocol.get("code_signature")
    output: List[Dict[str, Any]] = []

    for method_root in sorted(root.glob("*/full/*")):
        if not method_root.is_dir() or method_root.name not in METHODS:
            continue
        dataset_id = method_root.parents[1].name
        method_id = method_root.name
        fold_directories = sorted(method_root.glob("fold_*"))
        fits: List[Dict[str, Any]] = []
        prediction_frames: List[pd.DataFrame] = []
        for directory in fold_directories:
            fit_path = directory / "fit.json"
            prediction_path = directory / "predictions.csv.gz"
            if not fit_path.is_file() or not prediction_path.is_file():
                continue
            fit = json.loads(fit_path.read_text(encoding="utf-8"))
            identity = fit.get("cache_identity", {})
            if identity.get("code_signature") != active_signature:
                continue
            fits.append(fit)
            prediction_frames.append(pd.read_csv(prediction_path))

        if not fits:
            continue
        expected_folds = int(
            fits[0].get("cache_identity", {})
            .get("fold_protocol", {})
            .get("folds", len(fits))
        )
        complete = len(fits) == expected_folds
        converged = all(
            fit.get("status") == "success"
            and fit.get("fit_status") in {"converged", "completed"}
            for fit in fits
        )
        if not complete or not converged:
            continue

        predictions = pd.concat(prediction_frames, ignore_index=True)
        if predictions["row_id"].duplicated().any():
            raise RuntimeError(
                f"Duplicate OOF row IDs for {dataset_id}/{method_id}"
            )
        method_config = METHODS[method_id]
        dataset_label = dataset_id
        runs_path = method_root.parent / "runs.csv"
        if runs_path.is_file():
            run_labels = pd.read_csv(
                runs_path, usecols=["dataset_id", "dataset"]
            ).drop_duplicates()
            labels = run_labels.loc[
                run_labels["dataset_id"].eq(dataset_id), "dataset"
            ]
            if len(labels):
                dataset_label = str(labels.iloc[0])

        for evaluation_scope in (
            "all_test",
            "seen_rater",
            "unseen_rater",
            "dense_seen_rater",
        ):
            selected = predictions.loc[
                _evaluation_mask(
                    predictions,
                    evaluation_scope,
                    dense_rater_min_annotations,
                )
            ].copy()
            decisive = selected["target"].isin([0.0, 1.0])
            scored = selected.loc[decisive].copy()
            groups = scored["comparison_group"].astype(str).to_numpy()
            nll_mean, nll_se, nll_low, nll_high = _cluster_mean_ci(
                scored["nll_improvement"].to_numpy(dtype=float), groups
            )
            brier_mean, brier_se, brier_low, brier_high = _cluster_mean_ci(
                scored["brier_improvement"].to_numpy(dtype=float), groups
            )

            target = scored["target"].to_numpy(dtype=float)
            p_model = scored["p_model"].to_numpy(dtype=float)
            p_rater = scored["p_rater"].to_numpy(dtype=float)
            prevalence = np.clip(
                scored["train_prevalence"].to_numpy(dtype=float),
                1e-12,
                1.0 - 1e-12,
            )
            prevalence_nll = -(
                target * np.log(prevalence)
                + (1.0 - target) * np.log(1.0 - prevalence)
            ) if len(scored) else np.array([], dtype=float)
            prevalence_brier = (
                (prevalence - target) ** 2
                if len(scored) else np.array([], dtype=float)
            )
            output.append({
                "dataset_id": dataset_id,
                "dataset": dataset_label,
                "method_id": method_id,
                "method": method_config["label"],
                "model": method_config["model"],
                "shared_rater": method_config["shared_rater"],
                "evaluation_scope": evaluation_scope,
                "folds": expected_folds,
                "evaluation_rows": len(selected),
                "decisive_rows": len(scored),
                "tie_rows": int((selected["target"] == 0.5).sum()),
                "evaluation_raters": int(selected["answerer"].nunique()),
                "comparison_groups": int(
                    selected["comparison_group"].nunique()
                ),
                "prediction_coverage": float(
                    np.isfinite(selected[["p_model", "p_rater"]])
                    .all(axis=1).mean()
                ) if len(selected) else math.nan,
                "prior_fallback_rate": float(
                    selected["rater_prior_fallback"].astype(bool).mean()
                ) if len(selected) and bool(
                    selected["rater_conditioned_supported"].any()
                ) else math.nan,
                "rater_conditioned_supported": bool(
                    selected["rater_conditioned_supported"].any()
                ) if len(selected) else False,
                "model_nll": float(scored["model_nll_loss"].mean()),
                "rater_nll": float(scored["rater_nll_loss"].mean()),
                "nll_improvement": nll_mean,
                "nll_improvement_cluster_se": nll_se,
                "nll_improvement_ci95_low": nll_low,
                "nll_improvement_ci95_high": nll_high,
                "model_brier": float(scored["model_brier_loss"].mean()),
                "rater_brier": float(scored["rater_brier_loss"].mean()),
                "brier_improvement": brier_mean,
                "brier_improvement_cluster_se": brier_se,
                "brier_improvement_ci95_low": brier_low,
                "brier_improvement_ci95_high": brier_high,
                "positive_nll_improvement_row_rate": float(
                    scored["nll_improvement"].gt(0).mean()
                ) if len(scored) else math.nan,
                "model_auc": (
                    _binary_auc(target, p_model) if len(scored) else math.nan
                ),
                "rater_auc": (
                    _binary_auc(target, p_rater) if len(scored) else math.nan
                ),
                "model_accuracy": float(
                    ((p_model >= 0.5) == target).mean()
                ) if len(scored) else math.nan,
                "rater_accuracy": float(
                    ((p_rater >= 0.5) == target).mean()
                ) if len(scored) else math.nan,
                "half_nll": math.log(2.0) if len(scored) else math.nan,
                "half_brier": 0.25 if len(scored) else math.nan,
                "train_prevalence_nll": float(prevalence_nll.mean())
                if len(prevalence_nll) else math.nan,
                "train_prevalence_brier": float(prevalence_brier.mean())
                if len(prevalence_brier) else math.nan,
                "model_nll_improvement_vs_half": (
                    math.log(2.0) - float(scored["model_nll_loss"].mean())
                    if len(scored) else math.nan
                ),
                "model_brier_improvement_vs_half": (
                    0.25 - float(scored["model_brier_loss"].mean())
                    if len(scored) else math.nan
                ),
                "fit_time_seconds_total": float(
                    sum(float(fit.get("time_seconds", 0.0)) for fit in fits)
                ),
            })

    result = pd.DataFrame(output)
    if not result.empty:
        result = result.sort_values(
            ["dataset_id", "evaluation_scope", "method_id"]
        ).reset_index(drop=True)
    _atomic_csv(root / "paired_oof_summary.csv", result)
    return result


def _write_protocol(
    root: Path,
    config_path: Path,
    dataset_ids: Sequence[str],
    method_ids: Sequence[str],
    device: str,
    data_variants: Sequence[str],
    dense_rater_min_annotations: int,
    filter_raters_min_annotations: int,
    folds_override: Optional[int],
    fold_seed_override: Optional[int],
    code_signature: str,
) -> None:
    _json_dump(
        root / "protocol.json",
        {
            "experiment": "held_out_preference_prediction",
            "protocol_version": 5,
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "code_signature": code_signature,
            "official_evaluation_set": "converged_only",
            "result_validation": {
                "prediction_coverage_required": 1.0,
                "finite_predictions_required": True,
                "crowd_bt_max_mean_random_probability": (
                    CROWD_BT_MAX_MEAN_RANDOM_PROBABILITY
                ),
                "crowd_bt_max_score_range": CROWD_BT_MAX_SCORE_RANGE,
            },
            "config": str(config_path.resolve()),
            "datasets": list(dataset_ids),
            "methods": list(method_ids),
            "device": device,
            "data_variants": list(data_variants),
            "cross_validation": {
                "type": "fixed_grouped_k_fold",
                "default_folds": DEFAULT_FOLDS,
                "default_fold_seed": DEFAULT_FOLD_SEED,
                "folds_override": folds_override,
                "fold_seed_override": fold_seed_override,
                "model_coverage_policy": (
                    "comparison groups that would leave a test model unseen "
                    "are marked train-only"
                ),
            },
            "dense_rater_min_annotations": (
                dense_rater_min_annotations
            ),
            "filter_raters_min_annotations": (
                filter_raters_min_annotations
            ),
            "golden_policy": (
                "never evaluated; training-only calibration for methods "
                "whose original likelihood models Golden checks"
            ),
            "targets": {
                "model_only": (
                    "held-out observed label using only fitted model scores"
                ),
                "rater_conditioned": (
                    "held-out observed label using model scores and the "
                    "fitted rater parameter"
                ),
            },
            "evaluation_scopes": {
                "all_test": "all test rows; unseen raters use the prior",
                "seen_rater": "test raters observed in training",
                "unseen_rater": (
                    "test raters absent from training; conditioned methods "
                    "use their declared prior fallback"
                ),
                "dense_seen_rater": (
                    "seen test raters with at least the configured number "
                    "of training-fold annotations"
                ),
            },
            "prior_fallback": {
                "m_elo_bayes_bt": "rater probability equals model-only",
                "am_elo": "relative ability 1",
                "crowd_bt": "random-answer probability 0",
                "bbq_correctness_adaptive": (
                    "Beta prior mode from models.py"
                ),
            },
            "tie_policy": {
                "primary_binary_metrics": "decisive A/B rows only",
                "soft_auxiliary_metrics": "ties use target 0.5",
            },
            "outputs": {
                "runs": "{dataset}/{variant}/runs.csv",
                "predictions": (
                    "{dataset}/{variant}/{method}/fold_{fold}/"
                    "predictions.csv.gz"
                ),
                "fit": (
                    "{dataset}/{variant}/{method}/fold_{fold}/fit.json"
                ),
                "summary": "summary.csv",
            },
        },
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--datasets", nargs="+", default=["default"],
        help="Dataset IDs, default, or all",
    )
    parser.add_argument(
        "--methods", nargs="+", default=["all"],
        help="Method IDs, baseline, ours, shared, paper, or all",
    )
    parser.add_argument(
        "--folds", type=int,
        help="Override the configured fold count (default: 5)",
    )
    parser.add_argument(
        "--fold-seed", type=int,
        help="Override the deterministic fold-assignment seed",
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto"
    )
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--data-variants",
        nargs="+",
        choices=sorted(DATA_VARIANTS),
        default=["full"],
        help="Run the full dataset, the filtered-rater dataset, or both",
    )
    parser.add_argument(
        "--dense-rater-min-annotations",
        type=int,
        default=DEFAULT_DENSE_RATER_MIN_ANNOTATIONS,
    )
    parser.add_argument(
        "--filter-raters-min-annotations",
        type=int,
        default=DEFAULT_FILTER_RATER_MIN_ANNOTATIONS,
    )
    parser.add_argument(
        "--summarize-only", action="store_true",
        help="Rebuild summary.csv from existing runs.csv files",
    )
    parser.add_argument(
        "--recompute", action="store_true",
        help=(
            "Refit selected dataset/method tasks even when cached outputs exist"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = args.results_root
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    if args.summarize_only:
        summary = _summarize(root)
        paired = _summarize_oof(
            root, args.dense_rater_min_annotations
        )
        print(f"Updated {root / 'summary.csv'} ({len(summary)} rows)")
        print(
            f"Updated {root / 'paired_oof_summary.csv'} "
            f"({len(paired)} rows)"
        )
        return

    if args.dense_rater_min_annotations <= 0:
        raise ValueError("dense-rater-min-annotations must be positive")
    if args.filter_raters_min_annotations <= 0:
        raise ValueError("filter-raters-min-annotations must be positive")
    if args.folds is not None and args.folds < 2:
        raise ValueError("folds must be at least 2")

    config = _load_config(args.config)
    dataset_ids = _resolve_datasets(args.datasets, config)
    method_ids = _resolve_methods(args.methods)
    code_signature = _code_signature()
    resolved_device = resolve_device(args.device)
    workers = max(1, int(args.workers))
    if resolved_device == "cuda" and workers != 1:
        print("CUDA execution uses one worker to avoid GPU contention.")
        workers = 1

    _acquire_run_lock(root)
    _write_protocol(
        root,
        args.config,
        dataset_ids,
        method_ids,
        resolved_device,
        args.data_variants,
        args.dense_rater_min_annotations,
        args.filter_raters_min_annotations,
        args.folds,
        args.fold_seed,
        code_signature,
    )
    hardware = cuda_metadata(resolved_device)
    hardware.update({
        "requested_workers": int(args.workers),
        "effective_workers": workers,
        "scheduler": "global_cross_dataset_pool",
    })
    _json_dump(root / "hardware.json", hardware)

    contexts: Dict[str, Dict[str, Any]] = {}
    runs_paths: Dict[str, Path] = {}
    runs_by_context: Dict[str, pd.DataFrame] = {}
    preparation_bar = tqdm(
        dataset_ids,
        desc="Preparing Experiment 4 datasets",
        unit="dataset",
        dynamic_ncols=True,
    )
    for dataset_id in preparation_bar:
        settings = config[dataset_id]
        preparation_bar.set_postfix(dataset=dataset_id)
        path = _project_path(settings["csv"])
        full_frame, full_golden = _load_data(path)
        subgroup_column = settings.get("subgroup_column")
        if subgroup_column and subgroup_column in full_frame.columns:
            full_frame["__subgroup"] = (
                full_frame[subgroup_column].fillna("missing").astype(str)
            )
        else:
            full_frame["__subgroup"] = "overall"

        variants = _build_data_variants(
            full_frame,
            args.data_variants,
            args.filter_raters_min_annotations,
        )
        folds = int(args.folds or settings["folds"])
        fold_seed = int(
            args.fold_seed
            if args.fold_seed is not None
            else settings["fold_seed"]
        )

        for data_variant, (frame, variant_metadata) in variants.items():
            eligible_raters = set(frame["answerer"].astype(str))
            variant_golden = full_golden[
                full_golden["answerer"].astype(str).isin(eligible_raters)
            ].copy()
            variant_metadata = {
                **variant_metadata,
                "golden_calibration_rows": len(variant_golden),
                "golden_calibration_raters": int(
                    variant_golden["answerer"].nunique()
                ),
            }
            variant_root = root / dataset_id / data_variant
            variant_root.mkdir(parents=True, exist_ok=True)
            metadata = {
                "dataset_id": dataset_id,
                "dataset": settings["label"],
                "data_variant": data_variant,
                "input_csv": str(path.resolve()),
                **variant_metadata,
            }
            _json_dump(variant_root / "dataset_metadata.json", metadata)

            model_count = int(
                pd.concat([frame["methodA"], frame["methodB"]])
                .dropna()
                .astype(str)
                .nunique()
            )
            if len(frame) < 2 or model_count < 2:
                metadata["status"] = "skipped"
                metadata["reason"] = (
                    "fewer than two rows or fewer than two models"
                )
                _json_dump(variant_root / "dataset_metadata.json", metadata)
                print(
                    f"[skip] {settings['label']}/{data_variant}: "
                    f"{metadata['reason']}",
                    flush=True,
                )
                continue

            signature = _data_signature(path, frame)
            manifest_key = f"{dataset_id}/{data_variant}"
            manifest = _ensure_fold_manifest(
                root,
                manifest_key,
                frame,
                folds,
                fold_seed,
                settings.get("comparison_id_column"),
                signature,
            )
            oof_rows = int(manifest["fold"].ge(0).sum())
            train_only_rows = int(manifest["fold"].lt(0).sum())
            metadata.update({
                "folds": folds,
                "fold_seed": fold_seed,
                "oof_rows": oof_rows,
                "train_only_rows": train_only_rows,
                "oof_coverage": oof_rows / len(frame),
            })
            _json_dump(variant_root / "dataset_metadata.json", metadata)
            runs_path = variant_root / "runs.csv"
            context_key = f"{dataset_id}/{data_variant}"
            runs_paths[context_key] = runs_path
            runs_by_context[context_key] = (
                pd.read_csv(runs_path)
                if runs_path.exists()
                else pd.DataFrame()
            )
            contexts[context_key] = {
                "dataset_id": dataset_id,
                "dataset_label": settings["label"],
                "data_variant": data_variant,
                "variant_metadata": variant_metadata,
                "golden_frame": variant_golden,
                "dense_rater_min_annotations": (
                    args.dense_rater_min_annotations
                ),
                "frame": frame,
                "manifest": manifest,
                "results_root": str(root),
                "device": resolved_device,
                "recompute": args.recompute,
                "code_signature": code_signature,
                "data_signature": signature,
                "expected_runs": folds,
                "fold_protocol": {
                    "folds": folds,
                    "fold_seed": fold_seed,
                    "comparison_id_column": settings.get(
                        "comparison_id_column"
                    ),
                },
            }
    preparation_bar.close()

    # A single cross-dataset pool prevents cores from going idle while one
    # dataset is waiting for a few long Crowd-BT fits.  Slow methods are queued
    # first, and larger datasets first within each method, so long-tail jobs
    # start as early as possible.  Context data are inherited copy-on-write by
    # forked Linux workers rather than serialized once per fit.
    context_order = sorted(
        contexts,
        key=lambda key: len(contexts[key]["frame"]),
        reverse=True,
    )
    tasks = [
        (context_key, method_id, fold)
        for method_id in method_ids
        for context_key in context_order
        for fold in range(contexts[context_key]["expected_runs"])
    ]
    progress = tqdm(
        total=len(tasks),
        desc="Experiment 4 global fits",
        unit="fit",
        dynamic_ncols=True,
        leave=True,
    )
    if workers == 1:
        _global_worker_init(contexts)
        iterator: Iterable[Dict[str, Any]] = map(_global_worker, tasks)
        for result in iterator:
            context_key = result.pop("context_key")
            runs_by_context[context_key] = _update_runs(
                runs_paths[context_key],
                runs_by_context[context_key],
                result,
            )
            progress.update()
    elif tasks:
        pool_size = min(workers, len(tasks))
        with Pool(
            processes=pool_size,
            initializer=_global_worker_init,
            initargs=(contexts,),
        ) as pool:
            for result in pool.imap_unordered(
                _global_worker, tasks, chunksize=1
            ):
                context_key = result.pop("context_key")
                runs_by_context[context_key] = _update_runs(
                    runs_paths[context_key],
                    runs_by_context[context_key],
                    result,
                )
                progress.update()
    progress.close()

    for context_key in context_order:
        context = contexts[context_key]
        print(
            f"[done] {context['dataset_label']}/"
            f"{context['data_variant']}: {len(context['frame'])} rows, "
            f"{context['expected_runs']} folds, "
            f"{len(method_ids)} methods",
            flush=True,
        )

    summary = _summarize(root)
    paired = _summarize_oof(root, args.dense_rater_min_annotations)
    print(f"Finished. Summary: {root / 'summary.csv'} ({len(summary)} rows)")
    print(
        f"Paired OOF summary: {root / 'paired_oof_summary.csv'} "
        f"({len(paired)} rows)"
    )


if __name__ == "__main__":
    main()
