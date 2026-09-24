#!/usr/bin/env python3
"""Run BBQ Experiment 1 across original and newly prepared datasets.

Each bootstrap iteration samples raters with replacement and keeps all of each
selected rater's comparisons. A deterministic seed manifest is shared by every
method so method comparisons are paired. Results are appended incrementally and
can be resumed after interruption.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from multiprocessing import Pool
from pathlib import Path
import sys
import traceback
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Prevent NumPy/SciPy from creating a thread pool inside every worker process.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bootstrap import build_bootstrap_sample  # noqa: E402
from elo_processor import (  # noqa: E402
    DataProcessor,
    EloProcessor,
    GPU_BAYESIAN_MODELS,
)
from models import BETA_PRIOR_ALPHA, BETA_PRIOR_BETA  # noqa: E402
from torch_bayesian_backend import (  # noqa: E402
    cuda_metadata,
    resolve_device,
)


DEFAULT_CROWD_BT_REPETITIONS = 1000
DEFAULT_RESULTS_ROOT = (
    PROJECT_ROOT / "experiment_results" / "experiment_1_cross_dataset_stability"
)

HUMAINE_NEW_METHODS = [
    "m_elo",
    "am_elo",
    "correctness_downweight",
    "correctness_reverse",
    "adaptive_clip",
    "adaptive_flip",
    "correctness_downweight_shared",
    "correctness_reverse_shared",
    "adaptive_clip_shared",
    "adaptive_flip_shared",
]

PRESET_HUMAINE_OFFICIAL = "humaine_official_new_methods"
PRESET_IHQ_ALL_REFERENCE = "ihq_vs_all_full_fit"
PRESET_LABEL_SMOOTHING_TABLE1 = "label_smoothing_table1"

LABEL_SMOOTHING_TABLE1_DATASETS = [
    "humaine",
    "conha",
    "ihq_all",
    "ihq_screened",
    "ihq_unscreened",
    "chatbot_arena_33k",
    "computer_agent_arena",
    "search_arena",
    "vision_arena",
]


DATASETS: Dict[str, Dict[str, Any]] = {
    # BBQ paper datasets.
    "humaine": {
        "label": "HUMAINE",
        "project": "prolific",
        "csv": "feedback_comparisons.csv",
        "group": "original",
        "paper_repetitions": 1000,
        "external_reference": {
            "csv": "references/humaine_official_ground_truth_2025-09-03.csv",
            "method_column": "model_name",
            "score_column": "score",
            "label": "HUMAINE official leaderboard (2025-09-03 snapshot)",
        },
    },
    "mt_bench": {
        "label": "MT-Bench",
        "project": "mtbench",
        "csv": "human_judgments.csv",
        "group": "original",
        "paper_repetitions": 10000,
    },
    "wd": {
        "label": "WD",
        "project": "WD",
        "csv": "answers.csv",
        "group": "original",
        "paper_repetitions": 10000,
    },
    "hific": {
        "label": "HiFiC",
        "project": "hific",
        "csv": "userstudy_google_elo.csv",
        "group": "original",
        "paper_repetitions": 10000,
    },
    "conha": {
        "label": "ConHa",
        "project": "conha",
        "csv": "answers.csv",
        "group": "original",
        "paper_repetitions": 10000,
    },
    "ihq_all": {
        "label": "IHQ-all",
        "project": "clic2024",
        "csv": "2AFC_google_elo.csv",
        "group": "original",
        "paper_repetitions": 10000,
    },
    "ihq_screened": {
        "label": "IHQ-screened",
        "project": "clic2024",
        "csv": "2AFC_filtered_google_elo.csv",
        "group": "original",
        "paper_repetitions": 10000,
    },
    "ihq_unscreened": {
        "label": "IHQ-unscreened",
        "project": "clic2024",
        "csv": "2AFC_unfiltered_google_elo.csv",
        "group": "original",
        "paper_repetitions": 10000,
    },
    # Additional datasets prepared in this repository.
    "chatbot_arena_33k": {
        "label": "Chatbot Arena 33K",
        "project": "chatbot_arena_33k",
        "csv": "preferences.csv",
        "group": "new",
        "paper_repetitions": 10000,
    },
    "computer_agent_arena": {
        "label": "Computer Agent Arena",
        "project": "computer_agent_arena",
        "csv": "preferences.csv",
        "group": "new",
        "paper_repetitions": 10000,
    },
    "llm_judge_holdout": {
        "label": "LLM Judge Holdout",
        "project": "llm_judge_holdout",
        "csv": "preferences.csv",
        "group": "new",
        "paper_repetitions": 10000,
    },
    "multipref_normal": {
        "label": "MultiPref-normal",
        "project": "multipref",
        "csv": "multipref_normal.csv",
        "group": "new",
        "paper_repetitions": 10000,
    },
    "multipref_expert": {
        "label": "MultiPref-expert",
        "project": "multipref",
        "csv": "multipref_expert.csv",
        "group": "new",
        "paper_repetitions": 10000,
    },
    "multipref_all": {
        "label": "MultiPref-all",
        "project": "multipref",
        "csv": "multipref_all.csv",
        "group": "new",
        "paper_repetitions": 10000,
    },
    "search_arena": {
        "label": "Search Arena v1-7k",
        "project": "search_arena",
        "csv": "preferences.csv",
        "group": "new",
        "paper_repetitions": 10000,
    },
    "vision_arena": {
        "label": "VisionArena-Battle",
        "project": "vision_arena",
        "csv": "preferences.csv",
        "group": "new",
        "paper_repetitions": 10000,
    },
}


METHODS: Dict[str, Dict[str, Any]] = {
    "crowd_bt": {
        "label": "Crowd-BT",
        "model": "google_elo",
        "shared_rater": False,
        "group": "baseline",
    },
    "bayes_bt": {
        "label": "Bayes-BT",
        "model": "bayesian_elo",
        "shared_rater": False,
        "group": "baseline",
    },
    "bbq": {
        "label": "BBQ",
        "model": "bayesian_elo_noise",
        "shared_rater": False,
        "group": "baseline",
    },
    "m_elo": {
        "label": "m-ELO",
        "model": "m_elo",
        "shared_rater": False,
        "group": "baseline",
    },
    "am_elo": {
        "label": "am-ELO",
        "model": "am_elo",
        "shared_rater": False,
        "group": "baseline",
    },
    "ls_bt": {
        "label": "LS-BT (epsilon=0.10)",
        "model": "label_smoothed_bt",
        "shared_rater": False,
        "group": "baseline",
        "label_smoothing": 0.10,
    },
    "correctness_downweight": {
        "label": "Correctness Downweight",
        "model": "bayesian_elo_correctness",
        "shared_rater": False,
        "group": "ours",
    },
    "correctness_reverse": {
        "label": "Correctness Reverse",
        "model": "bayesian_elo_correctness_reverse",
        "shared_rater": False,
        "group": "ours",
    },
    "adaptive_clip": {
        "label": "Adaptive Clip",
        "model": "bayesian_elo_adaptive_clip",
        "shared_rater": False,
        "group": "ours",
    },
    "adaptive_flip": {
        "label": "Adaptive Flip",
        "model": "bayesian_elo_adaptive_flip",
        "shared_rater": False,
        "group": "ours",
    },
    "correctness_downweight_shared": {
        "label": "Correctness Downweight (shared rater)",
        "model": "bayesian_elo_correctness",
        "shared_rater": True,
        "group": "shared",
    },
    "correctness_reverse_shared": {
        "label": "Correctness Reverse (shared rater)",
        "model": "bayesian_elo_correctness_reverse",
        "shared_rater": True,
        "group": "shared",
    },
    "adaptive_clip_shared": {
        "label": "Adaptive Clip (shared rater)",
        "model": "bayesian_elo_adaptive_clip",
        "shared_rater": True,
        "group": "shared",
    },
    "adaptive_flip_shared": {
        "label": "Adaptive Flip (shared rater)",
        "model": "bayesian_elo_adaptive_flip",
        "shared_rater": True,
        "group": "shared",
    },
}


RUN_FIELDS = [
    "dataset_id",
    "dataset",
    "method_id",
    "method",
    "model",
    "shared_rater",
    "iteration",
    "seed",
    "status",
    "fit_status",
    "backend",
    "error_type",
    "error_message",
    "top1_agreement",
    "spearman",
    "kendall_tau",
    "time_seconds",
    "method_coverage",
    "ranked_methods",
    "reference_methods",
    "sample_rows",
    "sampled_raters",
    "ranking_json",
]


_WORKER_CONTEXT: Dict[str, Any] = {}


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=True, indent=2, sort_keys=True)


def _resolve_selection(
    tokens: Sequence[str], definitions: Dict[str, Dict[str, Any]], groups: Iterable[str]
) -> List[str]:
    group_set = set(groups)
    if not tokens or "all" in tokens:
        return list(definitions)

    selected: List[str] = []
    for token in tokens:
        if token in group_set:
            selected.extend(
                key for key, value in definitions.items() if value["group"] == token
            )
        elif token in definitions:
            selected.append(token)
        else:
            valid = sorted(set(definitions) | group_set | {"all"})
            raise ValueError(f"Unknown selection {token!r}. Valid values: {valid}")
    return list(dict.fromkeys(selected))


def _dataset_path(config: Dict[str, Any]) -> Path:
    return PROJECT_ROOT / "projects" / config["project"] / "data" / config["csv"]


def _data_signature(
    path: Path, df: pd.DataFrame, valid_users: Sequence[Any]
) -> str:
    stat = path.stat()
    payload = {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "rows": len(df),
        "raters": len(valid_users),
        "rater_hash": hashlib.sha256(
            "\n".join(sorted(map(str, valid_users))).encode("utf-8")
        ).hexdigest(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _ensure_manifest(
    root: Path, dataset_id: str, repetitions: int, base_seed: int
) -> pd.DataFrame:
    manifest_path = root / "manifests" / f"{dataset_id}.csv"
    expected = pd.DataFrame(
        {
            "iteration": np.arange(repetitions, dtype=np.int64),
            "seed": base_seed + np.arange(repetitions, dtype=np.int64),
        }
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        existing = pd.read_csv(manifest_path)
        overlap = min(len(existing), len(expected))
        if overlap and not existing.iloc[:overlap].reset_index(drop=True).equals(
            expected.iloc[:overlap].reset_index(drop=True)
        ):
            raise ValueError(
                f"Manifest {manifest_path} conflicts with seed {base_seed}. "
                "Use a different results root for a different seed."
            )
        if len(existing) >= repetitions:
            return existing.iloc[:repetitions].copy()
    expected.to_csv(manifest_path, index=False)
    return expected


def _optimization_status(processor: EloProcessor) -> str:
    if processor.metric is None:
        return "completed"
    state = processor.metric.state
    for key in (
        "m_elo_optimization",
        "am_elo_optimization",
        "ls_bt_optimization",
    ):
        optimization = state.get(key)
        if optimization is not None:
            return "converged" if optimization.get("success") else "optimizer_warning"
    if "converged" in state:
        return "converged" if state["converged"] else "iteration_limit"
    return "completed"


def _fit_ranking(
    df: pd.DataFrame,
    valid_users: Sequence[Any],
    method_config: Dict[str, Any],
    device: str,
) -> Tuple[pd.DataFrame, float, str, str]:
    processor = EloProcessor(df, list(valid_users))
    ranking, elapsed = processor.process(
        df=df,
        valid_users=list(valid_users),
        model=method_config["model"],
        shared_rater=method_config["shared_rater"],
        device=device,
        label_smoothing=method_config.get("label_smoothing", 0.10),
    )
    if ranking is None or ranking.empty:
        raise RuntimeError("The method returned an empty ranking")
    required = {"Method", "ELO Score"}
    if not required.issubset(ranking.columns):
        raise RuntimeError(f"Ranking is missing columns {sorted(required - set(ranking))}")
    ranking = ranking.sort_values("ELO Score", ascending=False).reset_index(drop=True)
    backend = (
        processor.metric.state.get("computation_backend", "cpu")
        if processor.metric is not None
        else "cpu"
    )
    return ranking, float(elapsed), _optimization_status(processor), backend


def _canonical_method_name(value: Any) -> str:
    """Match repository-qualified and bare model identifiers consistently."""
    return str(value).strip().rsplit("/", 1)[-1]


def _ranking_metrics(
    ranking: pd.DataFrame, reference: pd.DataFrame
) -> Dict[str, float]:
    left = ranking.assign(
        _name=ranking["Method"].map(_canonical_method_name)
    ).set_index("_name")
    right = reference.assign(
        _name=reference["Method"].map(_canonical_method_name)
    ).set_index("_name")
    if left.index.has_duplicates or right.index.has_duplicates:
        raise ValueError("Canonical model names are not unique")

    ranked_names = set(left.index)
    reference_names = set(right.index)
    common = ranked_names & reference_names
    coverage = len(common) / len(reference_names) if reference_names else math.nan

    metrics = {
        "top1_agreement": float(
            _canonical_method_name(ranking.iloc[0]["Method"])
            == _canonical_method_name(reference.iloc[0]["Method"])
        ),
        "spearman": math.nan,
        "kendall_tau": math.nan,
        "method_coverage": coverage,
        "ranked_methods": len(ranked_names),
        "reference_methods": len(reference_names),
    }
    # Correlations over only the overlapping subset can look artificially
    # perfect when a bootstrap sample omits models. Ranking recovery is only
    # defined here when the complete reference model set is present.
    if common != reference_names or len(common) < 2:
        return metrics

    names = sorted(reference_names)
    left_scores = left.loc[names, "ELO Score"].astype(float)
    right_scores = right.loc[names, "ELO Score"].astype(float)
    metrics["spearman"] = float(
        spearmanr(left_scores, right_scores).correlation
    )
    metrics["kendall_tau"] = float(
        kendalltau(left_scores, right_scores).correlation
    )
    return metrics


def _load_external_reference(
    dataset_config: Dict[str, Any],
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    config = dataset_config.get("external_reference")
    if not config:
        return None, {}

    path = PROJECT_ROOT / config["csv"]
    if not path.exists():
        raise FileNotFoundError(f"External reference does not exist: {path}")
    source = pd.read_csv(path)
    method_column = config["method_column"]
    score_column = config["score_column"]
    missing = {method_column, score_column} - set(source.columns)
    if missing:
        raise ValueError(
            f"External reference {path} is missing columns {sorted(missing)}"
        )

    reference = source[[method_column, score_column]].rename(
        columns={method_column: "Method", score_column: "ELO Score"}
    )
    reference["ELO Score"] = pd.to_numeric(
        reference["ELO Score"], errors="coerce"
    )
    reference = reference.dropna(subset=["Method", "ELO Score"])
    reference = reference.sort_values(
        "ELO Score", ascending=False
    ).reset_index(drop=True)
    canonical_names = reference["Method"].map(_canonical_method_name)
    if canonical_names.duplicated().any():
        duplicates = sorted(canonical_names[canonical_names.duplicated()].unique())
        raise ValueError(
            f"External reference has duplicate canonical model names: {duplicates}"
        )
    if len(reference) < 2:
        raise ValueError(f"External reference has fewer than two models: {path}")

    metadata = {
        "evaluation_reference_type": "external",
        "evaluation_reference_label": config.get("label", path.stem),
        "evaluation_reference_path": str(path.relative_to(PROJECT_ROOT)),
        "evaluation_reference_models": len(reference),
        "evaluation_reference_top1": str(reference.iloc[0]["Method"]),
        "evaluation_reference_status": "valid",
    }
    return reference, metadata

def _worker_init(
    df: pd.DataFrame,
    valid_users: Sequence[Any],
    n_users: Optional[int],
    n_comp_per_user: Optional[int],
    dataset_id: str,
    dataset_label: str,
    method_id: str,
    method_config: Dict[str, Any],
    reference: pd.DataFrame,
    device: str,
    save_run_rankings: bool,
) -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = {
        "df": df,
        "valid_users": list(valid_users),
        "n_users": n_users,
        "n_comp_per_user": n_comp_per_user,
        "dataset_id": dataset_id,
        "dataset_label": dataset_label,
        "method_id": method_id,
        "method_config": method_config,
        "reference": reference,
        "device": device,
        "save_run_rankings": save_run_rankings,
    }


def _base_run_record(iteration: int, seed: int) -> Dict[str, Any]:
    context = _WORKER_CONTEXT
    method_config = context["method_config"]
    return {
        "dataset_id": context["dataset_id"],
        "dataset": context["dataset_label"],
        "method_id": context["method_id"],
        "method": method_config["label"],
        "model": method_config["model"],
        "shared_rater": method_config["shared_rater"],
        "iteration": iteration,
        "seed": seed,
        "status": "failed",
        "fit_status": "not_run",
        "backend": "",
        "error_type": "",
        "error_message": "",
        "top1_agreement": math.nan,
        "spearman": math.nan,
        "kendall_tau": math.nan,
        "time_seconds": math.nan,
        "method_coverage": math.nan,
        "ranked_methods": 0,
        "reference_methods": len(context["reference"]),
        "sample_rows": 0,
        "sampled_raters": 0,
        "ranking_json": "",
    }


def _run_worker(task: Tuple[int, int]) -> Dict[str, Any]:
    iteration, seed = task
    record = _base_run_record(iteration, seed)
    context = _WORKER_CONTEXT
    try:
        sample, sample_raters = build_bootstrap_sample(
            df=context["df"],
            valid_users=context["valid_users"],
            n_users=context["n_users"],
            n_comp_per_user=context["n_comp_per_user"],
            seed=seed,
        )
        ranking, elapsed, fit_status, backend = _fit_ranking(
            sample,
            sample_raters,
            context["method_config"],
            context["device"],
        )
        record.update(_ranking_metrics(ranking, context["reference"]))
        if context["save_run_rankings"]:
            record["ranking_json"] = json.dumps(
                [
                    {
                        "method": str(row["Method"]),
                        "elo_score": float(row["ELO Score"]),
                    }
                    for _, row in ranking.iterrows()
                ],
                ensure_ascii=True,
                separators=(",", ":"),
            )
        record.update(
            {
                "status": "success",
                "fit_status": fit_status,
                "backend": backend,
                "time_seconds": elapsed,
                "sample_rows": len(sample),
                "sampled_raters": len(sample_raters),
            }
        )
    except Exception as exc:  # A failed fit must be recorded, not discarded.
        record["error_type"] = type(exc).__name__
        record["error_message"] = " | ".join(
            line.strip() for line in traceback.format_exc(limit=4).splitlines()
        )
    return record


def _load_last_runs(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=RUN_FIELDS)
    runs = pd.read_csv(path)
    if "iteration" not in runs.columns:
        raise ValueError(f"Invalid run file: {path}")
    for field in RUN_FIELDS:
        if field not in runs.columns:
            runs[field] = ""
    runs = runs[RUN_FIELDS]
    runs["iteration"] = pd.to_numeric(runs["iteration"], errors="coerce")
    runs = runs.dropna(subset=["iteration"])
    runs["iteration"] = runs["iteration"].astype(int)
    return runs.drop_duplicates("iteration", keep="last")


def _fit_or_load_reference(
    output_dir: Path,
    df: pd.DataFrame,
    valid_users: Sequence[Any],
    method_config: Dict[str, Any],
    signature: str,
    recompute: bool,
    device: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    ranking_path = output_dir / "full_ranking.csv"
    metadata_path = output_dir / "metadata.json"
    if not recompute and ranking_path.exists() and metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        cached_smoothing = metadata.get("label_smoothing")
        requested_smoothing = method_config.get("label_smoothing")
        smoothing_matches = (
            cached_smoothing is None and requested_smoothing is None
        ) or (
            cached_smoothing is not None
            and requested_smoothing is not None
            and math.isclose(
                float(cached_smoothing),
                float(requested_smoothing),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        )
        if (
            metadata.get("data_signature") == signature
            and smoothing_matches
        ):
            return pd.read_csv(ranking_path), metadata

    ranking, elapsed, fit_status, backend = _fit_ranking(
        df, valid_users, method_config, device
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(ranking_path, index=False)
    metadata = {
        "data_signature": signature,
        "full_fit_time_seconds": elapsed,
        "full_fit_status": fit_status,
        "full_fit_backend": backend,
        "full_top1": str(ranking.iloc[0]["Method"]),
        "full_ranked_methods": len(ranking),
        "label_smoothing": method_config.get("label_smoothing"),
    }
    _json_dump(metadata_path, metadata)
    return ranking, metadata


def _load_or_fit_other_dataset_reference(
    reference_dataset_id: str,
    reference_dataset_config: Dict[str, Any],
    reference_df: pd.DataFrame,
    reference_users: Sequence[Any],
    reference_signature: str,
    method_id: str,
    method_config: Dict[str, Any],
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load a method's cached full fit on another dataset, or fit it once.

    Cached rankings from the original Experiment 1 result tree are read-only
    inputs. If no compatible cache exists, the new fit is written below the
    new protocol's result root, never into the original result tree.
    """
    source_dir = args.reference_results_root / reference_dataset_id / method_id
    source_ranking = source_dir / "full_ranking.csv"
    source_metadata = source_dir / "metadata.json"
    ranking: Optional[pd.DataFrame] = None
    fit_metadata: Dict[str, Any] = {}
    source_label = ""

    if source_ranking.exists() and source_metadata.exists():
        fit_metadata = json.loads(source_metadata.read_text(encoding="utf-8"))
        if fit_metadata.get("data_signature") == reference_signature:
            ranking = pd.read_csv(source_ranking)
            source_label = str(source_ranking)

    if ranking is None:
        cache_dir = (
            args.results_root
            / "_reference_fits"
            / reference_dataset_id
            / method_id
        )
        ranking, fit_metadata = _fit_or_load_reference(
            cache_dir,
            reference_df,
            reference_users,
            method_config,
            reference_signature,
            args.recompute_full,
            args.device,
        )
        source_label = str(cache_dir / "full_ranking.csv")

    required = {"Method", "ELO Score"}
    if not required.issubset(ranking.columns):
        raise ValueError(
            f"Reference ranking for {reference_dataset_id}/{method_id} "
            f"is missing columns {sorted(required - set(ranking.columns))}"
        )
    ranking = ranking.sort_values("ELO Score", ascending=False).reset_index(
        drop=True
    )
    metadata = {
        "evaluation_reference_type": "method_full_fit_other_dataset",
        "evaluation_reference_label": (
            f"{method_config['label']} full-data fit on "
            f"{reference_dataset_config['label']}"
        ),
        "evaluation_reference_dataset_id": reference_dataset_id,
        "evaluation_reference_dataset": reference_dataset_config["label"],
        "evaluation_reference_input_csv": str(
            _dataset_path(reference_dataset_config).relative_to(PROJECT_ROOT)
        ),
        "evaluation_reference_data_signature": reference_signature,
        "evaluation_reference_source_path": source_label,
        "evaluation_reference_models": len(ranking),
        "evaluation_reference_top1": str(ranking.iloc[0]["Method"]),
        "evaluation_reference_status": fit_metadata.get(
            "full_fit_status", ""
        ),
    }
    return ranking, metadata


def _append_results(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", newline="", encoding="utf-8") as handle:
            existing_header = next(csv.reader(handle), [])
        if existing_header != RUN_FIELDS:
            # Migrate resumable result files created before new metadata fields
            # were added, so appended rows never shift into the wrong columns.
            existing = _load_last_runs(path)
            existing.to_csv(path, index=False)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_FIELDS)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in RUN_FIELDS})
            handle.flush()


def _run_method(
    dataset_id: str,
    dataset_config: Dict[str, Any],
    method_id: str,
    method_config: Dict[str, Any],
    df: pd.DataFrame,
    valid_users: Sequence[Any],
    signature: str,
    manifest: pd.DataFrame,
    args: argparse.Namespace,
    evaluation_reference_override: Optional[pd.DataFrame] = None,
    evaluation_reference_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    output_dir = args.results_root / dataset_id / method_id
    full_ranking, reference_metadata = _fit_or_load_reference(
        output_dir,
        df,
        valid_users,
        method_config,
        signature,
        args.recompute_full,
        args.device,
    )
    evaluation_reference = (
        evaluation_reference_override
        if evaluation_reference_override is not None
        else full_ranking
    )
    evaluation_metadata = evaluation_reference_metadata or {
        "evaluation_reference_type": "method_full_fit",
        "evaluation_reference_label": f"{method_config['label']} full-data fit",
        "evaluation_reference_models": len(full_ranking),
        "evaluation_reference_top1": str(full_ranking.iloc[0]["Method"]),
        "evaluation_reference_status": reference_metadata.get(
            "full_fit_status", ""
        ),
    }
    if evaluation_reference_override is not None:
        evaluation_reference.to_csv(
            output_dir / "evaluation_reference.csv", index=False
        )
        evaluation_metadata = {
            **evaluation_metadata,
            "evaluation_reference_snapshot": "evaluation_reference.csv",
        }
    metadata = {
        **reference_metadata,
        **evaluation_metadata,
        "dataset_id": dataset_id,
        "dataset": dataset_config["label"],
        "input_csv": str(_dataset_path(dataset_config).relative_to(PROJECT_ROOT)),
        "input_rows": len(df),
        "input_raters": len(valid_users),
        "method_id": method_id,
        "method": method_config["label"],
        "model": method_config["model"],
        "shared_rater": method_config["shared_rater"],
        "label_smoothing": method_config.get("label_smoothing"),
        "beta_prior_alpha": BETA_PRIOR_ALPHA,
        "beta_prior_beta": BETA_PRIOR_BETA,
        "requested_iterations": len(manifest),
        "n_users": args.n_users,
        "n_comp_per_user": args.n_comp_per_user,
        "base_seed": args.seed,
        "requested_device": args.device,
        "save_run_rankings": args.save_run_rankings,
        "experiment_protocol": args.experiment_protocol,
    }
    _json_dump(output_dir / "metadata.json", metadata)

    run_path = output_dir / "runs.csv"
    previous = _load_last_runs(run_path)
    if args.retry_failed:
        reusable = previous[
            previous["status"].eq("success")
            & ~previous["fit_status"].isin({"iteration_limit", "optimizer_warning"})
        ]
        finished = set(reusable["iteration"].astype(int))
    else:
        finished = set(previous["iteration"].astype(int))
    tasks = [
        (int(row.iteration), int(row.seed))
        for row in manifest.itertuples(index=False)
        if int(row.iteration) not in finished
    ]
    if not tasks:
        print(f"[resume] {dataset_id}/{method_id}: already complete")
        return

    description = f"{dataset_id}/{method_id}"
    initializer_args = (
        df,
        list(valid_users),
        args.n_users,
        args.n_comp_per_user,
        dataset_id,
        dataset_config["label"],
        method_id,
        method_config,
        evaluation_reference,
        args.device,
        args.save_run_rankings,
    )
    actual_device = (
        resolve_device(args.device)
        if method_config["model"] in GPU_BAYESIAN_MODELS
        else "cpu"
    )
    workers = 1 if actual_device.startswith("cuda") else args.workers
    if workers == 1:
        _worker_init(*initializer_args)
        iterator = map(_run_worker, tasks)
        _append_results(
            run_path, tqdm(iterator, total=len(tasks), desc=description)
        )
        return

    with Pool(
        processes=workers,
        initializer=_worker_init,
        initargs=initializer_args,
        maxtasksperchild=args.max_tasks_per_child,
    ) as pool:
        iterator = pool.imap_unordered(_run_worker, tasks, chunksize=1)
        _append_results(
            run_path, tqdm(iterator, total=len(tasks), desc=description)
        )


def _mean_std_se(values: pd.Series) -> Tuple[float, float, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return math.nan, math.nan, math.nan
    mean = float(clean.mean())
    std = float(clean.std(ddof=1)) if len(clean) > 1 else 0.0
    return mean, std, std / math.sqrt(len(clean))


def failure_aware_correlation(
    values: pd.Series,
    coverage: pd.Series,
    eligible: pd.Series,
) -> pd.Series:
    """Map correlation and coverage to [-1, 1], assigning failures -1."""
    correlations = pd.to_numeric(values, errors="coerce")
    coverage_values = pd.to_numeric(coverage, errors="coerce").clip(0.0, 1.0)
    scores = pd.Series(-1.0, index=values.index, dtype=float)
    usable = eligible & correlations.notna() & coverage_values.notna()
    scores.loc[usable] = (
        coverage_values.loc[usable]
        * (correlations.loc[usable].clip(-1.0, 1.0) + 1.0)
        - 1.0
    )
    return scores


def summarize_results(
    root: Path,
    iteration_limits: Optional[Dict[str, int]] = None,
    output_name: str = "summary.csv",
) -> pd.DataFrame:
    summaries: List[Dict[str, Any]] = []
    for run_path in sorted(root.glob("*/*/runs.csv")):
        runs = _load_last_runs(run_path)
        if runs.empty:
            continue
        metadata_path = run_path.parent / "metadata.json"
        metadata: Dict[str, Any] = {}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        requested_iterations = metadata.get("requested_iterations")
        dataset_id = str(runs.iloc[0]["dataset_id"])
        if iteration_limits and dataset_id in iteration_limits:
            configured_limit = int(iteration_limits[dataset_id])
            requested_iterations = (
                min(int(requested_iterations), configured_limit)
                if requested_iterations is not None
                else configured_limit
            )
        if requested_iterations is not None and "iteration" in runs.columns:
            requested_iterations = int(requested_iterations)
            iteration = pd.to_numeric(runs["iteration"], errors="coerce")
            runs = runs[
                iteration.ge(0) & iteration.lt(requested_iterations)
            ].copy()
        if runs.empty:
            continue
        successful = runs[runs["status"] == "success"].copy()
        valid_fits = successful[
            ~successful["fit_status"].isin({"iteration_limit", "optimizer_warning"})
        ].copy()
        terminal_coverage = pd.to_numeric(
            successful["method_coverage"], errors="coerce"
        )
        terminal_full = successful[terminal_coverage.ge(1.0 - 1e-12)].copy()
        coverage = pd.to_numeric(valid_fits["method_coverage"], errors="coerce")
        full_coverage = valid_fits[coverage.ge(1.0 - 1e-12)].copy()

        reference_fit_status = metadata.get(
            "evaluation_reference_status",
            metadata.get("full_fit_status", ""),
        )
        reference_valid = reference_fit_status not in {
            "iteration_limit",
            "optimizer_warning",
        }
        metric_runs = full_coverage if reference_valid else full_coverage.iloc[:0]
        top1_runs = valid_fits if reference_valid else valid_fits.iloc[:0]

        eligible = (
            runs["status"].eq("success")
            & ~runs["fit_status"].isin({"iteration_limit", "optimizer_warning"})
            & reference_valid
        )
        all_coverage = pd.to_numeric(runs["method_coverage"], errors="coerce")
        failure_top1_values = pd.to_numeric(
            runs["top1_agreement"], errors="coerce"
        ).where(eligible, 0.0).fillna(0.0)
        failure_spearman_values = failure_aware_correlation(
            runs["spearman"], all_coverage, eligible
        )
        failure_kendall_values = failure_aware_correlation(
            runs["kendall_tau"], all_coverage, eligible
        )

        terminal_top1 = _mean_std_se(successful["top1_agreement"])
        terminal_spearman = _mean_std_se(terminal_full["spearman"])
        terminal_kendall = _mean_std_se(terminal_full["kendall_tau"])
        top1 = _mean_std_se(top1_runs["top1_agreement"])
        spearman = _mean_std_se(metric_runs["spearman"])
        kendall = _mean_std_se(metric_runs["kendall_tau"])
        failure_top1 = _mean_std_se(failure_top1_values)
        failure_spearman = _mean_std_se(failure_spearman_values)
        failure_kendall = _mean_std_se(failure_kendall_values)
        times = pd.to_numeric(valid_fits["time_seconds"], errors="coerce").dropna()
        first = runs.iloc[0]
        summaries.append(
            {
                "experiment_protocol": metadata.get(
                    "experiment_protocol", "standard"
                ),
                "dataset_id": first["dataset_id"],
                "dataset": first["dataset"],
                "evaluation_reference_type": metadata.get(
                    "evaluation_reference_type", "method_full_fit"
                ),
                "evaluation_reference_label": metadata.get(
                    "evaluation_reference_label", ""
                ),
                "evaluation_reference_dataset_id": metadata.get(
                    "evaluation_reference_dataset_id", dataset_id
                ),
                "method_id": first["method_id"],
                "method": first["method"],
                "model": first["model"],
                "shared_rater": first["shared_rater"],
                "backend": first.get("backend", "cpu"),
                "recorded_runs": len(runs),
                "successful_runs": len(successful),
                "terminal_metric_runs": len(terminal_full),
                "valid_fit_runs": len(valid_fits),
                "valid_fit_rate": len(valid_fits) / len(runs),
                "valid_metric_runs": len(metric_runs),
                "valid_metric_rate": len(metric_runs) / len(runs),
                "failed_runs": int((runs["status"] != "success").sum()),
                "partial_coverage_runs": int((coverage < 1.0 - 1e-12).sum()),
                "reference_fit_status": reference_fit_status,
                "converged_runs": int(
                    (successful["fit_status"] == "converged").sum()
                ),
                "iteration_limit_runs": int(
                    (successful["fit_status"] == "iteration_limit").sum()
                ),
                "nonconvergence_rate": (
                    float(
                        (successful["fit_status"] == "iteration_limit").mean()
                    )
                    if len(successful)
                    else math.nan
                ),
                "optimizer_warnings": int(
                    (successful["fit_status"] == "optimizer_warning").sum()
                ),
                "terminal_top1_mean": terminal_top1[0],
                "terminal_top1_std": terminal_top1[1],
                "terminal_top1_se": terminal_top1[2],
                "terminal_spearman_mean": terminal_spearman[0],
                "terminal_spearman_std": terminal_spearman[1],
                "terminal_spearman_se": terminal_spearman[2],
                "terminal_kendall_mean": terminal_kendall[0],
                "terminal_kendall_std": terminal_kendall[1],
                "terminal_kendall_se": terminal_kendall[2],
                "top1_mean": top1[0],
                "top1_std": top1[1],
                "top1_se": top1[2],
                "spearman_mean": spearman[0],
                "spearman_std": spearman[1],
                "spearman_se": spearman[2],
                "kendall_mean": kendall[0],
                "kendall_std": kendall[1],
                "kendall_se": kendall[2],
                "failure_aware_top1_mean": failure_top1[0],
                "failure_aware_top1_std": failure_top1[1],
                "failure_aware_top1_se": failure_top1[2],
                "failure_aware_spearman_mean": failure_spearman[0],
                "failure_aware_spearman_std": failure_spearman[1],
                "failure_aware_spearman_se": failure_spearman[2],
                "failure_aware_kendall_mean": failure_kendall[0],
                "failure_aware_kendall_std": failure_kendall[1],
                "failure_aware_kendall_se": failure_kendall[2],
                "mean_method_coverage": pd.to_numeric(
                    valid_fits["method_coverage"], errors="coerce"
                ).mean(),
                "time_mean_seconds": times.mean() if len(times) else math.nan,
                "time_median_seconds": times.median() if len(times) else math.nan,
                "time_p95_seconds": times.quantile(0.95) if len(times) else math.nan,
            }
        )
    summary = pd.DataFrame(summaries)
    if not summary.empty:
        summary = summary.sort_values(["dataset_id", "method_id"])
    root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(root / output_name, index=False)
    return summary


def _apply_protocol_preset(args: argparse.Namespace) -> argparse.Namespace:
    """Resolve safe, non-overwriting presets for the requested reruns."""
    default_root = DEFAULT_RESULTS_ROOT.resolve()
    requested_root = args.results_root.resolve()
    args.experiment_protocol = args.preset or "standard"
    METHODS["ls_bt"]["label_smoothing"] = float(args.label_smoothing)
    METHODS["ls_bt"]["label"] = (
        f"LS-BT (epsilon={args.label_smoothing:.2f})"
    )

    if args.preset == PRESET_HUMAINE_OFFICIAL:
        if args.datasets == ["all"]:
            args.datasets = ["humaine"]
        if args.methods == ["all"]:
            args.methods = list(HUMAINE_NEW_METHODS)
        if args.bootstrap_n is None:
            args.bootstrap_n = 1000
        args.use_external_reference = True
        args.save_run_rankings = True
        if requested_root == default_root:
            args.results_root = (
                PROJECT_ROOT
                / "experiment_results"
                / "experiment_1_humaine_official_reference"
            )

    elif args.preset == PRESET_IHQ_ALL_REFERENCE:
        if args.datasets == ["all"]:
            args.datasets = ["ihq_screened", "ihq_unscreened"]
        if args.bootstrap_n is None:
            args.bootstrap_n = 10000
        args.crowd_bt_repetitions = args.bootstrap_n
        args.method_full_reference_dataset = "ihq_all"
        args.save_run_rankings = True
        if requested_root == default_root:
            args.results_root = (
                PROJECT_ROOT
                / "experiment_results"
                / "experiment_1_ihq_vs_all_full_fit"
            )

    elif args.preset == PRESET_LABEL_SMOOTHING_TABLE1:
        if args.datasets == ["all"]:
            args.datasets = list(LABEL_SMOOTHING_TABLE1_DATASETS)
        if args.methods == ["all"]:
            args.methods = ["ls_bt"]
        args.use_external_reference = True
        args.save_run_rankings = True
        if requested_root == default_root:
            epsilon_token = f"{args.label_smoothing:.2f}".replace(".", "_")
            args.results_root = (
                PROJECT_ROOT
                / "experiment_results"
                / f"experiment_1_label_smoothing_eps_{epsilon_token}"
            )

    args.results_root = args.results_root.resolve()
    args.reference_results_root = args.reference_results_root.resolve()
    return args


def _print_resolved_protocol(args: argparse.Namespace) -> None:
    dataset_ids = _resolve_selection(
        args.datasets, DATASETS, groups={"original", "new"}
    )
    method_ids = _resolve_selection(
        args.methods, METHODS, groups={"baseline", "ours", "shared"}
    )
    print(f"Protocol: {args.experiment_protocol}")
    print(f"Datasets: {' '.join(dataset_ids)}")
    print(f"Methods: {' '.join(method_ids)}")
    print(f"Bootstrap repetitions: {args.bootstrap_n or 'dataset default'}")
    print(f"Results root: {args.results_root}")
    print(f"External reference: {args.use_external_reference}")
    print(
        "Method full-fit reference dataset: "
        f"{args.method_full_reference_dataset or 'none'}"
    )
    if args.method_full_reference_dataset:
        print(f"Reference cache root: {args.reference_results_root}")
    print(f"Save per-run rankings: {args.save_run_rankings}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=[
            PRESET_HUMAINE_OFFICIAL,
            PRESET_IHQ_ALL_REFERENCE,
            PRESET_LABEL_SMOOTHING_TABLE1,
        ],
        default=None,
        help=(
            "Safe protocol preset. Each preset writes to a dedicated result "
            "tree unless --results-root is explicitly changed."
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        help="Dataset IDs, or one of: original, new, all.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["all"],
        help="Method IDs, or one of: baseline, ours, shared, all.",
    )
    parser.add_argument(
        "--bootstrap-n",
        type=int,
        default=None,
        help="Override repetitions for every selected dataset. Default uses paper counts.",
    )
    parser.add_argument(
        "--crowd-bt-repetitions",
        type=int,
        default=DEFAULT_CROWD_BT_REPETITIONS,
        help=(
            "Maximum repetitions for Crowd-BT only. Other methods use "
            "--bootstrap-n or the dataset paper count."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.10,
        help=(
            "Symmetric LS-BT smoothing/flip probability in [0, 0.5). "
            "The Table-1 baseline uses 0.10."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Bayesian backend: auto, cpu, cuda, or cuda:N.",
    )
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--max-tasks-per-child", type=int, default=50)
    parser.add_argument("--n-users", type=int, default=None)
    parser.add_argument("--n-comp-per-user", type=int, default=None)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
    )
    parser.add_argument(
        "--method-full-reference-dataset",
        choices=sorted(DATASETS),
        default=None,
        help=(
            "Evaluate every selected dataset against the selected method's "
            "full-data fit on this other dataset."
        ),
    )
    parser.add_argument(
        "--reference-results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help=(
            "Read compatible cached full rankings from this result tree. "
            "Missing references are fitted under the new --results-root."
        ),
    )
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--recompute-full", action="store_true")
    parser.add_argument(
        "--use-external-reference",
        action="store_true",
        help=(
            "Use a dataset's configured external leaderboard as the Bootstrap "
            "evaluation target. Datasets without one keep the method full-data fit."
        ),
    )
    parser.add_argument(
        "--save-run-rankings",
        action="store_true",
        help="Store each Bootstrap ranking as JSON in runs.csv.",
    )
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved protocol without fitting any model.",
    )
    parser.add_argument(
        "--summary-paper-counts",
        action="store_true",
        help=(
            "When summarizing, cap each dataset at its configured paper "
            "repetition count and write summary_paper_protocol.csv."
        ),
    )
    parser.add_argument("--list", action="store_true", dest="list_options")
    return parser.parse_args()


def main() -> None:
    args = _apply_protocol_preset(_parse_args())
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    if args.bootstrap_n is not None and args.bootstrap_n <= 0:
        raise ValueError("bootstrap-n must be positive")
    if args.crowd_bt_repetitions <= 0:
        raise ValueError("crowd-bt-repetitions must be positive")
    if not 0.0 <= args.label_smoothing < 0.5:
        raise ValueError("label-smoothing must be in [0, 0.5)")
    if args.use_external_reference and args.method_full_reference_dataset:
        raise ValueError(
            "Choose either --use-external-reference or "
            "--method-full-reference-dataset, not both"
        )
    if (
        args.method_full_reference_dataset
        and args.results_root == DEFAULT_RESULTS_ROOT.resolve()
    ):
        raise ValueError(
            "A cross-dataset full-fit reference must use a separate "
            "--results-root so the original Experiment 1 results are not "
            "overwritten. Use --preset ihq_vs_all_full_fit for the safe path."
        )
    if args.list_options:
        print("Datasets:")
        for key, value in DATASETS.items():
            print(f"  {key:24s} {value['label']} [{value['group']}]")
        print("Methods:")
        for key, value in METHODS.items():
            print(f"  {key:34s} {value['label']} [{value['group']}]")
        return

    if args.dry_run:
        _print_resolved_protocol(args)
        return

    if args.summary_only:
        iteration_limits = None
        output_name = "summary.csv"
        if args.summary_paper_counts:
            iteration_limits = {
                dataset_id: int(config["paper_repetitions"])
                for dataset_id, config in DATASETS.items()
            }
            output_name = "summary_paper_protocol.csv"
        summary = summarize_results(
            args.results_root,
            iteration_limits=iteration_limits,
            output_name=output_name,
        )
        print(f"Wrote {len(summary)} rows to {args.results_root / output_name}")
        return

    # Fail early for an invalid explicit CUDA request. Auto may fall back to CPU.
    device_metadata = cuda_metadata(args.device)

    dataset_ids = _resolve_selection(
        args.datasets, DATASETS, groups={"original", "new"}
    )
    method_ids = _resolve_selection(
        args.methods, METHODS, groups={"baseline", "ours", "shared"}
    )

    method_reference_context: Optional[Dict[str, Any]] = None
    if args.method_full_reference_dataset:
        reference_dataset_id = args.method_full_reference_dataset
        reference_dataset_config = DATASETS[reference_dataset_id]
        reference_path = _dataset_path(reference_dataset_config)
        reference_processor = DataProcessor(str(reference_path))
        if not reference_processor.load_data() or reference_processor.df is None:
            raise RuntimeError(
                f"Could not load reference dataset {reference_dataset_id}: "
                f"{reference_path}"
            )
        reference_df = reference_processor.df.reset_index(drop=True)
        reference_users = reference_processor.get_valid_users()
        if len(reference_users) < 2:
            raise ValueError(
                f"Reference dataset {reference_dataset_id} has fewer than two raters"
            )
        reference_signature = _data_signature(
            reference_path, reference_df, reference_users
        )
        method_reference_context = {
            "dataset_id": reference_dataset_id,
            "config": reference_dataset_config,
            "path": reference_path,
            "df": reference_df,
            "valid_users": reference_users,
            "signature": reference_signature,
            "models": {
                _canonical_method_name(value)
                for value in pd.concat(
                    [reference_df["methodA"], reference_df["methodB"]]
                ).dropna()
            },
        }

    args.results_root.mkdir(parents=True, exist_ok=True)
    _json_dump(
        args.results_root / "experiment_config.json",
        {
            "experiment_protocol": args.experiment_protocol,
            "datasets": dataset_ids,
            "methods": method_ids,
            "bootstrap_n_override": args.bootstrap_n,
            "method_repetition_caps": {
                "crowd_bt": args.crowd_bt_repetitions,
            },
            "seed": args.seed,
            "workers": args.workers,
            "requested_device": args.device,
            "label_smoothing": args.label_smoothing,
            "resolved_bayesian_backend": device_metadata,
            "n_users": args.n_users,
            "n_comp_per_user": args.n_comp_per_user,
            "use_external_reference": args.use_external_reference,
            "method_full_reference_dataset": (
                args.method_full_reference_dataset
            ),
            "reference_results_root": str(args.reference_results_root),
            "save_run_rankings": args.save_run_rankings,
            "beta_prior_alpha": BETA_PRIOR_ALPHA,
            "beta_prior_beta": BETA_PRIOR_BETA,
            "reference_only_files": [
                "projects/llm_judge_holdout/data/human_ground_truth.csv",
                "projects/multipref/data/multipref_expert_consensus.csv",
            ],
        },
    )

    for dataset_id in dataset_ids:
        dataset_config = DATASETS[dataset_id]
        path = _dataset_path(dataset_config)
        processor = DataProcessor(str(path))
        if not processor.load_data() or processor.df is None:
            print(f"[failed] Could not load {dataset_id}: {path}")
            continue
        df = processor.df.reset_index(drop=True)
        valid_users = processor.get_valid_users()
        if len(valid_users) < 2:
            print(f"[skip] {dataset_id} has fewer than two raters")
            continue
        signature = _data_signature(path, df, valid_users)
        data_models = {
            _canonical_method_name(value)
            for value in pd.concat([df["methodA"], df["methodB"]]).dropna()
        }
        if (
            method_reference_context is not None
            and data_models != method_reference_context["models"]
        ):
            missing = sorted(data_models - method_reference_context["models"])
            extra = sorted(method_reference_context["models"] - data_models)
            raise ValueError(
                f"Dataset {dataset_id} and reference dataset "
                f"{method_reference_context['dataset_id']} have different model "
                f"sets. Missing from reference: {missing}; extra in reference: {extra}"
            )
        external_reference = None
        external_reference_metadata: Dict[str, Any] = {}
        if args.use_external_reference:
            external_reference, external_reference_metadata = (
                _load_external_reference(dataset_config)
            )
            if external_reference is not None:
                reference_models = {
                    _canonical_method_name(value)
                    for value in external_reference["Method"]
                }
                if data_models != reference_models:
                    missing = sorted(data_models - reference_models)
                    extra = sorted(reference_models - data_models)
                    raise ValueError(
                        "External reference model set does not match the data. "
                        f"Missing from reference: {missing}; extra in reference: {extra}"
                    )
                reference_dir = args.results_root / dataset_id / "reference"
                reference_dir.mkdir(parents=True, exist_ok=True)
                external_reference.to_csv(
                    reference_dir / "evaluation_reference.csv", index=False
                )
                _json_dump(
                    reference_dir / "metadata.json",
                    external_reference_metadata,
                )
        repetitions = (
            args.bootstrap_n
            if args.bootstrap_n is not None
            else dataset_config["paper_repetitions"]
        )
        manifest = _ensure_manifest(
            args.results_root, dataset_id, repetitions, args.seed
        )
        _json_dump(
            args.results_root / dataset_id / "dataset_metadata.json",
            {
                "dataset_id": dataset_id,
                "dataset": dataset_config["label"],
                "group": dataset_config["group"],
                "input_csv": str(path.relative_to(PROJECT_ROOT)),
                "rows": len(df),
                "raters": len(valid_users),
                "models": int(
                    pd.concat([df["methodA"], df["methodB"]]).dropna().nunique()
                ),
                "repetitions": repetitions,
                "data_signature": signature,
                "experiment_protocol": args.experiment_protocol,
                "evaluation_reference_dataset_id": (
                    args.method_full_reference_dataset
                ),
                **external_reference_metadata,
            },
        )
        print(
            f"\nDataset {dataset_config['label']}: {len(df)} rows, "
            f"{len(valid_users)} raters, {repetitions} repetitions"
        )
        for method_id in method_ids:
            method_manifest = manifest
            if method_id == "crowd_bt":
                method_manifest = manifest.iloc[
                    : min(len(manifest), args.crowd_bt_repetitions)
                ].copy()
            try:
                evaluation_reference = external_reference
                evaluation_reference_metadata = external_reference_metadata
                if method_reference_context is not None:
                    (
                        evaluation_reference,
                        evaluation_reference_metadata,
                    ) = _load_or_fit_other_dataset_reference(
                        method_reference_context["dataset_id"],
                        method_reference_context["config"],
                        method_reference_context["df"],
                        method_reference_context["valid_users"],
                        method_reference_context["signature"],
                        method_id,
                        METHODS[method_id],
                        args,
                    )
                _run_method(
                    dataset_id,
                    dataset_config,
                    method_id,
                    METHODS[method_id],
                    df,
                    valid_users,
                    signature,
                    method_manifest,
                    args,
                    evaluation_reference_override=evaluation_reference,
                    evaluation_reference_metadata=(
                        evaluation_reference_metadata
                    ),
                )
            except Exception as exc:
                error_path = args.results_root / dataset_id / method_id / "setup_error.txt"
                error_path.parent.mkdir(parents=True, exist_ok=True)
                error_path.write_text(traceback.format_exc(), encoding="utf-8")
                print(f"[failed] {dataset_id}/{method_id}: {exc}")

        summary = summarize_results(args.results_root)
        print(f"Updated {args.results_root / 'summary.csv'} ({len(summary)} rows)")


if __name__ == "__main__":
    main()
