#!/usr/bin/env python3
"""Run BBQ's data-scale experiment with resumable, paired bootstraps.

For each dataset this script varies one dimension at a time:

* ``n_raters``: sample R raters and retain all of their comparisons.
* ``comparisons_per_rater``: sample the full number of raters and retain at
  most K comparisons from each sampled rater.

Every method uses the same deterministic seed manifest for a given setting.
The main user-facing output is one long-form ``summary.csv`` table.
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
import time
import traceback
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Keep each fit single-threaded; parallelism is controlled by --workers.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr
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
from run_cross_dataset_stability import (  # noqa: E402
    METHODS,
    failure_aware_correlation,
)
from torch_bayesian_backend import cuda_metadata, resolve_device  # noqa: E402


PAPER_METHODS = ["crowd_bt", "bayes_bt", "bbq"]
AXES = ("n_raters", "comparisons_per_rater")
MIN_VALID_METRIC_RATE = 0.95
DEFAULT_CROWD_BT_REPETITIONS = 1
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiment_2_datasets.json"
DEFAULT_RESULTS = (
    PROJECT_ROOT
    / "experiment_results"
    / "experiment_2_paper_replication"
)

RUN_FIELDS = [
    "dataset_id",
    "dataset",
    "group",
    "axis",
    "scale_value",
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
    "pearson",
    "spearman",
    "kendall_tau",
    "time_seconds",
    "method_coverage",
    "ranked_methods",
    "reference_methods",
    "sample_rows",
    "sampled_raters",
    "mean_comparisons_per_sampled_rater",
]


_WORKER_CONTEXT: Dict[str, Any] = {}


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=True, indent=2, sort_keys=True)


def _load_dataset_config(path: Path) -> Dict[str, Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict) or not config:
        raise ValueError(f"Dataset config must be a non-empty object: {path}")
    required = {
        "label",
        "group",
        "csv",
        "repetitions",
        "rater_grid",
        "comparison_grid",
    }
    for dataset_id, value in config.items():
        missing = required - set(value)
        if missing:
            raise ValueError(
                f"Dataset {dataset_id!r} is missing fields {sorted(missing)}"
            )
    return config


def _resolve_datasets(
    requested: Sequence[str], config: Dict[str, Dict[str, Any]]
) -> List[str]:
    if not requested or "default" in requested:
        selected = [key for key, value in config.items() if value.get("enabled")]
    elif "all" in requested:
        selected = list(config)
    else:
        unknown = sorted(set(requested) - set(config))
        if unknown:
            raise ValueError(
                f"Unknown datasets {unknown}; edit {DEFAULT_CONFIG.name} to add one"
            )
        selected = list(requested)
    if not selected:
        raise ValueError("No datasets selected")
    return list(dict.fromkeys(selected))


def _resolve_methods(requested: Sequence[str]) -> List[str]:
    groups = {
        "paper": PAPER_METHODS,
        "baseline": [k for k, v in METHODS.items() if v["group"] == "baseline"],
        "ours": [k for k, v in METHODS.items() if v["group"] == "ours"],
        "shared": [k for k, v in METHODS.items() if v["group"] == "shared"],
        "all": list(METHODS),
    }
    selected: List[str] = []
    for token in requested or ["paper"]:
        if token in groups:
            selected.extend(groups[token])
        elif token in METHODS:
            selected.append(token)
        else:
            raise ValueError(
                f"Unknown method {token!r}; valid groups are {sorted(groups)}"
            )
    return list(dict.fromkeys(selected))


def _load_data(csv_path: Path) -> Tuple[pd.DataFrame, List[Any]]:
    processor = DataProcessor(str(csv_path))
    if not processor.load_data() or processor.df is None:
        raise RuntimeError(f"Failed to load {csv_path}")
    # Keep Golden rows in the shared bootstrap frame. Google ELO/Crowd-BT
    # needs them for rater calibration; Bayes-BT and BBQ discard them in
    # their own CLIC conversion, as in the original implementation.
    df = processor.df.reset_index(drop=True)
    valid_users = processor.get_valid_users()
    if not valid_users:
        raise ValueError(f"Dataset has no valid raters: {csv_path}")
    return df, valid_users


def _resolve_grid(values: Sequence[Any], maximum: int) -> List[int]:
    resolved: List[int] = []
    for raw in values:
        value = maximum if str(raw).strip().lower() == "max" else int(raw)
        if value <= 0:
            raise ValueError("Scale values must be positive")
        value = min(value, maximum)
        if value not in resolved:
            resolved.append(value)
    return resolved


def _data_signature(path: Path, df: pd.DataFrame) -> str:
    stat = path.stat()
    payload = {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "rows": len(df),
        "golden_rows": int(
            (
                df["isGolden"].notna()
                & df["isGolden"].astype(str).str.lower().eq("true")
            ).sum()
        ) if "isGolden" in df.columns else 0,
        "golden_protocol": "retain_for_rater_aware_methods",
        "raters": int(df["answerer"].nunique()),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _ensure_dataset_identity(
    root: Path,
    dataset_id: str,
    identity: Dict[str, Any],
) -> None:
    """Prevent data changes from being mixed with existing fitted runs."""
    path = root / dataset_id / "protocol_identity.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing == identity:
            return
        has_runs = any((root / dataset_id).glob("**/runs.csv"))
        if has_runs:
            raise ValueError(
                f"Dataset protocol changed for {dataset_id}. Use a new "
                f"--results-root instead of mixing results in {root}."
            )
    _json_dump(path, identity)


def _ensure_manifest(
    root: Path,
    dataset_id: str,
    axis: str,
    scale_value: int,
    repetitions: int,
    base_seed: int,
) -> pd.DataFrame:
    path = root / "manifests" / dataset_id / f"{axis}_{scale_value}.csv"
    # The offset keeps settings independent while remaining deterministic.
    setting_hash = int(
        hashlib.sha256(f"{dataset_id}:{axis}:{scale_value}".encode()).hexdigest()[:8],
        16,
    )
    expected = pd.DataFrame(
        {
            "iteration": np.arange(repetitions, dtype=np.int64),
            "seed": base_seed + setting_hash + np.arange(repetitions, dtype=np.int64),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_csv(path)
        overlap = min(len(existing), repetitions)
        if overlap and not existing.iloc[:overlap].reset_index(drop=True).equals(
            expected.iloc[:overlap].reset_index(drop=True)
        ):
            raise ValueError(
                f"Manifest conflict at {path}; use a new results directory"
            )
        if len(existing) >= repetitions:
            return existing.iloc[:repetitions].copy()
    expected.to_csv(path, index=False)
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
    process_kwargs = {
        "df": df,
        "valid_users": list(valid_users),
        "model": method_config["model"],
        "shared_rater": method_config["shared_rater"],
        "device": device,
    }
    if "label_smoothing" in method_config:
        process_kwargs["label_smoothing"] = float(
            method_config["label_smoothing"]
        )
    ranking, elapsed = processor.process(**process_kwargs)
    if ranking is None or ranking.empty:
        raise RuntimeError("The method returned an empty ranking")
    required = {"Method", "ELO Score"}
    if not required.issubset(ranking.columns):
        raise RuntimeError(
            f"Ranking is missing {sorted(required - set(ranking.columns))}"
        )
    ranking = ranking.sort_values("ELO Score", ascending=False).reset_index(drop=True)
    backend = (
        processor.metric.state.get("computation_backend", "cpu")
        if processor.metric is not None
        else "cpu"
    )
    return ranking, float(elapsed), _optimization_status(processor), backend


def _ranking_metrics(
    ranking: pd.DataFrame, reference: pd.DataFrame
) -> Dict[str, float]:
    ranked = set(ranking["Method"].astype(str))
    reference_names = set(reference["Method"].astype(str))
    common = ranked & reference_names
    metrics = {
        "top1_agreement": float(
            str(ranking.iloc[0]["Method"]) == str(reference.iloc[0]["Method"])
        ),
        "pearson": math.nan,
        "spearman": math.nan,
        "kendall_tau": math.nan,
        "method_coverage": (
            len(common) / len(reference_names) if reference_names else math.nan
        ),
        "ranked_methods": len(ranked),
        "reference_methods": len(reference_names),
    }
    # Do not report correlations on a favorable subset when sparse samples
    # omit models from the full-data reference ranking.
    if metrics["method_coverage"] < 1.0 or len(common) < 2:
        return metrics
    left = (
        ranking[ranking["Method"].astype(str).isin(common)]
        .assign(_name=lambda x: x["Method"].astype(str))
        .set_index("_name")
        .sort_index()
    )
    right = (
        reference[reference["Method"].astype(str).isin(common)]
        .assign(_name=lambda x: x["Method"].astype(str))
        .set_index("_name")
        .sort_index()
    )
    metrics["spearman"] = float(
        spearmanr(left["ELO Score"], right["ELO Score"]).correlation
    )
    metrics["pearson"] = float(
        pearsonr(left["ELO Score"], right["ELO Score"]).statistic
    )
    metrics["kendall_tau"] = float(
        kendalltau(left["ELO Score"], right["ELO Score"]).correlation
    )
    return metrics


def _fit_or_load_reference(
    root: Path,
    dataset_id: str,
    df: pd.DataFrame,
    valid_users: Sequence[Any],
    method_id: str,
    method_config: Dict[str, Any],
    signature: str,
    device: str,
    recompute: bool,
) -> pd.DataFrame:
    output = root / dataset_id / "reference" / method_id
    ranking_path = output / "full_ranking.csv"
    metadata_path = output / "metadata.json"
    if not recompute and ranking_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        requested_smoothing = method_config.get("label_smoothing")
        cached_smoothing = metadata.get("label_smoothing")
        smoothing_matches = (
            requested_smoothing is None and cached_smoothing is None
        ) or (
            requested_smoothing is not None
            and cached_smoothing is not None
            and math.isclose(
                float(requested_smoothing),
                float(cached_smoothing),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
        if (
            metadata.get("data_signature") == signature
            and smoothing_matches
        ):
            print(f"[reference cached] {dataset_id}/{method_id}", flush=True)
            return pd.read_csv(ranking_path)

    print(f"[reference fit] {dataset_id}/{method_id} ...", flush=True)
    started = time.time()
    ranking, elapsed, fit_status, backend = _fit_ranking(
        df, valid_users, method_config, device
    )
    output.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(ranking_path, index=False)
    _json_dump(
        metadata_path,
        {
            "data_signature": signature,
            "method_id": method_id,
            "model": method_config["model"],
            "label_smoothing": method_config.get("label_smoothing"),
            "fit_time_seconds": elapsed,
            "wall_time_seconds": time.time() - started,
            "fit_status": fit_status,
            "backend": backend,
            "top1": str(ranking.iloc[0]["Method"]),
        },
    )
    print(
        f"[reference done] {dataset_id}/{method_id}: {elapsed:.3f}s, {backend}",
        flush=True,
    )
    return ranking


def _worker_init(context: Dict[str, Any]) -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = context


def _base_record(iteration: int, seed: int) -> Dict[str, Any]:
    c = _WORKER_CONTEXT
    m = c["method_config"]
    return {
        "dataset_id": c["dataset_id"],
        "dataset": c["dataset_label"],
        "group": c["group"],
        "axis": c["axis"],
        "scale_value": c["scale_value"],
        "method_id": c["method_id"],
        "method": m["label"],
        "model": m["model"],
        "shared_rater": m["shared_rater"],
        "iteration": iteration,
        "seed": seed,
        "status": "failed",
        "fit_status": "not_run",
        "backend": "",
        "error_type": "",
        "error_message": "",
        "top1_agreement": math.nan,
        "pearson": math.nan,
        "spearman": math.nan,
        "kendall_tau": math.nan,
        "time_seconds": math.nan,
        "method_coverage": math.nan,
        "ranked_methods": 0,
        "reference_methods": len(c["reference"]),
        "sample_rows": 0,
        "sampled_raters": 0,
        "mean_comparisons_per_sampled_rater": math.nan,
    }


def _run_worker(task: Tuple[int, int]) -> Dict[str, Any]:
    iteration, seed = task
    c = _WORKER_CONTEXT
    record = _base_record(iteration, seed)
    try:
        n_users = c["scale_value"] if c["axis"] == "n_raters" else c["max_raters"]
        n_comp = (
            c["scale_value"]
            if c["axis"] == "comparisons_per_rater"
            else None
        )
        sample, sample_raters = build_bootstrap_sample(
            df=c["df"],
            valid_users=c["valid_users"],
            n_users=n_users,
            n_comp_per_user=n_comp,
            seed=seed,
            preserve_golden=True,
        )
        ranking, elapsed, fit_status, backend = _fit_ranking(
            sample, sample_raters, c["method_config"], c["device"]
        )
        record.update(_ranking_metrics(ranking, c["reference"]))
        record.update(
            {
                "status": "success",
                "fit_status": fit_status,
                "backend": backend,
                "time_seconds": elapsed,
                "sample_rows": len(sample),
                "sampled_raters": len(sample_raters),
                "mean_comparisons_per_sampled_rater": (
                    len(sample) / len(sample_raters)
                    if len(sample_raters)
                    else math.nan
                ),
            }
        )
    except Exception as exc:
        record["error_type"] = type(exc).__name__
        record["error_message"] = " | ".join(
            line.strip() for line in traceback.format_exc(limit=4).splitlines()
        )
    return record


def _load_runs(path: Path) -> pd.DataFrame:
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


def _append_rows(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size:
        with path.open("r", newline="", encoding="utf-8") as handle:
            header = next(csv.reader(handle), [])
        if header != RUN_FIELDS:
            _load_runs(path).to_csv(path, index=False)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_FIELDS)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in RUN_FIELDS})
            handle.flush()


def _run_setting(
    root: Path,
    dataset_id: str,
    dataset_config: Dict[str, Any],
    df: pd.DataFrame,
    valid_users: Sequence[Any],
    axis: str,
    scale_value: int,
    method_id: str,
    method_config: Dict[str, Any],
    reference: pd.DataFrame,
    manifest: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    output = root / dataset_id / axis / str(scale_value) / method_id
    run_path = output / "runs.csv"
    previous = _load_runs(run_path)
    if previous.empty:
        finished = set()
    elif args.retry_failed:
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
    label = f"{dataset_id}/{axis}={scale_value}/{method_id}"
    if not tasks:
        print(f"[complete] {label}", flush=True)
        return

    context = {
        "dataset_id": dataset_id,
        "dataset_label": dataset_config["label"],
        "group": dataset_config["group"],
        "df": df,
        "valid_users": list(valid_users),
        "max_raters": len(valid_users),
        "axis": axis,
        "scale_value": scale_value,
        "method_id": method_id,
        "method_config": method_config,
        "reference": reference,
        "device": args.device,
    }
    actual_device = (
        resolve_device(args.device)
        if method_config["model"] in GPU_BAYESIAN_MODELS
        else "cpu"
    )
    workers = 1 if actual_device.startswith("cuda") else args.workers
    if workers == 1:
        _worker_init(context)
        iterator = map(_run_worker, tasks)
        _append_rows(run_path, tqdm(iterator, total=len(tasks), desc=label))
        return

    with Pool(
        processes=workers,
        initializer=_worker_init,
        initargs=(context,),
        maxtasksperchild=args.max_tasks_per_child,
    ) as pool:
        iterator = pool.imap_unordered(_run_worker, tasks, chunksize=1)
        _append_rows(run_path, tqdm(iterator, total=len(tasks), desc=label))


def _mean_std_se(values: pd.Series) -> Tuple[float, float, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return math.nan, math.nan, math.nan
    mean = float(clean.mean())
    std = float(clean.std(ddof=1)) if len(clean) > 1 else 0.0
    return mean, std, std / math.sqrt(len(clean))


def summarize_data_efficiency(
    summary: pd.DataFrame, thresholds: Sequence[float]
) -> pd.DataFrame:
    columns = [
        "dataset_id",
        "dataset",
        "group",
        "axis",
        "method_id",
        "method",
        "spearman_threshold",
        "achieved",
        "minimum_scale_value",
        "spearman_at_minimum",
        "maximum_spearman_observed",
        "failure_aware_spearman_at_minimum",
        "maximum_failure_aware_spearman_observed",
        "score_type",
        "minimum_valid_metric_rate",
    ]
    if summary.empty:
        return pd.DataFrame(columns=columns)

    rows: List[Dict[str, Any]] = []
    keys = ["dataset_id", "axis", "method_id"]
    for _, group in summary.groupby(keys, sort=True):
        group = group.sort_values("scale_value")
        valid = group[
            group["successful_runs"].gt(0)
            & pd.to_numeric(group["spearman_mean"], errors="coerce").notna()
            & pd.to_numeric(group["valid_metric_rate"], errors="coerce").ge(
                MIN_VALID_METRIC_RATE
            )
        ]
        first = group.iloc[0]
        for threshold in thresholds:
            reached = valid[
                valid["failure_aware_spearman_mean"].ge(threshold)
            ]
            crossing = reached.iloc[0] if not reached.empty else None
            rows.append(
                {
                    "dataset_id": first["dataset_id"],
                    "dataset": first["dataset"],
                    "group": first["group"],
                    "axis": first["axis"],
                    "method_id": first["method_id"],
                    "method": first["method"],
                    "spearman_threshold": threshold,
                    "achieved": crossing is not None,
                    "minimum_scale_value": (
                        int(crossing["scale_value"])
                        if crossing is not None
                        else math.nan
                    ),
                    "spearman_at_minimum": (
                        float(crossing["spearman_mean"])
                        if crossing is not None
                        else math.nan
                    ),
                    "maximum_spearman_observed": (
                        float(valid["spearman_mean"].max())
                        if not valid.empty
                        else math.nan
                    ),
                    "failure_aware_spearman_at_minimum": (
                        float(crossing["failure_aware_spearman_mean"])
                        if crossing is not None
                        else math.nan
                    ),
                    "maximum_failure_aware_spearman_observed": (
                        float(valid["failure_aware_spearman_mean"].max())
                        if not valid.empty
                        else math.nan
                    ),
                    "score_type": "failure_aware_spearman",
                    "minimum_valid_metric_rate": MIN_VALID_METRIC_RATE,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def summarize_results(
    root: Path, efficiency_thresholds: Sequence[float] = (0.90, 0.95)
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    global_repetition_cap: Optional[int] = None
    method_repetition_caps: Dict[str, int] = {}
    experiment_config_path = root / "experiment_config.json"
    if experiment_config_path.exists():
        experiment_config = json.loads(
            experiment_config_path.read_text(encoding="utf-8")
        )
        repetitions_override = experiment_config.get("repetitions_override")
        if repetitions_override is not None:
            global_repetition_cap = int(repetitions_override)
        method_repetition_caps = {
            str(method_id): int(repetitions)
            for method_id, repetitions in experiment_config.get(
                "method_repetition_caps", {}
            ).items()
        }
    pattern = "*/n_raters/*/*/runs.csv"
    paths = list(root.glob(pattern)) + list(
        root.glob("*/comparisons_per_rater/*/*/runs.csv")
    )
    for path in sorted(paths):
        runs = _load_runs(path)
        if runs.empty:
            continue
        caps = [
            cap
            for cap in (
                global_repetition_cap,
                method_repetition_caps.get(path.parent.name),
            )
            if cap is not None
        ]
        if caps:
            repetition_cap = min(caps)
            iteration = pd.to_numeric(runs["iteration"], errors="coerce")
            runs = runs[
                iteration.ge(0) & iteration.lt(repetition_cap)
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
        metric_runs = valid_fits[coverage.ge(1.0 - 1e-12)].copy()
        eligible = (
            runs["status"].eq("success")
            & ~runs["fit_status"].isin({"iteration_limit", "optimizer_warning"})
        )
        all_coverage = pd.to_numeric(runs["method_coverage"], errors="coerce")
        failure_top1_values = pd.to_numeric(
            runs["top1_agreement"], errors="coerce"
        ).where(eligible, 0.0).fillna(0.0)
        failure_pearson_values = failure_aware_correlation(
            runs["pearson"], all_coverage, eligible
        )
        failure_spearman_values = failure_aware_correlation(
            runs["spearman"], all_coverage, eligible
        )
        failure_kendall_values = failure_aware_correlation(
            runs["kendall_tau"], all_coverage, eligible
        )
        terminal_top1 = _mean_std_se(successful["top1_agreement"])
        terminal_pearson = _mean_std_se(terminal_full["pearson"])
        terminal_spearman = _mean_std_se(terminal_full["spearman"])
        terminal_kendall = _mean_std_se(terminal_full["kendall_tau"])
        top1 = _mean_std_se(valid_fits["top1_agreement"])
        pearson = _mean_std_se(metric_runs["pearson"])
        spearman = _mean_std_se(metric_runs["spearman"])
        kendall = _mean_std_se(metric_runs["kendall_tau"])
        failure_top1 = _mean_std_se(failure_top1_values)
        failure_pearson = _mean_std_se(failure_pearson_values)
        failure_spearman = _mean_std_se(failure_spearman_values)
        failure_kendall = _mean_std_se(failure_kendall_values)
        times = pd.to_numeric(valid_fits["time_seconds"], errors="coerce")
        first = runs.iloc[0]
        rows.append(
            {
                "dataset_id": first["dataset_id"],
                "dataset": first["dataset"],
                "group": first["group"],
                "axis": first["axis"],
                "scale_value": int(first["scale_value"]),
                "method_id": first["method_id"],
                "method": first["method"],
                "backend": (
                    successful["backend"].mode().iloc[0]
                    if not successful.empty
                    else ""
                ),
                "recorded_runs": len(runs),
                "successful_runs": len(successful),
                "terminal_metric_runs": len(terminal_full),
                "valid_fit_runs": len(valid_fits),
                "valid_fit_rate": len(valid_fits) / len(runs),
                "valid_metric_runs": len(metric_runs),
                "valid_metric_rate": len(metric_runs) / len(runs),
                "failed_runs": int((runs["status"] != "success").sum()),
                "partial_coverage_runs": int((coverage < 1.0 - 1e-12).sum()),
                "iteration_limit_runs": int(
                    (successful["fit_status"] == "iteration_limit").sum()
                ),
                "nonconvergence_rate": (
                    float((successful["fit_status"] == "iteration_limit").mean())
                    if len(successful)
                    else math.nan
                ),
                "optimizer_warnings": int(
                    (successful["fit_status"] == "optimizer_warning").sum()
                ),
                "terminal_top1_mean": terminal_top1[0],
                "terminal_top1_std": terminal_top1[1],
                "terminal_top1_se": terminal_top1[2],
                "terminal_pearson_mean": terminal_pearson[0],
                "terminal_pearson_std": terminal_pearson[1],
                "terminal_pearson_se": terminal_pearson[2],
                "terminal_spearman_mean": terminal_spearman[0],
                "terminal_spearman_std": terminal_spearman[1],
                "terminal_spearman_se": terminal_spearman[2],
                "terminal_kendall_mean": terminal_kendall[0],
                "terminal_kendall_std": terminal_kendall[1],
                "terminal_kendall_se": terminal_kendall[2],
                "top1_mean": top1[0],
                "top1_std": top1[1],
                "top1_se": top1[2],
                "pearson_mean": pearson[0],
                "pearson_std": pearson[1],
                "pearson_se": pearson[2],
                "kendall_mean": kendall[0],
                "kendall_std": kendall[1],
                "kendall_se": kendall[2],
                "spearman_mean": spearman[0],
                "spearman_std": spearman[1],
                "spearman_se": spearman[2],
                "failure_aware_top1_mean": failure_top1[0],
                "failure_aware_top1_std": failure_top1[1],
                "failure_aware_top1_se": failure_top1[2],
                "failure_aware_pearson_mean": failure_pearson[0],
                "failure_aware_pearson_std": failure_pearson[1],
                "failure_aware_pearson_se": failure_pearson[2],
                "failure_aware_spearman_mean": failure_spearman[0],
                "failure_aware_spearman_std": failure_spearman[1],
                "failure_aware_spearman_se": failure_spearman[2],
                "failure_aware_kendall_mean": failure_kendall[0],
                "failure_aware_kendall_std": failure_kendall[1],
                "failure_aware_kendall_se": failure_kendall[2],
                "mean_sample_rows": pd.to_numeric(
                    successful["sample_rows"], errors="coerce"
                ).mean(),
                "mean_sampled_raters": pd.to_numeric(
                    successful["sampled_raters"], errors="coerce"
                ).mean(),
                "mean_comparisons_per_sampled_rater": pd.to_numeric(
                    valid_fits["mean_comparisons_per_sampled_rater"],
                    errors="coerce",
                ).mean(),
                "mean_method_coverage": coverage.mean(),
                "time_mean_seconds": times.mean(),
                "time_median_seconds": times.median(),
            }
        )
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(
            ["dataset_id", "axis", "scale_value", "method_id"]
        )
    root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(root / "summary.csv", index=False)
    efficiency = summarize_data_efficiency(summary, efficiency_thresholds)
    efficiency.to_csv(root / "data_efficiency.csv", index=False)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG,
        help="JSON registry used to add datasets without changing this script.",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["default"],
        help="Dataset IDs from the config, or default/all.",
    )
    parser.add_argument(
        "--methods", nargs="+", default=["paper"],
        help="Method IDs, or paper/baseline/ours/shared/all.",
    )
    parser.add_argument(
        "--axes", nargs="+", choices=AXES, default=list(AXES),
    )
    parser.add_argument("--repetitions", type=int, default=None)
    parser.add_argument(
        "--crowd-bt-repetitions",
        type=int,
        default=DEFAULT_CROWD_BT_REPETITIONS,
        help=(
            "Maximum repetitions for Crowd-BT only. Other methods use "
            "--repetitions or the dataset default."
        ),
    )
    parser.add_argument(
        "--rater-grid", nargs="+", default=None,
        help="Optional override, for example: 1 2 5 10 max.",
    )
    parser.add_argument(
        "--comparison-grid", nargs="+", default=None,
        help="Optional override, for example: 1 2 5 10 max.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--efficiency-thresholds",
        type=float,
        nargs="+",
        default=[0.90, 0.95],
        help="Spearman thresholds used by data_efficiency.csv.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--max-tasks-per-child", type=int, default=50)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument(
        "--additive-only",
        action="store_true",
        help=(
            "Add only the selected methods to an existing results root "
            "without rewriting experiment_config.json or dataset metadata."
        ),
    )
    parser.add_argument("--recompute-reference", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--list", action="store_true", dest="list_options")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.config = args.config.resolve()
    args.results_root = args.results_root.resolve()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    if args.repetitions is not None and args.repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if args.crowd_bt_repetitions <= 0:
        raise ValueError("crowd-bt-repetitions must be positive")
    if any(not -1.0 <= value <= 1.0 for value in args.efficiency_thresholds):
        raise ValueError("efficiency-thresholds must be between -1 and 1")
    args.efficiency_thresholds = list(dict.fromkeys(args.efficiency_thresholds))
    if args.additive_only and not args.results_root.is_dir():
        raise ValueError(
            "--additive-only requires an existing --results-root"
        )

    datasets = _load_dataset_config(args.config)
    if args.list_options:
        print("Datasets:")
        for key, value in datasets.items():
            print(
                f"  {key:24s} {value['label']} "
                f"[{value['group']}, enabled={bool(value.get('enabled'))}]"
            )
        print("Methods:")
        for key, value in METHODS.items():
            paper = ", paper" if key in PAPER_METHODS else ""
            print(f"  {key:34s} {value['label']} [{value['group']}{paper}]")
        return
    if args.summary_only:
        summary = summarize_results(
            args.results_root, args.efficiency_thresholds
        )
        print(f"Wrote {len(summary)} rows to {args.results_root / 'summary.csv'}")
        return

    dataset_ids = _resolve_datasets(args.datasets, datasets)
    method_ids = _resolve_methods(args.methods)
    device_info = cuda_metadata(args.device)
    args.results_root.mkdir(parents=True, exist_ok=True)
    experiment_config_path = args.results_root / "experiment_config.json"
    if args.additive_only:
        if not experiment_config_path.exists():
            raise ValueError(
                "--additive-only requires an existing experiment_config.json"
            )
        existing_config = json.loads(
            experiment_config_path.read_text(encoding="utf-8")
        )
        if int(existing_config.get("seed", args.seed)) != args.seed:
            raise ValueError(
                "The requested seed does not match the existing experiment"
            )
        existing_repetitions = existing_config.get("repetitions_override")
        if args.repetitions is None and existing_repetitions is not None:
            args.repetitions = int(existing_repetitions)
        elif (
            args.repetitions is not None
            and existing_repetitions is not None
            and args.repetitions != int(existing_repetitions)
        ):
            raise ValueError(
                "The requested repetitions do not match the existing experiment"
            )
        existing_axes = set(existing_config.get("axes", AXES))
        if not set(args.axes).issubset(existing_axes):
            raise ValueError(
                "The requested axes are not part of the existing experiment"
            )
        additive_name = "__".join(method_ids)
        _json_dump(
            args.results_root / "additive_runs" / f"{additive_name}.json",
            {
                "dataset_config": str(args.config),
                "datasets": dataset_ids,
                "methods": method_ids,
                "method_configs": {
                    method_id: METHODS[method_id] for method_id in method_ids
                },
                "axes": args.axes,
                "repetitions": args.repetitions,
                "seed": args.seed,
                "device": args.device,
                "resolved_bayesian_backend": device_info,
                "workers": args.workers,
                "golden_protocol": "retain_for_rater_aware_methods",
            },
        )
    else:
        _json_dump(
            experiment_config_path,
            {
                "dataset_config": str(args.config),
                "datasets": dataset_ids,
                "methods": method_ids,
                "axes": args.axes,
                "repetitions_override": args.repetitions,
                "method_repetition_caps": {
                    "crowd_bt": args.crowd_bt_repetitions,
                },
                "seed": args.seed,
                "device": args.device,
                "resolved_bayesian_backend": device_info,
                "workers": args.workers,
                "efficiency_thresholds": args.efficiency_thresholds,
                "golden_protocol": "retain_for_rater_aware_methods",
            },
        )

    for dataset_id in dataset_ids:
        config = datasets[dataset_id]
        csv_path = Path(config["csv"])
        if not csv_path.is_absolute():
            csv_path = PROJECT_ROOT / csv_path
        df, valid_users = _load_data(csv_path)
        max_raters = len(valid_users)
        max_comparisons = int(df.groupby("answerer").size().max())
        rater_values = _resolve_grid(
            args.rater_grid or config["rater_grid"], max_raters
        )
        comparison_values = _resolve_grid(
            args.comparison_grid or config["comparison_grid"], max_comparisons
        )
        active_axes = [
            axis
            for axis in args.axes
            if not (axis == "n_raters" and max_raters <= 1)
        ]
        if "n_raters" in args.axes and "n_raters" not in active_axes:
            print(
                f"[skip] {dataset_id}/n_raters: only {max_raters} valid rater",
                flush=True,
            )
        repetitions = args.repetitions or int(config["repetitions"])
        signature = _data_signature(csv_path, df)
        _ensure_dataset_identity(
            args.results_root,
            dataset_id,
            {
                "data_signature": signature,
                "golden_protocol": "retain_for_rater_aware_methods",
            },
        )
        dataset_metadata_path = (
            args.results_root / dataset_id / "dataset_metadata.json"
        )
        dataset_metadata = {
            "dataset_id": dataset_id,
            "dataset": config["label"],
            "group": config["group"],
            "csv": str(csv_path),
            "rows": len(df),
            "raters": max_raters,
            "models": int(
                pd.concat([df["methodA"], df["methodB"]]).nunique()
            ),
            "maximum_comparisons_per_rater": max_comparisons,
            "rater_grid": rater_values,
            "comparison_grid": comparison_values,
            "active_axes": active_axes,
            "repetitions": repetitions,
            "data_signature": signature,
        }
        if args.additive_only:
            if not dataset_metadata_path.exists():
                raise ValueError(
                    f"Missing existing metadata for {dataset_id}"
                )
            existing_metadata = json.loads(
                dataset_metadata_path.read_text(encoding="utf-8")
            )
            if existing_metadata.get("data_signature") != signature:
                raise ValueError(
                    f"Dataset signature mismatch for {dataset_id}"
                )
        else:
            _json_dump(dataset_metadata_path, dataset_metadata)
        print(
            f"\nDataset {config['label']}: {len(df)} rows, {max_raters} raters; "
            f"R={rater_values}, K={comparison_values}, repetitions={repetitions}",
            flush=True,
        )

        references: Dict[str, pd.DataFrame] = {}
        for method_id in method_ids:
            references[method_id] = _fit_or_load_reference(
                args.results_root,
                dataset_id,
                df,
                valid_users,
                method_id,
                METHODS[method_id],
                signature,
                args.device,
                args.recompute_reference,
            )

        axis_values = {
            "n_raters": rater_values,
            "comparisons_per_rater": comparison_values,
        }
        for axis in active_axes:
            for scale_value in axis_values[axis]:
                manifest = _ensure_manifest(
                    args.results_root,
                    dataset_id,
                    axis,
                    scale_value,
                    repetitions,
                    args.seed,
                )
                for method_id in method_ids:
                    method_manifest = manifest
                    if method_id == "crowd_bt":
                        method_manifest = manifest.iloc[
                            : min(len(manifest), args.crowd_bt_repetitions)
                        ].copy()
                    _run_setting(
                        args.results_root,
                        dataset_id,
                        config,
                        df,
                        valid_users,
                        axis,
                        scale_value,
                        method_id,
                        METHODS[method_id],
                        references[method_id],
                        method_manifest,
                        args,
                    )
                if not args.additive_only:
                    summary = summarize_results(
                        args.results_root, args.efficiency_thresholds
                    )
                    print(
                        f"[table updated] {args.results_root / 'summary.csv'} "
                        f"({len(summary)} rows)",
                        flush=True,
                    )

    if args.additive_only:
        summary = summarize_results(
            args.results_root, args.efficiency_thresholds
        )
        print(
            f"[table updated] {args.results_root / 'summary.csv'} "
            f"({len(summary)} rows)",
            flush=True,
        )


if __name__ == "__main__":
    main()
