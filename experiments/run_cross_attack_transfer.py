#!/usr/bin/env python3
"""Cross-method transfer attacks over nested perturbation-budget prefixes."""

import _bootstrap  # noqa: F401
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from copy import deepcopy
import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

PROJECT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT / "experiment_results/victim_specific_robustness"
OUTPUT = PROJECT / "experiment_results/cross_attack_transfer_multibudget"
GAMMA_PATH = PROJECT / "experiment_results/rank_balance_gamma_validation/selected_gamma.json"
METHODS_USED = [
    "m_elo", "crowd_bt", "bayes_bt", "bbq", "am_elo",
    "rank_balance_tuned_full", "matched_gamma0",
]
LABELS = {
    "m_elo": "ELO",
    "crowd_bt": "Crowd-BT",
    "bayes_bt": "Bayes-BT",
    "bbq": "BBQ",
    "am_elo": "am-ELO",
    "rank_balance_tuned_full": "RankBalance (selected)",
    "matched_gamma0": "Matched (gamma=0)",
}
SOURCES = {
    "crowd_bt": "Crowd-BT source",
    "am_elo": "am-ELO source",
    "matched_gamma0": "Matched gamma=0 source",
    "rank_balance_selected": "RankBalance selected source",
}

from run_convergence_experiment import METHODS
from run_pollution_experiment import _fit_method, _load_data
from run_rank_balance_ablation import DEFAULT_CONFIG, DEFAULT_DATASETS, _load_datasets
from run_victim_specific_robustness import _candidate_scores, _score_map, _select_batch


def write_csv(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def clean_fit(dataset_id, method_id, frame):
    path = SOURCE_ROOT / dataset_id / method_id / "checkpoints/clean.pkl"
    if path.exists():
        try:
            return pickle.load(open(path, "rb"))["fit"]
        except Exception:
            pass
    raters = sorted(frame.answerer.astype(str).unique())
    return _fit_method(frame.copy(deep=True), raters, METHODS[method_id], "cpu")


def variant_config(gamma):
    config = deepcopy(METHODS["rank_balance_tuned_full"])
    config["rank_balance_config"]["gamma"] = float(gamma)
    return config


def variant_fit(frame, gamma):
    raters = sorted(frame.answerer.astype(str).unique())
    return _fit_method(frame.copy(deep=True), raters, variant_config(gamma), "cpu")


def source_fit(dataset_id, source_id, frame, args):
    if source_id in {"matched_gamma0", "rank_balance_selected"}:
        gamma = 0.0 if source_id == "matched_gamma0" else args.rank_balance_gamma
        return variant_fit(frame, gamma)
    return clean_fit(dataset_id, source_id, frame)


def victim_fit(dataset_id, victim_id, frame):
    if victim_id == "matched_gamma0":
        return variant_fit(frame, 0.0)
    return clean_fit(dataset_id, victim_id, frame)


def victim_config(victim_id):
    if victim_id == "matched_gamma0":
        return variant_config(0.0)
    return METHODS[victim_id]


def requested_budgets(size, fractions):
    return [(fraction, max(1, int(np.ceil(size * fraction)))) for fraction in fractions]


def source_order(frame, fit, attack, max_budget, top_ks, ridge):
    """Freeze one clean-fit source ordering so every budget is a true prefix."""
    clean_order = fit["ranking"].Method.astype(str).tolist()
    clean_scores = _score_map(fit["ranking"])
    scores = _candidate_scores(
        frame, fit, clean_order, clean_scores, top_ks, attack, ridge
    )
    return _select_batch(frame, frame, scores, set(), max_budget, attack)


def random_order(frame, attack, max_budget, seed):
    eligible = frame if attack == "delete" else frame.loc[frame.answerValue.isin(["A", "B"])]
    return eligible.sample(
        n=min(max_budget, len(eligible)), random_state=seed
    ).__row_id.astype(int).tolist()


def apply_attack(frame, row_ids, attack):
    selected = frame.__row_id.isin(set(row_ids))
    result = frame.copy(deep=True)
    if attack == "delete":
        return result.loc[~selected].reset_index(drop=True)
    result.loc[selected, "answerValue"] = result.loc[selected, "answerValue"].map(
        {"A": "B", "B": "A"}
    )
    return result.reset_index(drop=True)


def ranking_metrics(clean, attacked):
    clean_order = clean.Method.astype(str).tolist()
    attacked_order = attacked.Method.astype(str).tolist()
    common = sorted(set(clean_order) & set(attacked_order))
    clean_rank = {model: index for index, model in enumerate(clean_order)}
    attacked_rank = {model: index for index, model in enumerate(attacked_order)}
    left = [clean_rank[model] for model in common]
    right = [attacked_rank[model] for model in common]
    tau = kendalltau(left, right, variant="b").statistic
    rho = spearmanr(left, right).statistic
    max_change = max((abs(clean_rank[m] - attacked_rank[m]) for m in common), default=0)
    return float(tau), float(rho), int(max_change)


def topk_metrics(clean, attacked, top_k):
    clean_set = set(clean.Method.astype(str).head(top_k))
    attacked_set = set(attacked.Method.astype(str).head(top_k))
    union = clean_set | attacked_set
    changed = float(clean_set != attacked_set)
    jaccard = 1.0 - len(clean_set & attacked_set) / len(union)
    return changed, jaccard


def build_conditions(dataset_id, frame, budgets, args):
    conditions = []
    max_budget = max(count for _, count in budgets)
    for source_id, source_label in SOURCES.items():
        fit = source_fit(dataset_id, source_id, frame, args)
        for attack in ["delete", "flip"]:
            order = source_order(frame, fit, attack, max_budget, args.top_ks, args.ridge)
            for fraction, count in budgets:
                conditions.append(
                    (source_id, source_label, attack, 0, fraction, count, order[:count])
                )
    for seed in args.random_seeds:
        stable = int(hashlib.sha256(f"{dataset_id}:{seed}".encode()).hexdigest()[:8], 16)
        for attack in ["delete", "flip"]:
            order = random_order(frame, attack, max_budget, stable + (attack == "flip"))
            for fraction, count in budgets:
                conditions.append(
                    ("random", "Random source", attack, seed, fraction, count, order[:count])
                )
    return conditions


def run_dataset(dataset_id, settings, args):
    output = args.results_root / "raw" / f"{dataset_id}.csv"
    path = Path(settings["csv"])
    path = path if path.is_absolute() else PROJECT / path
    frame, _ = _load_data(path)
    budgets = requested_budgets(len(frame), args.budget_fractions)
    conditions = build_conditions(dataset_id, frame, budgets, args)
    existing = pd.read_csv(output) if output.exists() and not args.overwrite else pd.DataFrame()
    if not existing.empty and "victim_id" in existing:
        existing.loc[existing["victim_id"].eq("rank_balance_tuned_full"), "victim"] = LABELS["rank_balance_tuned_full"]
    key_columns = ["victim_id", "source_id", "attack_type", "source_seed", "requested_fraction"]
    done = set(existing[key_columns].itertuples(index=False, name=None)) if all(
        column in existing for column in key_columns
    ) else set()
    rows = existing.to_dict("records")

    for victim_id in args.victims:
        clean = victim_fit(dataset_id, victim_id, frame)
        pending = [
            condition for condition in conditions
            if (victim_id, condition[0], condition[2], condition[3], condition[4]) not in done
        ]

        def evaluate(condition):
            source_id, source_label, attack, source_seed, fraction, requested, row_ids = condition
            attacked_frame = apply_attack(frame, row_ids, attack)
            raters = sorted(attacked_frame.answerer.astype(str).unique())
            fit = _fit_method(attacked_frame, raters, victim_config(victim_id), "cpu")
            tau, rho, max_change = ranking_metrics(clean["ranking"], fit["ranking"])
            result = {
                "dataset_id": dataset_id,
                "victim_id": victim_id,
                "victim": LABELS[victim_id],
                "source_id": source_id,
                "source": source_label,
                "source_seed": source_seed,
                "attack_type": attack,
                "requested_budget": requested,
                "requested_fraction": fraction,
                "budget": len(row_ids),
                "budget_fraction": len(row_ids) / len(frame),
                "kendall_tau": tau,
                "kendall_distance": 1.0 - tau,
                "spearman_rho": rho,
                "spearman_distance": 1.0 - rho,
                "max_rank_change": max_change,
                "status": "success",
                "fit_status": fit["fit_status"],
                "rank_balance_gamma": args.rank_balance_gamma,
            }
            for top_k in args.top_ks:
                changed, jaccard = topk_metrics(clean["ranking"], fit["ranking"], top_k)
                result[f"top{top_k}_changed"] = changed
                result[f"top{top_k}_jaccard"] = jaccard
            return result

        with ThreadPoolExecutor(max_workers=min(args.condition_workers, len(pending) or 1)) as executor:
            futures = [executor.submit(evaluate, condition) for condition in pending]
            for future in as_completed(futures):
                rows.append(future.result())
                write_csv(output, pd.DataFrame(rows))
        print(f"{dataset_id}/{victim_id}: {len(pending)} new conditions", flush=True)
    print(f"{dataset_id}: {len(rows)} source-victim-budget conditions complete", flush=True)


def bootstrap_values(values, seed, repetitions):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(repetitions, len(values)))
    samples = values[indices].mean(axis=1)
    return values.mean(), np.quantile(samples, 0.025), np.quantile(samples, 0.975)


def aggregate(root, args):
    paths = sorted((root / "raw").glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No raw CSV files found under {root / 'raw'}")
    raw = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    top_changed = [f"top{k}_changed" for k in args.top_ks]
    top_jaccard = [f"top{k}_jaccard" for k in args.top_ks]
    raw["overall_changed"] = raw[top_changed].mean(axis=1)
    raw["overall_jaccard"] = raw[top_jaccard].mean(axis=1)
    write_csv(root / "transfer_runs.csv", raw)

    metrics = [
        "overall_changed", "overall_jaccard", "kendall_tau", "kendall_distance",
        "spearman_rho", "spearman_distance", "max_rank_change",
    ] + top_changed + top_jaccard
    dataset_keys = [
        "dataset_id", "source_id", "source", "victim_id", "victim",
        "attack_type", "requested_fraction",
    ]
    dataset = raw.groupby(dataset_keys, as_index=False)[metrics].mean()
    write_csv(root / "dataset_budget_metrics.csv", dataset)

    summary_rows = []
    group_keys = ["source_id", "source", "victim_id", "victim", "attack_type", "requested_fraction"]
    for group_number, (key, group) in enumerate(dataset.groupby(group_keys, sort=True)):
        base = dict(zip(group_keys, key))
        base["datasets"] = group.dataset_id.nunique()
        for metric_number, metric in enumerate(metrics):
            mean, low, high = bootstrap_values(
                group[metric], args.bootstrap_seed + group_number * 100 + metric_number,
                args.bootstrap_repetitions,
            )
            base[metric] = mean
            base[f"{metric}_ci_lower"] = low
            base[f"{metric}_ci_upper"] = high
        summary_rows.append(base)
    operation_summary = pd.DataFrame(summary_rows)
    write_csv(root / "budget_curve_by_operation_with_ci.csv", operation_summary)

    combined = dataset.groupby(
        ["dataset_id", "source_id", "source", "victim_id", "victim", "requested_fraction"],
        as_index=False,
    )[metrics].mean()
    combined_rows = []
    combined_keys = ["source_id", "source", "victim_id", "victim", "requested_fraction"]
    for group_number, (key, group) in enumerate(combined.groupby(combined_keys, sort=True)):
        base = dict(zip(combined_keys, key))
        base["datasets"] = group.dataset_id.nunique()
        for metric_number, metric in enumerate(metrics):
            mean, low, high = bootstrap_values(
                group[metric], args.bootstrap_seed + 100000 + group_number * 100 + metric_number,
                args.bootstrap_repetitions,
            )
            base[metric] = mean
            base[f"{metric}_ci_lower"] = low
            base[f"{metric}_ci_upper"] = high
        combined_rows.append(base)
    curve = pd.DataFrame(combined_rows)
    write_csv(root / "budget_curve_with_ci.csv", curve)

    auc_rows = []
    for (source_id, source, victim_id, victim), group in curve.groupby(
        ["source_id", "source", "victim_id", "victim"]
    ):
        group = group.sort_values("requested_fraction")
        x = group.requested_fraction.to_numpy(float)
        scale = x[-1] - x[0]
        row = {"source_id": source_id, "source": source, "victim_id": victim_id, "victim": victim}
        for metric in ["overall_changed", "overall_jaccard", "kendall_distance", "spearman_distance"]:
            row[f"{metric}_auc"] = np.trapz(group[metric].to_numpy(float), x) / scale
        auc_rows.append(row)
    write_csv(root / "budget_auc.csv", pd.DataFrame(auc_rows))

    largest = curve.loc[np.isclose(curve.requested_fraction, max(args.budget_fractions))]
    write_csv(root / "transfer_matrix_5pct_with_ci.csv", largest)
    pivot = largest.pivot(index="source", columns="victim", values="overall_changed")
    write_csv(root / "transfer_matrix_5pct.csv", pivot.reset_index())
    print("\nOverall Top-k change rate at the 5% budget")
    print(pivot.to_string(float_format=lambda value: f"{value:.3f}"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--results-root", type=Path, default=OUTPUT)
    parser.add_argument(
        "--budget-fractions", nargs="+", type=float,
        default=[0.0005, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05],
    )
    parser.add_argument("--top-ks", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--victims", nargs="+", choices=METHODS_USED, default=METHODS_USED)
    parser.add_argument("--random-seeds", nargs="+", type=int, default=[42, 2023, 3407, 7919, 15401])
    parser.add_argument("--condition-workers", type=int, default=8)
    parser.add_argument("--ridge", type=float, default=1e-5)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260918)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--no-aggregate", action="store_true")
    args = parser.parse_args()
    args.results_root = args.results_root.resolve()
    args.results_root.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        aggregate(args.results_root, args)
        return
    selected_gamma = float(json.loads(GAMMA_PATH.read_text(encoding="utf-8"))["gamma"])
    args.rank_balance_gamma = selected_gamma
    METHODS["rank_balance_tuned_full"]["rank_balance_config"]["gamma"] = selected_gamma
    definitions = _load_datasets(args.config.resolve())
    for dataset_id in args.datasets:
        run_dataset(dataset_id, definitions[dataset_id], args)
    if not args.no_aggregate:
        aggregate(args.results_root, args)


if __name__ == "__main__":
    main()
