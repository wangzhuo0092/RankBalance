#!/usr/bin/env python3
"""Known-truth ranking experiment using real comparison graphs and fixed BT attacks."""

import _bootstrap  # noqa: F401
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import kendalltau

from run_pollution_experiment import _load_data
from rank_balance import _fit_initial_bt
from rank_balance_ablation import (
    ELO_OFFSET, ELO_SCALE, _arrays, decisive_preference_metrics, fit_rank_balance,
)
from run_cross_attack_transfer import apply_attack, source_order, variant_config
from run_rank_balance_ablation import DEFAULT_CONFIG, _load_datasets, _split_data


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT / "experiment_results/synthetic_truth_attack"
DEFAULT_GAMMA_PATH = PROJECT / "experiment_results/rank_balance_gamma_validation/selected_gamma.json"
DEFAULT_DATASETS = ("conha", "wd", "hific")
METHODS = ("bt", "matched_gamma0", "rank_balance_selected")
PROTOCOL = "known_truth_fixed_bt_v1"


def stable_seed(*parts):
    payload = ":".join(map(str, parts)).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16) % (2**32)


def atomic_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    frame.to_csv(temp, index=False)
    os.replace(temp, path)


def synthetic_frame(frame, dataset_id, seed):
    models = sorted(set(frame.methodA.astype(str)) | set(frame.methodB.astype(str)))
    if len(models) < 6:
        raise ValueError(f"Top-5 comparison requires at least 6 models: {dataset_id}")
    truth_rng = np.random.default_rng(stable_seed(dataset_id, seed, "abilities"))
    theta = truth_rng.standard_normal(len(models))
    theta -= theta.mean()
    values = dict(zip(models, theta))
    probabilities = expit(
        frame.methodA.astype(str).map(values).to_numpy(float)
        - frame.methodB.astype(str).map(values).to_numpy(float)
    )
    label_rng = np.random.default_rng(stable_seed(dataset_id, seed, "labels"))
    synthetic = frame.copy(deep=True)
    synthetic["answerValue"] = np.where(label_rng.random(len(frame)) < probabilities, "A", "B")
    truth = pd.DataFrame({"Method": models, "ability": theta})
    truth = truth.sort_values(["ability", "Method"], ascending=[False, True]).reset_index(drop=True)
    return synthetic, models, truth


def fit_method(frame, models, raters, method, selected_gamma):
    if method == "bt":
        config = variant_config(0.0)["rank_balance_config"]
        design = _arrays(frame, models=models, raters=raters)
        theta = _fit_initial_bt(
            len(models), design["a"], design["b"], design["sign"],
            design["weight"], config["lambda_theta"],
            int(config["max_iter"]), float(config["tol"]),
        )
        ranking = pd.DataFrame({"Method": models, "ELO Score": ELO_OFFSET + ELO_SCALE * theta})
        ranking = ranking.sort_values(["ELO Score", "Method"], ascending=[False, True]).reset_index(drop=True)
        return {"ranking": ranking, "theta": theta, "models": models,
                "rater_values": {}, "fit_status": "not_reported"}

    gamma = 0.0 if method == "matched_gamma0" else selected_gamma
    fit = fit_rank_balance(frame, variant_config(gamma)["rank_balance_config"],
                           models=models, raters=raters)
    fit["fit_status"] = "converged" if fit["success"] else "optimizer_warning"
    fit["rater_values"] = dict(zip(fit["design"]["raters"], expit(fit["u"])))
    return fit


def evaluate(fit, method, truth, test):
    true_order = truth.Method.astype(str).tolist()
    estimated = fit["ranking"].Method.astype(str).tolist()
    if set(estimated) != set(true_order):
        raise ValueError("Fitted ranking does not include exactly the true model set")
    actual_rank = {model: i for i, model in enumerate(true_order)}
    fitted_rank = {model: i for i, model in enumerate(estimated)}
    tau = kendalltau([actual_rank[model] for model in true_order],
                     [fitted_rank[model] for model in true_order], variant="b").statistic
    top5 = len(set(true_order[:5]) & set(estimated[:5])) / 5.0
    if method == "bt":
        index = {model: i for i, model in enumerate(fit["models"])}
        a = test.methodA.astype(str).map(index).to_numpy(int)
        b = test.methodB.astype(str).map(index).to_numpy(int)
        target = test.answerValue.eq("A").to_numpy(float)
        margin = fit["theta"][a] - fit["theta"][b]
        nll = np.mean(np.logaddexp(0.0, margin) - target * margin)
    else:
        metrics = decisive_preference_metrics(test, fit)
        if metrics["rows"] != len(test):
            raise ValueError("Test evaluation dropped synthetic comparisons")
        nll = metrics["nll"]
    return {"tau": float(tau), "top5": float(top5), "test_nll": float(nll)}


def dataset_seed_task(dataset_id, csv_path, seed, budgets, top_ks, selected_gamma, output):
    target = output / "raw" / dataset_id / f"seed_{seed}.csv"
    identity = {"protocol": PROTOCOL, "dataset_id": dataset_id, "seed": seed,
                "budgets": list(budgets), "top_ks": list(top_ks), "methods": list(METHODS),
                "selected_gamma": selected_gamma,
                "data_path": str(csv_path.resolve()), "data_mtime_ns": csv_path.stat().st_mtime_ns}
    metadata = target.with_suffix(".json")
    if target.exists() and metadata.exists():
        if json.loads(metadata.read_text(encoding="utf-8")) != identity:
            raise ValueError(f"Saved scenario uses a different protocol: {target}")
        saved = pd.read_csv(target)
        if len(saved) != len(METHODS) * (1 + 2 * len(budgets)):
            raise ValueError(f"Incomplete saved scenario: {target}")
        return target

    frame, _ = _load_data(csv_path)
    synthetic, models, truth = synthetic_frame(frame, dataset_id, seed)
    train, validation, test, split = _split_data(synthetic, seed)
    if test.empty or validation.empty:
        raise ValueError("The synthetic split has an empty held-out partition")
    raters = sorted(train.answerer.astype(str).unique())
    clean = {method: fit_method(train, models, raters, method, selected_gamma) for method in METHODS}
    clean_metrics = {method: evaluate(fit, method, truth, test) for method, fit in clean.items()}
    max_budget = max(max(1, int(np.ceil(len(train) * fraction))) for fraction in budgets)

    rows = []
    for method in METHODS:
        rows.append({"dataset_id": dataset_id, "seed": seed, "method": method,
                     "operation": "clean", "budget_fraction": 0.0, "budget": 0,
                     "train_rows": len(train), "validation_rows": len(validation), "test_rows": len(test),
                     "fit_status": clean[method]["fit_status"], **clean_metrics[method]})

    manifest = []
    for operation in ("delete", "flip"):
        order = source_order(train, clean["bt"], operation, max_budget, top_ks, 1e-5)
        manifest.extend({"operation": operation, "selection_order": index + 1, "row_id": row_id}
                        for index, row_id in enumerate(order))
        for fraction in budgets:
            requested = max(1, int(np.ceil(len(train) * fraction)))
            edited = apply_attack(train, order[:requested], operation)
            edited_raters = sorted(edited.answerer.astype(str).unique())
            for method in METHODS:
                fit = fit_method(edited, models, edited_raters, method, selected_gamma)
                metrics = evaluate(fit, method, truth, test)
                rows.append({"dataset_id": dataset_id, "seed": seed, "method": method,
                             "operation": operation, "budget_fraction": fraction,
                             "budget": min(requested, len(order)), "train_rows": len(train),
                             "validation_rows": len(validation), "test_rows": len(test),
                             "fit_status": fit["fit_status"], **metrics})

    atomic_csv(truth, target.with_name(f"seed_{seed}_truth.csv"))
    atomic_csv(synthetic[["__row_id", "answerValue"]], target.with_name(f"seed_{seed}_labels.csv"))
    atomic_csv(split, target.with_name(f"seed_{seed}_split.csv"))
    atomic_csv(pd.DataFrame(manifest), target.with_name(f"seed_{seed}_attack_rows.csv"))
    atomic_csv(pd.DataFrame(rows), target)
    temporary = metadata.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(identity, indent=2), encoding="utf-8")
    os.replace(temporary, metadata)
    return target


def bootstrap(values, rng, repetitions):
    values = np.asarray(values, dtype=float)
    samples = values[rng.integers(len(values), size=(repetitions, len(values)))].mean(axis=1)
    return values.mean(), *np.quantile(samples, [0.025, 0.975])


def aggregate(root, datasets, seeds, budgets, repetitions, bootstrap_seed):
    paths = [root / "raw" / dataset / f"seed_{seed}.csv"
             for dataset in datasets for seed in seeds]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise ValueError(f"Missing {len(missing)} dataset-seed results; first: {missing[0]}")
    raw = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    expected = len(paths) * len(METHODS) * (1 + 2 * len(budgets))
    if len(raw) != expected or raw[["tau", "top5", "test_nll"]].isna().any().any():
        raise ValueError("Some raw results are missing or contain invalid metrics")
    atomic_csv(raw, root / "raw_results.csv")

    status = raw.groupby(["dataset_id", "method", "fit_status"], dropna=False).size().rename("fits").reset_index()
    atomic_csv(status, root / "convergence_counts.csv")
    clean = raw.loc[raw.operation.eq("clean"), ["dataset_id", "seed", "method", "tau", "top5", "test_nll"]]
    attacked = raw.loc[~raw.operation.eq("clean")].merge(
        clean, on=["dataset_id", "seed", "method"], validate="many_to_one", suffixes=("_attacked", "_clean")
    )
    for metric in ("tau", "top5", "test_nll"):
        attacked[f"delta_{metric}"] = attacked[f"{metric}_attacked"] - attacked[f"{metric}_clean"]
    atomic_csv(attacked, root / "paired_seed_results.csv")

    rng = np.random.default_rng(bootstrap_seed)
    absolute_metrics = [f"{metric}_{stage}" for metric in ("tau", "top5", "test_nll")
                        for stage in ("clean", "attacked")]
    absolute_metrics += [f"delta_{metric}" for metric in ("tau", "top5", "test_nll")]
    absolute_rows = []
    for key, part in attacked.groupby(["dataset_id", "operation", "budget_fraction", "method"], sort=True):
        if set(part.seed) != set(seeds):
            raise ValueError(f"Unpaired or missing seeds for {key}")
        row = dict(zip(["dataset_id", "operation", "budget_fraction", "method"], key))
        row["seeds"] = len(part)
        for metric in absolute_metrics:
            row[metric], row[f"{metric}_ci_lower"], row[f"{metric}_ci_upper"] = bootstrap(
                part[metric], rng, repetitions
            )
        absolute_rows.append(row)
    absolute_summary = pd.DataFrame(absolute_rows)
    atomic_csv(absolute_summary, root / "truth_accuracy_by_dataset_with_ci.csv")

    matched = attacked.loc[attacked.method.eq("matched_gamma0")]
    keys = ["dataset_id", "seed", "operation", "budget_fraction"]
    comparisons = attacked.loc[~attacked.method.eq("matched_gamma0")].merge(
        matched[keys + ["tau_clean", "tau_attacked", "top5_clean", "top5_attacked",
                        "test_nll_clean", "test_nll_attacked", "delta_tau", "delta_top5", "delta_test_nll"]],
        on=keys, validate="many_to_one", suffixes=("", "_matched"),
    )
    for metric in ("tau", "top5", "test_nll"):
        direction = -1 if metric == "test_nll" else 1
        for stage in ("clean", "attacked"):
            column = f"{metric}_{stage}"
            comparisons[f"advantage_{column}"] = direction * (comparisons[column] - comparisons[f"{column}_matched"])
        comparisons[f"advantage_delta_{metric}"] = direction * (
            comparisons[f"delta_{metric}"] - comparisons[f"delta_{metric}_matched"]
        )
    atomic_csv(comparisons, root / "vs_matched_by_seed.csv")

    metrics = [f"advantage_{metric}_{stage}" for metric in ("tau", "top5", "test_nll")
               for stage in ("clean", "attacked")]
    metrics += [f"advantage_delta_{metric}" for metric in ("tau", "top5", "test_nll")]
    rows = []
    for group_key, part in comparisons.groupby(["dataset_id", "operation", "budget_fraction", "method"], sort=True):
        if set(part.seed) != set(seeds):
            raise ValueError(f"Unpaired or missing seeds for {group_key}")
        row = dict(zip(["dataset_id", "operation", "budget_fraction", "method"], group_key))
        row["seeds"] = len(part)
        for metric in metrics:
            row[metric], row[f"{metric}_ci_lower"], row[f"{metric}_ci_upper"] = bootstrap(
                part[metric], rng, repetitions
            )
        rows.append(row)
    summary = pd.DataFrame(rows)
    atomic_csv(summary, root / "vs_matched_by_dataset_with_ci.csv")
    print("Truth-relative performance at the largest attack budget:")
    print(absolute_summary.loc[absolute_summary.budget_fraction.eq(max(budgets)),
          ["dataset_id", "operation", "method", "tau_clean", "tau_attacked",
           "top5_clean", "top5_attacked", "test_nll_clean", "test_nll_attacked"]].to_string(index=False))
    print("Paired effects: positive advantage favors the method over matched gamma=0")
    print(summary.loc[summary.budget_fraction.eq(max(budgets)),
                      ["dataset_id", "operation", "method", "advantage_tau_attacked",
                       "advantage_tau_attacked_ci_lower", "advantage_tau_attacked_ci_upper",
                       "advantage_top5_attacked", "advantage_test_nll_attacked"]].to_string(index=False))
    print("\nFit statuses:")
    print(status.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--num-seeds", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=2023)
    parser.add_argument("--budgets", nargs="+", type=float, default=[0.001, 0.005, 0.01])
    parser.add_argument("--top-ks", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--gamma-path", type=Path, default=DEFAULT_GAMMA_PATH)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.num_seeds < 1 or not args.budgets or any(
        budget <= 0 or budget > 1 for budget in args.budgets
    ):
        parser.error("workers, num-seeds and budgets must be positive; budgets must be <= 1")
    args.results_root = args.results_root.resolve()
    selected_gamma = float(json.loads(args.gamma_path.resolve().read_text(encoding="utf-8"))["gamma"])
    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))
    budgets = sorted(set(args.budgets))
    if not args.summarize_only:
        definitions = _load_datasets(args.config.resolve())
        jobs = []
        for dataset_id in args.datasets:
            path = Path(definitions[dataset_id]["csv"])
            csv_path = path if path.is_absolute() else PROJECT / path
            jobs.extend((dataset_id, csv_path, seed, budgets, args.top_ks, selected_gamma, args.results_root)
                        for seed in seeds)
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(dataset_seed_task, *job) for job in jobs]
            for future in as_completed(futures):
                print(f"Completed {future.result()}", flush=True)
    aggregate(args.results_root, args.datasets, seeds, budgets,
              args.bootstrap_repetitions, stable_seed(PROTOCOL, "bootstrap"))


if __name__ == "__main__":
    main()
