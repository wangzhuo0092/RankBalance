#!/usr/bin/env python3
"""Held-out utility for Ability-only, Reliability-only, and Flip-only."""

import _bootstrap  # noqa: F401
from concurrent.futures import ThreadPoolExecutor
import argparse, json, os
from pathlib import Path
import sys
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
OUTPUT = PROJECT / "experiment_results" / "rank_balance_component_utility"
DEFAULT_GAMMA_PATH = PROJECT / "experiment_results/rank_balance_gamma_validation/selected_gamma.json"
VARIANTS = {
    "ability_only": "rank_balance_ability_only",
    "reliability_only": "rank_balance_reliability_only",
    "flip_only": "rank_balance_flip_only",
}

from rank_balance_ablation import decisive_preference_metrics, fit_rank_balance
from run_convergence_experiment import METHODS
from run_pollution_experiment import _load_data
from run_rank_balance_ablation import DEFAULT_CONFIG, DEFAULT_DATASETS, _load_datasets, _split_data


def write_csv(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    frame.to_csv(temp, index=False); os.replace(temp, path)


def run_dataset(dataset_id, settings, args):
    output = args.results_root / "raw" / f"{dataset_id}.csv"
    if output.exists() and not args.overwrite:
        print(f"Resume component utility: {dataset_id}", flush=True); return
    path = Path(settings["csv"])
    if not path.is_absolute(): path = PROJECT / path
    frame, _ = _load_data(path)
    tasks = []
    for seed in args.seeds:
        train, _, test, _ = _split_data(frame, seed)
        models = sorted(set(train.methodA.astype(str)) | set(train.methodB.astype(str)))
        raters = sorted(train.answerer.astype(str).unique())
        for variant, method_id in VARIANTS.items():
            tasks.append((seed, train, test, models, raters, variant, method_id))

    def evaluate(task):
        seed, train, test, models, raters, variant, method_id = task
        config = METHODS[method_id]["rank_balance_config"]
        fit = fit_rank_balance(train.copy(deep=True), config, models=models, raters=raters)
        metric = decisive_preference_metrics(test, fit)
        return {"dataset_id":dataset_id, "seed":seed, "variant":variant,
                "method_id":method_id, "test_nll":metric["nll"],
                "test_accuracy":metric["accuracy"], "test_rows":metric["rows"],
                "fit_success":fit["success"], "time_seconds":fit["elapsed"]}

    with ThreadPoolExecutor(max_workers=min(args.fit_workers, len(tasks))) as executor:
        rows = list(executor.map(evaluate, tasks))
    write_csv(output, pd.DataFrame(rows))


def aggregate(root):
    runs = pd.concat([pd.read_csv(path) for path in sorted((root/"raw").glob("*.csv"))], ignore_index=True)
    write_csv(root/"utility_runs.csv", runs)
    dataset = runs.groupby(["dataset_id","variant","method_id"], as_index=False)[["test_nll","test_accuracy"]].mean()
    write_csv(root/"utility_dataset_level.csv", dataset)
    summary = dataset.groupby(["variant","method_id"])[["test_nll","test_accuracy"]].agg(["mean","std","count"])
    summary.to_csv(root/"utility_summary.csv")
    print(summary.to_string())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=DEFAULT_CONFIG)
    parser.add_argument("--datasets",nargs="+",default=DEFAULT_DATASETS)
    parser.add_argument("--results-root",type=Path,default=OUTPUT)
    parser.add_argument("--gamma-path",type=Path,default=DEFAULT_GAMMA_PATH)
    parser.add_argument("--seeds",nargs="+",type=int,default=[42,2023,3407,7919,15401])
    parser.add_argument("--fit-workers",type=int,default=8)
    parser.add_argument("--overwrite",action="store_true")
    parser.add_argument("--summarize-only",action="store_true")
    parser.add_argument("--no-aggregate",action="store_true")
    args=parser.parse_args(); args.results_root=args.results_root.resolve(); args.results_root.mkdir(parents=True,exist_ok=True)
    if args.summarize_only: aggregate(args.results_root); return
    selected_gamma=float(json.loads(args.gamma_path.resolve().read_text(encoding="utf-8"))["gamma"])
    for method_id in VARIANTS.values():
        METHODS[method_id]["rank_balance_config"]["gamma"]=selected_gamma
    definitions=_load_datasets(args.config.resolve())
    for dataset_id in args.datasets: run_dataset(dataset_id,definitions[dataset_id],args)
    if not args.no_aggregate: aggregate(args.results_root)


if __name__=="__main__": main()
