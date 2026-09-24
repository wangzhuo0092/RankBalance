#!/usr/bin/env python3
"""RankBalance stability--utility curve over gamma/gamma*."""

import _bootstrap  # noqa: F401
from concurrent.futures import ThreadPoolExecutor
import argparse, json, math, os
from pathlib import Path
import sys
import numpy as np
import pandas as pd

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
BASE=PROJECT/"experiment_results/experiment_7a_rank_robustness"
EXT=PROJECT/"experiment_results/experiment_7a_extended_attack_cost"
OUTPUT=PROJECT/"experiment_results/rank_balance_gamma_curve"
DEFAULT_GAMMA_PATH=PROJECT/"experiment_results/rank_balance_gamma_validation/selected_gamma.json"
RATIOS=[0,.125,.25,.5,.75,1,1.5,2,3,4,8,16,100/3]

from rank_balance_ablation import decisive_preference_metrics, fit_rank_balance
from run_pollution_experiment import _load_data
from run_rank_balance_ablation import DEFAULT_CONFIG, DEFAULT_DATASETS, _load_datasets, _split_data


def write_csv(path,frame):
    path.parent.mkdir(parents=True,exist_ok=True); temp=path.with_name(path.name+".tmp")
    frame.to_csv(temp,index=False); os.replace(temp,path)


def gamma_slug(value): return f"{value:.8g}".replace(".","p")


def conditions_for(dataset_id):
    conditions=[]
    path=BASE/"rankamip"/dataset_id/"attack_manifest.csv"
    if path.exists():
        manifest=pd.read_csv(path)
        for (attack,k), group in manifest.groupby(["attack_type","top_k"]):
            conditions.append({"attack_type":attack,"top_k":int(k),"row_ids":group.sort_values("attack_order").row_id.astype(int).tolist()})
    additions=pd.read_csv(EXT/"supplemental_attack_costs.csv")
    for row in additions.loc[additions.dataset_id.eq(dataset_id)].itertuples(index=False):
        selected=pd.read_csv(EXT/"conditions"/dataset_id/f"{row.attack_type}_top{int(row.top_k)}"/"selected_comparisons.csv.gz")
        conditions.append({"attack_type":row.attack_type,"top_k":int(row.top_k),"row_ids":selected.__row_id.astype(int).tolist()})
    return conditions


def apply_attack(frame,condition):
    mask=frame.__row_id.isin(set(condition["row_ids"])); result=frame.copy(deep=True)
    if condition["attack_type"]=="delete": return result.loc[~mask].reset_index(drop=True)
    result.loc[mask,"answerValue"]=result.loc[mask,"answerValue"].map({"A":"B","B":"A"})
    return result.reset_index(drop=True)


def topk_changed(clean,attacked,k):
    return float(set(clean.Method.astype(str).head(k)) != set(attacked.Method.astype(str).head(k)))


def run_gamma(dataset_id,frame,conditions,gamma,ratio,args):
    folder=args.results_root/dataset_id/f"gamma_{gamma_slug(gamma)}"
    attack_path,utility_path=folder/"attacks.csv",folder/"utility.csv"
    if attack_path.exists() and utility_path.exists() and not args.overwrite: return
    config={"gamma":gamma,"regularizer":"full","lambda_theta":1.0,"lambda_u":1.0,"mu":2.0,"max_iter":1000,"tol":1e-8}
    models=sorted(set(frame.methodA.astype(str))|set(frame.methodB.astype(str))); raters=sorted(frame.answerer.astype(str).unique())
    clean=fit_rank_balance(frame.copy(deep=True),config,models=models,raters=raters)

    def attack_fit(condition):
        attacked_frame=apply_attack(frame,condition)
        fit=fit_rank_balance(attacked_frame,config,models=models,raters=raters)
        return {"dataset_id":dataset_id,"gamma":gamma,"ratio":ratio,"attack_type":condition["attack_type"],"top_k":condition["top_k"],"topk_changed":topk_changed(clean["ranking"],fit["ranking"],condition["top_k"]),"edited_rows":len(condition["row_ids"]),"edited_fraction":len(condition["row_ids"])/len(frame),"fit_success":fit["success"]}
    with ThreadPoolExecutor(max_workers=min(args.condition_workers,len(conditions) or 1)) as executor:
        attack_rows=list(executor.map(attack_fit,conditions))
    write_csv(attack_path,pd.DataFrame(attack_rows))

    def utility_fit(seed):
        train,_,test,_=_split_data(frame,seed)
        train_models=sorted(set(train.methodA.astype(str))|set(train.methodB.astype(str))); train_raters=sorted(train.answerer.astype(str).unique())
        fit=fit_rank_balance(train,config,models=train_models,raters=train_raters)
        metric=decisive_preference_metrics(test,fit)
        return {"dataset_id":dataset_id,"seed":seed,"gamma":gamma,"ratio":ratio,"test_nll":metric["nll"],"test_accuracy":metric["accuracy"],"test_rows":metric["rows"],"fit_success":fit["success"]}
    with ThreadPoolExecutor(max_workers=len(args.seeds)) as executor:
        utility_rows=list(executor.map(utility_fit,args.seeds))
    write_csv(utility_path,pd.DataFrame(utility_rows))


def run_dataset(dataset_id,settings,args):
    path=Path(settings["csv"]); path=path if path.is_absolute() else PROJECT/path
    frame,_=_load_data(path); conditions=conditions_for(dataset_id)
    tasks=[(args.gamma_star*r,r) for r in args.ratios]
    with ThreadPoolExecutor(max_workers=min(args.gamma_workers,len(tasks))) as executor:
        list(executor.map(lambda item:run_gamma(dataset_id,frame,conditions,item[0],item[1],args),tasks))


def bootstrap(values,seed,repetitions):
    rng=np.random.default_rng(seed); indices=rng.integers(0,len(values),size=(repetitions,len(values)))
    samples=values[indices].mean(axis=1); return values.mean(),np.quantile(samples,.025),np.quantile(samples,.975)


def aggregate(root,args):
    attacks=pd.concat([pd.read_csv(p) for p in root.glob("*/gamma_*/attacks.csv")],ignore_index=True)
    utility=pd.concat([pd.read_csv(p) for p in root.glob("*/gamma_*/utility.csv")],ignore_index=True)
    write_csv(root/"attack_runs.csv",attacks); write_csv(root/"utility_runs.csv",utility)
    da=attacks.groupby(["dataset_id","ratio","gamma","attack_type"],as_index=False).topk_changed.mean()
    overall=da.groupby(["dataset_id","ratio","gamma"],as_index=False).topk_changed.mean().rename(columns={"topk_changed":"change_rate"})
    du=utility.groupby(["dataset_id","ratio","gamma"],as_index=False)[["test_nll","test_accuracy"]].mean()
    data=overall.merge(du,on=["dataset_id","ratio","gamma"],validate="one_to_one")
    baseline=du.loc[du.ratio.eq(0),["dataset_id","test_nll","test_accuracy"]].rename(columns={"test_nll":"baseline_nll","test_accuracy":"baseline_accuracy"})
    data=data.merge(baseline,on="dataset_id",validate="many_to_one")
    data["delta_nll"]=data.test_nll-data.baseline_nll; data["accuracy_drop"]=data.baseline_accuracy-data.test_accuracy
    write_csv(root/"dataset_level_curve.csv",data)
    rows=[]
    for index,((ratio,gamma),group) in enumerate(data.groupby(["ratio","gamma"],sort=True)):
        row={"ratio":ratio,"gamma":gamma,"datasets":len(group)}
        for offset,metric in enumerate(["change_rate","delta_nll","accuracy_drop","test_nll","test_accuracy"]):
            mean,low,high=bootstrap(group[metric].to_numpy(float),args.bootstrap_seed+index*10+offset,args.bootstrap_repetitions)
            row[f"{metric}_mean"]=mean; row[f"{metric}_ci_lower"]=low; row[f"{metric}_ci_upper"]=high
        rows.append(row)
    summary=pd.DataFrame(rows).sort_values("ratio"); write_csv(root/"gamma_curve_summary.csv",summary)
    print(summary.to_string(index=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=DEFAULT_CONFIG); parser.add_argument("--datasets",nargs="+",default=DEFAULT_DATASETS)
    parser.add_argument("--results-root",type=Path,default=OUTPUT); parser.add_argument("--gamma-path",type=Path,default=DEFAULT_GAMMA_PATH)
    parser.add_argument("--ratios",nargs="+",type=float,default=RATIOS); parser.add_argument("--seeds",nargs="+",type=int,default=[42,2023,3407,7919,15401])
    parser.add_argument("--gamma-workers",type=int,default=2); parser.add_argument("--condition-workers",type=int,default=6)
    parser.add_argument("--bootstrap-repetitions",type=int,default=10000); parser.add_argument("--bootstrap-seed",type=int,default=20260917)
    parser.add_argument("--overwrite",action="store_true"); parser.add_argument("--summarize-only",action="store_true"); parser.add_argument("--no-aggregate",action="store_true"); args=parser.parse_args()
    args.results_root=args.results_root.resolve(); args.results_root.mkdir(parents=True,exist_ok=True)
    if args.summarize_only: aggregate(args.results_root,args); return
    args.gamma_star=float(json.loads(args.gamma_path.resolve().read_text(encoding="utf-8"))["gamma"])
    definitions=_load_datasets(args.config.resolve())
    for dataset_id in args.datasets: run_dataset(dataset_id,definitions[dataset_id],args)
    if not args.no_aggregate: aggregate(args.results_root,args)


if __name__=="__main__": main()
