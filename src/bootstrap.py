from typing import Dict, List, Tuple, Optional
import pandas as pd
import os
import random
import time
import argparse
from multiprocessing import Pool
import numpy as np
from scipy.stats import spearmanr, kendalltau
from tqdm import tqdm

# 反复抽样用户 → 跑 ELO 排名 → 和全量结果比较稳定性 → 保存统计结果。

from elo_processor import DataProcessor, EloProcessor, GPU_BAYESIAN_MODELS
from torch_bayesian_backend import resolve_device

# run_bootstrap_wrapper 是 multiprocessing.Pool 使用的小包装函数。
# Pool 只能方便地传一个参数，所以这里把 tuple 拆开再调用 run_bootstrap。
def run_bootstrap_wrapper(args_tuple):
    (
        df,
        valid_users,
        n_users,
        n_comp_per_user,
        model,
        shared_rater,
        device,
        seed,
    ) = args_tuple
    return run_bootstrap(
        df,
        valid_users,
        n_users,
        n_comp_per_user,
        model,
        shared_rater,
        device,
        seed,
    )


def build_bootstrap_sample(
    df: pd.DataFrame,
    valid_users: List[str],
    n_users: Optional[int] = None,
    n_comp_per_user: Optional[int] = None,
    seed: Optional[int] = None,
    preserve_golden: bool = False,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Create one rater-cluster bootstrap sample.

    Raters are sampled with replacement. Each sampled copy receives a fresh
    temporary ID so rater-aware models treat duplicate draws as independent
    bootstrap clusters. Comparisons within a sampled rater are kept in full
    unless ``n_comp_per_user`` requests a without-replacement subsample.

    When ``preserve_golden`` is true, the comparison limit is applied only to
    ordinary preference rows. All Golden calibration rows belonging to the
    sampled rater are retained. This matches the IHQ protocol: Google ELO
    uses Golden rows to estimate rater reliability, while the Bayesian
    converters ignore those rows internally.
    """
    if not valid_users:
        raise ValueError("Cannot bootstrap an empty rater list")
    if n_users is not None and n_users <= 0:
        raise ValueError("n_users must be positive")
    if n_comp_per_user is not None and n_comp_per_user <= 0:
        raise ValueError("n_comp_per_user must be positive")

    rng = random.Random(seed)
    sample_size = len(valid_users) if n_users is None else n_users
    sampled_users = rng.choices(list(valid_users), k=sample_size)
    user_frames = {
        user: group for user, group in df.groupby('answerer', sort=False)
    }
    bootstrap_parts = []

    for draw_index, user in enumerate(sampled_users):
        if user not in user_frames:
            raise KeyError(f"Rater {user!r} is missing from the data frame")
        df_subset = user_frames[user]
        if preserve_golden and "isGolden" in df_subset.columns:
            golden_mask = (
                df_subset["isGolden"].notna()
                & df_subset["isGolden"]
                .astype(str)
                .str.strip()
                .str.lower()
                .eq("true")
            )
            golden_rows = df_subset.loc[golden_mask]
            comparison_rows = df_subset.loc[~golden_mask]
            if (
                n_comp_per_user is not None
                and len(comparison_rows) > n_comp_per_user
            ):
                positions = rng.sample(
                    range(len(comparison_rows)), n_comp_per_user
                )
                comparison_rows = comparison_rows.iloc[positions]
            df_subset = pd.concat(
                [comparison_rows, golden_rows], axis=0
            ).sort_index()
        elif (
            n_comp_per_user is not None
            and len(df_subset) > n_comp_per_user
        ):
            positions = rng.sample(range(len(df_subset)), n_comp_per_user)
            df_subset = df_subset.iloc[positions]
        df_subset = df_subset.copy()
        df_subset['answerer'] = f"user_{draw_index}"
        bootstrap_parts.append(df_subset)

    df_bootstrap = pd.concat(bootstrap_parts, ignore_index=True)
    return df_bootstrap, df_bootstrap['answerer'].unique()



# run_bootstrap 执行一次 bootstrap 实验：
# 1. 从 valid_users 中有放回抽样若干用户。
# 2. 拼出这次 bootstrap 用的数据表 df_bootstrap。
# 3. 调用 EloProcessor.process 跑指定 model，得到一次排名结果。
def run_bootstrap(
    df: pd.DataFrame,
    valid_users: List[str],
    n_users: int,
    n_comp_per_user: int,
    model: str,
    shared_rater: bool = False,
    device: str = 'auto',
    seed: int = None,
):
    df_bootstrap, valid_users_subset_new = build_bootstrap_sample(
        df=df,
        valid_users=valid_users,
        n_users=n_users,
        n_comp_per_user=n_comp_per_user,
        seed=seed,
    )
    # EloProcessor 会根据 model 选择 baseline、BBQ、Correctness 降权或反向更新方法。
    processor = EloProcessor(df_bootstrap, valid_users_subset_new)
    results, time = processor.process(
        df=df_bootstrap,
        valid_users=valid_users_subset_new,
        model=model,
        shared_rater=shared_rater,
        device=device,
    )

    return results, time

# main 是命令行入口：解析参数、读取数据、跑多次 bootstrap、计算稳定性指标并保存结果。
def main():
    parser = argparse.ArgumentParser(description='Run bootstrap analysis for ELO ratings')
    parser.add_argument('--project_name', type=str, required=True, help='Name of the project')
    parser.add_argument('--csv_name', type=str, required=True, help='Name of the CSV file to process')
    parser.add_argument('--model', type=str, default='google_elo', help='Model to use for analysis')
    parser.add_argument(
        '--shared_rater',
        action='store_true',
        help='Ignore available rater IDs and estimate one shared eta (correctness models only)',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help="Computation device for Bayesian methods: auto, cpu, cuda, or cuda:N",
    )
    parser.add_argument('--n_users', type=int, default=None, help='Number of users to sample (default: all available users)')
    parser.add_argument('--n_comp_per_user', type=int, default=None, help='Number of comparisons per user to sample (default: all available)')
    parser.add_argument('--bootstrap_n', type=int, required=True, help='Number of bootstraps')
    parser.add_argument('--seed', type=int, default=42, help='Base seed for reproducible, paired bootstrap samples')
    parser.add_argument('--main_csv_name', type=str, default=None, help='Name of the CSV file to process for main run')
    parser.add_argument('--results_path', type=str, default=None, help='Custom path to save results (if None, uses projects/{project_name}/results)')
    
    args = parser.parse_args()
    if args.shared_rater and args.model not in {
        'bayesian_elo_correctness',
        'bayesian_elo_correctness_reverse',
        'bayesian_elo_adaptive_clip',
        'bayesian_elo_adaptive_flip',
    }:
        parser.error('--shared_rater is only supported by correctness models')

    if args.n_users is not None:
        print(f"Running bootstrap for {args.project_name} with data {args.csv_name}, model {args.model} and {args.n_users} users")
    else:
        print(f"Running bootstrap for {args.project_name} with data {args.csv_name}, model {args.model} and all available users")
    if args.shared_rater:
        print("Using one shared rater for all comparisons")
    if args.n_comp_per_user is not None:
        print(f"Sampling {args.n_comp_per_user} comparisons per user")
    else:
        print("Sampling all available comparisons per user")
    
    data_path = os.path.join('projects', args.project_name, 'data', args.csv_name)
    
    # 读取 projects/{project_name}/data/{csv_name}，并筛掉比较次数不足的用户。
    data_processor = DataProcessor(data_path)
    if not data_processor.load_data():
        print("Failed to load data")
        return
        
    df = data_processor.df
    valid_users = data_processor.get_valid_users()
    n_users = args.n_users

    # 并行运行 bootstrap_n 次 bootstrap，每次都会重新抽用户并跑一次 ELO 排名。
    actual_device = (
        resolve_device(args.device)
        if args.model in GPU_BAYESIAN_MODELS
        else 'cpu'
    )
    # A single fit already uses the selected GPU. Multiple worker processes
    # would duplicate its tensors and contend for the same device.
    cpu_count = (
        1 if actual_device.startswith('cuda')
        else min(10, os.cpu_count() or 1)
    )
    print(f"Computation backend: {actual_device}")
    print(f"Starting {args.bootstrap_n} bootstrap iterations using {cpu_count} CPU cores...")
    
    with Pool(processes=cpu_count) as pool:
        args_list = [
            (
                df,
                valid_users,
                n_users,
                args.n_comp_per_user,
                args.model,
                args.shared_rater,
                args.device,
                args.seed + bootstrap_index,
            )
            for bootstrap_index in range(args.bootstrap_n)
        ]
        results = list(tqdm(pool.imap(run_bootstrap_wrapper, args_list), 
                           total=args.bootstrap_n, 
                           desc="Bootstrap iterations"))
    successful_runs = [(result, run_time) for result, run_time in results if not result.empty]
    if len(successful_runs) != len(results):
        print(f"Skipping {len(results) - len(successful_runs)} failed bootstrap runs")
    results = [result for result, _ in successful_runs]
    times = [run_time for _, run_time in successful_runs]

    # main run 是参考排名：后面每个 bootstrap 结果都会和它比较稳定性。
    # 如果指定 main_csv_name，就用另一个 CSV 做参考；否则用当前 CSV 的全量数据。
    if args.main_csv_name is not None:
        data_processor_main = DataProcessor(os.path.join('projects', args.project_name, 'data', args.main_csv_name))
        if not data_processor_main.load_data():
            print("Failed to load data")
            return
        df_main = data_processor_main.df
        valid_users_main = data_processor_main.get_valid_users()
    else:
        df_main = df
        valid_users_main = valid_users
    print(len(df_main), len(valid_users_main))
    # 跑一次全量/参考排名，作为 bootstrap 稳定性指标的比较对象。
    processor_main = EloProcessor(df_main, valid_users_main)
    results_main, time_main = processor_main.process(
        df=df_main,
        valid_users=valid_users_main,
        model=args.model,
        shared_rater=args.shared_rater,
        device=args.device,
    )
    print(results_main)
    
    # 计算每次 bootstrap 排名和参考排名的一致性：
    # Top1 agreement：第一名是否相同。
    # Spearman / Kendall：整体排序相关性。
    top1_agreement = []
    spearman_correlation = []
    kendall_tau = []
    for result in tqdm(results, desc="Calculating correlations"):
        if result.empty:
            continue
        top1_agreement.append(result['Method'].iloc[0] == results_main['Method'].iloc[0])
        
        # align methods properly for correlation calculation
        result_sorted = result.sort_values('Method').reset_index(drop=True)
        results_main_sorted = results_main.sort_values('Method').reset_index(drop=True)
        # only keep methods that exist in both
        common_methods = set(result_sorted['Method']) & set(results_main_sorted['Method'])
        result_aligned = result_sorted[result_sorted['Method'].isin(common_methods)].sort_values('Method')
        results_main_aligned = results_main_sorted[results_main_sorted['Method'].isin(common_methods)].sort_values('Method')
        
        if len(result_aligned) > 1:  # need at least 2 points for correlation
            spearman_correlation.append(spearmanr(result_aligned['ELO Score'], results_main_aligned['ELO Score']).correlation)
            kendall_tau.append(kendalltau(result_aligned['ELO Score'], results_main_aligned['ELO Score']).correlation)
        else:
            spearman_correlation.append(np.nan)
            kendall_tau.append(np.nan)

    print(f"Top1 agreement: {np.mean(top1_agreement):.3f}")
    print(f"Spearman correlation: {np.mean(spearman_correlation):.6f}")
    print(f"Kendall's tau: {np.mean(kendall_tau):.6f}")

    # 保存每次 bootstrap 的稳定性指标和耗时。
    results_df = pd.DataFrame({
        'Top1 agreement': top1_agreement,
        'Spearman correlation': spearman_correlation,
        'Kendall\'s tau': kendall_tau,
        'Time': times
    })

    csv_name = args.csv_name.replace('.csv', '')

    # 结果路径格式：results/{project}/{csv}/{model}/{n_users}/{n_comp_per_user}/bootstrap_results.csv
    # 如果传入 results_path，就用它作为根目录。
    if args.results_path is not None:
        results_path = args.results_path
    else:
        results_path = 'results'
    result_model_name = (
        f'{args.model}_shared_rater' if args.shared_rater else args.model
    )
    results_path = os.path.join(
        results_path, args.project_name, csv_name, result_model_name
    )
    
    if args.n_users is not None:
        results_path = os.path.join(results_path, str(args.n_users))
    else:
        results_path = os.path.join(results_path, 'all')
        
    if args.n_comp_per_user is not None:
        results_path = os.path.join(results_path, str(args.n_comp_per_user))
    else:
        results_path = os.path.join(results_path, 'all')
    
    results_path = os.path.join(results_path, 'bootstrap_results.csv')
    
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
