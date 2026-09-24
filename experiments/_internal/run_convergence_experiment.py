#!/usr/bin/env python3
"""Run Experiment 6: final fit, convergence, and initialization stability."""

from __future__ import annotations

import argparse
from collections import Counter
import itertools
import json
import math
import os
from multiprocessing import Pool
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Dict, Iterable, List, Sequence

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from convergence_optimizers import (  # noqa: E402
    update_am_elo_traced,
    update_m_elo_traced,
    update_traditional_elo,
)
from elo_processor import DataProcessor, EloProcessor, EloWrapper  # noqa: E402
from models import Metric, QuerySet  # noqa: E402


METHODS: Dict[str, Dict[str, Any]] = {
    'traditional_elo': {
        'label': 'Traditional ELO',
        'model': 'traditional_elo',
        'shared_rater': False,
        'group': 'baseline',
        'trajectory': False,
    },
    'crowd_bt': {
        'label': 'Crowd-BT',
        'model': 'google_elo',
        'shared_rater': False,
        'group': 'baseline',
        'trajectory': False,
    },
    'bayes_bt': {
        'label': 'Bayes-BT',
        'model': 'bayesian_elo',
        'shared_rater': False,
        'group': 'baseline',
        'trajectory': True,
    },
    'bbq': {
        'label': 'BBQ',
        'model': 'bayesian_elo_noise',
        'shared_rater': False,
        'group': 'baseline',
        'trajectory': True,
    },
    'm_elo': {
        'label': 'm-ELO',
        'model': 'm_elo',
        'shared_rater': False,
        'group': 'baseline',
        'trajectory': True,
    },
    'am_elo': {
        'label': 'am-ELO',
        'model': 'am_elo',
        'shared_rater': False,
        'group': 'baseline',
        'trajectory': True,
    },
    'correctness_downweight': {
        'label': 'Correctness Downweight',
        'model': 'bayesian_elo_correctness',
        'shared_rater': False,
        'group': 'ours',
        'trajectory': True,
    },
    'correctness_reverse': {
        'label': 'Correctness Reverse',
        'model': 'bayesian_elo_correctness_reverse',
        'shared_rater': False,
        'group': 'ours',
        'trajectory': True,
    },
    'adaptive_clip': {
        'label': 'Adaptive Clip',
        'model': 'bayesian_elo_adaptive_clip',
        'shared_rater': False,
        'group': 'ours',
        'trajectory': True,
    },
    'adaptive_flip': {
        'label': 'Adaptive Flip',
        'model': 'bayesian_elo_adaptive_flip',
        'shared_rater': False,
        'group': 'ours',
        'trajectory': True,
    },
    'rank_balance_gamma0': {
        'label': 'RankBalance (gamma=0)',
        'model': 'rank_balance_ablation',
        'shared_rater': False,
        'group': 'ours',
        'trajectory': False,
        'rank_balance_config': {
            'gamma': 0.0,
            'regularizer': 'full',
            'lambda_theta': 1.0,
            'lambda_u': 1.0,
            'mu': 2.0,
            'max_iter': 1000,
            'tol': 1e-8,
        },
    },
    'rank_balance_tuned_delete_only': {
        'label': 'RankBalance (Delete-only, tuned)',
        'model': 'rank_balance_ablation',
        'shared_rater': False,
        'group': 'ours',
        'trajectory': False,
        'rank_balance_config': {
            'gamma': 0.0,
            'regularizer': 'delete_only',
            'lambda_theta': 1.0,
            'lambda_u': 1.0,
            'mu': 2.0,
            'max_iter': 1000,
            'tol': 1e-8,
        },
    },
    'rank_balance_tuned_full': {
        'label': 'RankBalance (Full, tuned)',
        'model': 'rank_balance_ablation',
        'shared_rater': False,
        'group': 'ours',
        'trajectory': False,
        'rank_balance_config': {
            'gamma': 0.0,
            'regularizer': 'full',
            'lambda_theta': 1.0,
            'lambda_u': 1.0,
            'mu': 2.0,
            'max_iter': 1000,
            'tol': 1e-8,
        },
    },
    'rank_balance_ability_only': {
        'label': 'RankBalance (Ability-only)',
        'model': 'rank_balance_ablation',
        'shared_rater': False,
        'group': 'ours',
        'trajectory': False,
        'rank_balance_config': {
            'gamma': 0.0,
            'regularizer': 'ability_only',
            'lambda_theta': 1.0,
            'lambda_u': 1.0,
            'mu': 2.0,
            'max_iter': 1000,
            'tol': 1e-8,
        },
    },
    'rank_balance_reliability_only': {
        'label': 'RankBalance (Reliability-only)',
        'model': 'rank_balance_ablation',
        'shared_rater': False,
        'group': 'ours',
        'trajectory': False,
        'rank_balance_config': {
            'gamma': 0.0,
            'regularizer': 'reliability_only',
            'lambda_theta': 1.0,
            'lambda_u': 1.0,
            'mu': 2.0,
            'max_iter': 1000,
            'tol': 1e-8,
        },
    },
    'rank_balance_flip_only': {
        'label': 'RankBalance (Flip-only)',
        'model': 'rank_balance_ablation',
        'shared_rater': False,
        'group': 'ours',
        'trajectory': False,
        'rank_balance_config': {
            'gamma': 0.0,
            'regularizer': 'flip_only',
            'lambda_theta': 1.0,
            'lambda_u': 1.0,
            'mu': 2.0,
            'max_iter': 1000,
            'tol': 1e-8,
        },
    },
    'correctness_downweight_shared': {
        'label': 'Correctness Downweight (shared rater)',
        'model': 'bayesian_elo_correctness',
        'shared_rater': True,
        'group': 'shared',
        'trajectory': True,
    },
    'correctness_reverse_shared': {
        'label': 'Correctness Reverse (shared rater)',
        'model': 'bayesian_elo_correctness_reverse',
        'shared_rater': True,
        'group': 'shared',
        'trajectory': True,
    },
    'adaptive_clip_shared': {
        'label': 'Adaptive Clip (shared rater)',
        'model': 'bayesian_elo_adaptive_clip',
        'shared_rater': True,
        'group': 'shared',
        'trajectory': True,
    },
    'adaptive_flip_shared': {
        'label': 'Adaptive Flip (shared rater)',
        'model': 'bayesian_elo_adaptive_flip',
        'shared_rater': True,
        'group': 'shared',
        'trajectory': True,
    },
}


_WORKER: Dict[str, Any] = {}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
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


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True),
        encoding='utf-8',
    )
    temporary.replace(path)


def _load_config(path: Path) -> Dict[str, Any]:
    config = json.loads(path.read_text(encoding='utf-8'))
    defaults = config.get('defaults', {})
    datasets = {}
    for dataset_id, item in config.get('datasets', {}).items():
        merged = dict(defaults)
        merged.update(item)
        if merged.get('enabled', True):
            datasets[dataset_id] = merged
    config['datasets'] = datasets
    return config


def _select(
    tokens: Sequence[str],
    definitions: Dict[str, Dict[str, Any]],
    groups: Iterable[str] = (),
) -> List[str]:
    if not tokens or 'all' in tokens:
        return list(definitions)
    groups = set(groups)
    selected: List[str] = []
    for token in tokens:
        if token in groups:
            selected.extend(
                key for key, value in definitions.items()
                if value.get('group') == token
            )
        elif token in definitions:
            selected.append(token)
        else:
            valid = sorted(set(definitions) | groups | {'all'})
            raise ValueError(f'Unknown selection {token!r}; choose from {valid}')
    return list(dict.fromkeys(selected))


def _truthy(series: pd.Series) -> pd.Series:
    return series.notna() & series.astype(str).str.lower().isin(
        {'true', '1', 'yes'}
    )


def _prepare_dataset(config: Dict[str, Any]) -> tuple[pd.DataFrame, List[str]]:
    path = Path(config['csv'])
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    processor = DataProcessor(str(path))
    if not processor.load_data():
        raise RuntimeError(f'Could not load {path}')
    frame = processor.df.copy()

    if 'isGolden' in frame:
        frame = frame.loc[~_truthy(frame['isGolden'])].copy()
    if 'wasDiscarded' in frame:
        frame = frame.loc[~_truthy(frame['wasDiscarded'])].copy()
    frame = frame.loc[
        frame['methodA'].notna()
        & frame['methodB'].notna()
        & frame['answerValue'].notna()
        & frame['methodA'].astype(str).ne('N/A')
        & frame['methodB'].astype(str).ne('N/A')
    ].copy()

    frame['answerer'] = frame['answerer'].astype('string').fillna('__global_rater__').astype(str)
    minimum = int(config.get('min_annotations_per_rater', 1))
    counts = frame['answerer'].value_counts()
    valid_users = counts[counts >= minimum].index.astype(str).tolist()
    frame = frame.loc[frame['answerer'].isin(valid_users)].reset_index(drop=True)
    if frame.empty:
        raise RuntimeError(
            f'No rows remain after min_annotations_per_rater={minimum}'
        )
    return frame, valid_users


def _sessions(
    frame: pd.DataFrame,
    valid_users: Sequence[str],
) -> QuerySet:
    processor = EloProcessor(frame, list(valid_users))
    converted = processor._convert_google_elo_to_clic2024(frame)
    sessions = EloWrapper().convert_to_elo_format(converted, list(valid_users))
    return QuerySet(sessions)


def _fit_status(metric: Metric) -> str:
    for key in ('m_elo_optimization', 'am_elo_optimization'):
        optimization = metric.state.get(key)
        if optimization is not None:
            return (
                'converged'
                if optimization.get('success')
                else 'optimizer_warning'
            )
    if 'converged' in metric.state:
        return (
            'converged'
            if metric.state['converged']
            else 'iteration_limit'
        )
    return 'completed'


def _target(value: Any) -> float:
    normalized = str(value).strip().lower()
    if normalized in {'a', 'model_a', 'left', '1'}:
        return 1.0
    if normalized in {'b', 'model_b', 'right', '0'}:
        return 0.0
    if normalized in {'tie', 'draw', 'equal', '0.5'}:
        return 0.5
    return math.nan


def _ranking_nll(frame: pd.DataFrame, ranking: pd.DataFrame) -> float:
    scores = ranking.set_index('Method')['ELO Score'].astype(float)
    work = frame.loc[
        frame['methodA'].astype(str).isin(scores.index)
        & frame['methodB'].astype(str).isin(scores.index)
    ].copy()
    targets = work['answerValue'].map(_target).to_numpy(float)
    valid = np.isfinite(targets)
    if not np.any(valid):
        return math.nan
    score_a = work.loc[valid, 'methodA'].astype(str).map(scores).to_numpy(float)
    score_b = work.loc[valid, 'methodB'].astype(str).map(scores).to_numpy(float)
    logits = np.log(10.0) * (score_a - score_b) / 400.0
    loss = np.logaddexp(0.0, logits) - targets[valid] * logits
    return float(np.mean(loss))


def _metric_ranking(metric: Metric) -> pd.DataFrame:
    ranking = EloWrapper.get_elo_model_df(metric)
    if ranking.empty:
        raise RuntimeError('Method returned an empty ranking')
    return ranking.sort_values('ELO Score', ascending=False).reset_index(drop=True)


def _trace_config(seed: int, args: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'enabled': True,
        'random_initialization': True,
        'seed': int(seed),
        'initial_elo_std': float(args['initial_elo_std']),
        'initial_quality_std': float(args['initial_quality_std']),
        'initial_ability_std': float(args['initial_ability_std']),
        'record_every': int(args['record_every']),
    }


def _fit_task(task: Dict[str, Any]) -> Dict[str, Any]:
    frame = _WORKER['frame']
    valid_users = _WORKER['valid_users']
    output_root = Path(_WORKER['output_root'])
    dataset_id = _WORKER['dataset_id']
    dataset_label = _WORKER['dataset_label']
    method_id = task['method_id']
    method = METHODS[method_id]
    seed = int(task['seed'])
    run_dir = output_root / 'raw' / dataset_id / method_id / f'seed_{seed}'
    run_path = run_dir / 'run.json'

    if run_path.exists() and not task['overwrite']:
        existing = json.loads(run_path.read_text(encoding='utf-8'))
        if existing.get('status') == 'success':
            return {'status': 'skipped', 'method_id': method_id, 'seed': seed}

    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        config = _trace_config(seed, task)
        model = method['model']
        processor = None

        if model in {'traditional_elo', 'm_elo', 'am_elo'}:
            metric = Metric()
            metric.state['_optimization_trace_config'] = config
            sessions = _sessions(frame, valid_users)
            if model == 'traditional_elo':
                elapsed = update_traditional_elo(
                    metric,
                    sessions,
                    shuffles=int(task['traditional_shuffles']),
                    k_factor=float(task['traditional_k']),
                )
            elif model == 'm_elo':
                elapsed = update_m_elo_traced(metric, sessions)
            else:
                elapsed = update_am_elo_traced(metric, sessions)
            ranking = _metric_ranking(metric)
        else:
            processor = EloProcessor(frame, valid_users)
            ranking, elapsed = processor.process(
                frame,
                valid_users,
                model=model,
                shared_rater=method['shared_rater'],
                device='cpu',
                optimization_trace_config=(
                    config if method['trajectory'] else None
                ),
            )
            metric = processor.metric
            if ranking.empty:
                raise RuntimeError('Method returned an empty ranking')
            ranking = ranking.sort_values(
                'ELO Score', ascending=False
            ).reset_index(drop=True)

        ranking_rows = ranking[['Method', 'ELO Score']].copy()
        ranking_rows.insert(0, 'rank', np.arange(1, len(ranking_rows) + 1))
        ranking_rows.to_csv(run_dir / 'final_ranking.csv', index=False)

        trace = metric.state.get('optimization_trace', [])
        trace_rows = []
        score_rows = []
        for snapshot in trace:
            trace_rows.append({
                'iteration': snapshot['iteration'],
                'model_only_nll': snapshot.get('model_only_nll'),
                'objective': snapshot.get('objective'),
                'max_elo_change': snapshot.get('max_elo_change'),
                'converged': snapshot.get('converged', False),
                'ranking_json': json.dumps(snapshot.get('ranking', [])),
            })
            rank_map = {
                model_name: rank
                for rank, model_name in enumerate(
                    snapshot.get('ranking', []), start=1
                )
            }
            for model_name, score in snapshot.get('scores', {}).items():
                score_rows.append({
                    'iteration': snapshot['iteration'],
                    'model': model_name,
                    'elo_score': score,
                    'rank': rank_map.get(model_name),
                })
        trace_frame = pd.DataFrame(trace_rows)
        score_frame = pd.DataFrame(score_rows)
        if not trace_frame.empty:
            trace_frame = trace_frame.drop_duplicates('iteration', keep='last')
        if not score_frame.empty:
            score_frame = score_frame.drop_duplicates(
                ['iteration', 'model'], keep='last'
            )
        trace_frame.to_csv(run_dir / 'trajectory.csv', index=False)
        score_frame.to_csv(
            run_dir / 'trajectory_scores.csv', index=False
        )

        traditional_runs = metric.state.get('traditional_elo_runs', [])
        if traditional_runs:
            pd.DataFrame([
                {
                    'shuffle': item['shuffle'],
                    'model_only_nll': item['model_only_nll'],
                    'ranking_json': json.dumps(item['ranking']),
                    'scores_json': json.dumps(item['scores'], sort_keys=True),
                }
                for item in traditional_runs
            ]).to_csv(run_dir / 'traditional_order_runs.csv', index=False)

        final_objective = (
            trace[-1].get('objective') if trace else math.nan
        )
        record = {
            'dataset_id': dataset_id,
            'dataset': dataset_label,
            'method_id': method_id,
            'method': method['label'],
            'model': model,
            'shared_rater': method['shared_rater'],
            'seed': seed,
            'status': 'success',
            'fit_status': _fit_status(metric),
            'converged': bool(metric.state.get('converged', True)),
            'iterations': int(metric.state.get('iterations', 0)),
            'time_seconds': float(elapsed),
            'wall_time_seconds': float(time.time() - started),
            'model_only_nll': _ranking_nll(frame, ranking),
            'objective': final_objective,
            'rows': int(len(frame)),
            'raters': int(len(valid_users)),
            'models': int(len(ranking)),
            'trace_points': int(len(trace)),
            'backend': metric.state.get('computation_backend', 'cpu'),
            'error_type': '',
            'error_message': '',
        }
        _write_json(run_path, record)
        return {'status': 'success', 'method_id': method_id, 'seed': seed}
    except Exception as error:
        record = {
            'dataset_id': dataset_id,
            'dataset': dataset_label,
            'method_id': method_id,
            'method': method['label'],
            'model': method['model'],
            'shared_rater': method['shared_rater'],
            'seed': seed,
            'status': 'error',
            'fit_status': 'error',
            'converged': False,
            'iterations': 0,
            'time_seconds': math.nan,
            'wall_time_seconds': float(time.time() - started),
            'model_only_nll': math.nan,
            'objective': math.nan,
            'rows': int(len(frame)),
            'raters': int(len(valid_users)),
            'models': 0,
            'trace_points': 0,
            'backend': 'cpu',
            'error_type': type(error).__name__,
            'error_message': str(error),
        }
        _write_json(run_path, record)
        (run_dir / 'traceback.txt').write_text(
            traceback.format_exc(), encoding='utf-8'
        )
        return {'status': 'error', 'method_id': method_id, 'seed': seed}


def _init_worker(
    frame: pd.DataFrame,
    valid_users: List[str],
    output_root: str,
    dataset_id: str,
    dataset_label: str,
) -> None:
    _WORKER.update({
        'frame': frame,
        'valid_users': valid_users,
        'output_root': output_root,
        'dataset_id': dataset_id,
        'dataset_label': dataset_label,
    })


def _read_outputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    runs = []
    traces = []
    rankings = []
    for run_path in sorted((root / 'raw').glob('*/*/seed_*/run.json')):
        record = json.loads(run_path.read_text(encoding='utf-8'))
        runs.append(record)
        if record.get('status') != 'success':
            continue
        run_dir = run_path.parent
        keys = {
            key: record[key]
            for key in ('dataset_id', 'dataset', 'method_id', 'method', 'seed')
        }
        trace_path = run_dir / 'trajectory.csv'
        if trace_path.exists() and trace_path.stat().st_size:
            try:
                trace = pd.read_csv(trace_path)
            except pd.errors.EmptyDataError:
                trace = pd.DataFrame()
            if not trace.empty:
                for key, value in keys.items():
                    trace[key] = value
                traces.append(trace)
        ranking_path = run_dir / 'final_ranking.csv'
        if ranking_path.exists():
            ranking = pd.read_csv(ranking_path)
            for key, value in keys.items():
                ranking[key] = value
            rankings.append(ranking)
    return (
        pd.DataFrame(runs),
        pd.concat(traces, ignore_index=True) if traces else pd.DataFrame(),
        pd.concat(rankings, ignore_index=True) if rankings else pd.DataFrame(),
    )


def _pairwise_consistency(rankings: List[List[str]]) -> tuple[float, float]:
    if len(rankings) < 2:
        return math.nan, math.nan
    exact = []
    kendall = []
    for first, second in itertools.combinations(rankings, 2):
        exact.append(float(first == second))
        common = sorted(set(first) & set(second))
        if len(common) < 2:
            continue
        rank_first = {model: rank for rank, model in enumerate(first)}
        rank_second = {model: rank for rank, model in enumerate(second)}
        tau = kendalltau(
            [rank_first[model] for model in common],
            [rank_second[model] for model in common],
        ).statistic
        if np.isfinite(tau):
            kendall.append((float(tau) + 1.0) / 2.0)
    return (
        float(np.mean(exact)) if exact else math.nan,
        float(np.mean(kendall)) if kendall else math.nan,
    )


def _finite_mean(values: Sequence[Any]) -> float:
    numeric = pd.to_numeric(pd.Series(values), errors='coerce').to_numpy(float)
    finite = numeric[np.isfinite(numeric)]
    return float(finite.mean()) if len(finite) else math.nan



def _consistency_table(traces: pd.DataFrame) -> pd.DataFrame:
    if traces.empty:
        return pd.DataFrame()
    rows = []
    for (dataset_id, method_id), group in traces.groupby(
        ['dataset_id', 'method_id']
    ):
        seed_frames = {
            int(seed): frame.sort_values('iteration').reset_index(drop=True)
            for seed, frame in group.groupby('seed')
        }
        if len(seed_frames) < 2:
            continue
        iterations = sorted(group['iteration'].astype(int).unique())
        for iteration in iterations:
            snapshots = []
            nll = []
            objective = []
            max_change = []
            for seed, frame in seed_frames.items():
                available = frame.loc[frame['iteration'] <= iteration]
                if available.empty:
                    continue
                snapshot = available.iloc[-1]
                snapshots.append(json.loads(snapshot['ranking_json']))
                nll.append(snapshot['model_only_nll'])
                objective.append(snapshot['objective'])
                max_change.append(snapshot['max_elo_change'])
            exact, kendall = _pairwise_consistency(snapshots)
            rows.append({
                'dataset_id': dataset_id,
                'dataset': group['dataset'].iloc[0],
                'method_id': method_id,
                'method': group['method'].iloc[0],
                'iteration': int(iteration),
                'initializations': int(len(snapshots)),
                'exact_ranking_agreement': exact,
                'pairwise_kendall_consistency': kendall,
                'mean_model_only_nll': _finite_mean(nll),
                'mean_objective': _finite_mean(objective),
                'mean_max_elo_change': _finite_mean(max_change),
            })
    return pd.DataFrame(rows)


def _mean_rankings(rankings: pd.DataFrame) -> pd.DataFrame:
    if rankings.empty:
        return pd.DataFrame()
    grouped = (
        rankings.groupby(
            ['dataset_id', 'dataset', 'method_id', 'method', 'Method'],
            as_index=False,
        )
        .agg(
            mean_elo=('ELO Score', 'mean'),
            std_elo=('ELO Score', 'std'),
            mean_rank=('rank', 'mean'),
            std_rank=('rank', 'std'),
            runs=('seed', 'nunique'),
        )
    )
    grouped['rank_of_mean_elo'] = (
        grouped.groupby(['dataset_id', 'method_id'])['mean_elo']
        .rank(method='min', ascending=False)
        .astype(int)
    )
    return grouped.sort_values(
        ['dataset_id', 'method_id', 'rank_of_mean_elo']
    )


def _final_summary(
    runs: pd.DataFrame,
    consistency: pd.DataFrame,
) -> pd.DataFrame:
    if runs.empty:
        return pd.DataFrame()
    successful = runs.loc[runs['status'].eq('success')].copy()
    if successful.empty:
        return pd.DataFrame()
    summary = (
        successful.groupby(
            ['dataset_id', 'dataset', 'method_id', 'method'],
            as_index=False,
        )
        .agg(
            runs=('seed', 'nunique'),
            convergence_rate=('converged', 'mean'),
            mean_iterations=('iterations', 'mean'),
            mean_time_seconds=('time_seconds', 'mean'),
            std_time_seconds=('time_seconds', 'std'),
            mean_model_only_nll=('model_only_nll', 'mean'),
            std_model_only_nll=('model_only_nll', 'std'),
            mean_objective=('objective', 'mean'),
            trace_points=('trace_points', 'sum'),
        )
    )
    if not consistency.empty:
        final_consistency = (
            consistency.sort_values('iteration')
            .groupby(['dataset_id', 'method_id'], as_index=False)
            .tail(1)[[
                'dataset_id',
                'method_id',
                'exact_ranking_agreement',
                'pairwise_kendall_consistency',
            ]]
        )
        summary = summary.merge(
            final_consistency,
            on=['dataset_id', 'method_id'],
            how='left',
        )
    return summary.sort_values(['dataset_id', 'mean_model_only_nll'])


def _traditional_order_tables(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_frames = []
    summaries = []
    for path in sorted(
        (root / 'raw').glob('*/*/seed_*/traditional_order_runs.csv')
    ):
        if not path.exists() or path.stat().st_size <= 1:
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        run_record = json.loads(
            (path.parent / 'run.json').read_text(encoding='utf-8')
        )
        frame.insert(0, 'dataset_id', run_record['dataset_id'])
        frame.insert(1, 'dataset', run_record['dataset'])
        frame.insert(2, 'method_id', run_record['method_id'])
        frame.insert(3, 'method', run_record['method'])
        frame.insert(4, 'seed', run_record['seed'])
        detail_frames.append(frame)

        rankings = [
            json.loads(value) for value in frame['ranking_json'].tolist()
        ]
        counts = Counter(tuple(ranking) for ranking in rankings)
        pair_count = len(rankings) * (len(rankings) - 1) // 2
        exact_pairs = sum(
            count * (count - 1) // 2 for count in counts.values()
        )
        exact = exact_pairs / pair_count if pair_count else math.nan

        all_pairs = list(itertools.combinations(range(len(rankings)), 2))
        max_pairs = 10000
        if len(all_pairs) > max_pairs:
            rng = np.random.default_rng(104729)
            chosen = rng.choice(len(all_pairs), size=max_pairs, replace=False)
            evaluated_pairs = [all_pairs[index] for index in chosen]
        else:
            evaluated_pairs = all_pairs
        kendall_values = []
        for first_index, second_index in evaluated_pairs:
            first = rankings[first_index]
            second = rankings[second_index]
            common = sorted(set(first) & set(second))
            rank_first = {
                model: rank for rank, model in enumerate(first)
            }
            rank_second = {
                model: rank for rank, model in enumerate(second)
            }
            tau = kendalltau(
                [rank_first[model] for model in common],
                [rank_second[model] for model in common],
            ).statistic
            if np.isfinite(tau):
                kendall_values.append((float(tau) + 1.0) / 2.0)

        summaries.append({
            'dataset_id': run_record['dataset_id'],
            'dataset': run_record['dataset'],
            'method_id': run_record['method_id'],
            'method': run_record['method'],
            'order_runs': len(rankings),
            'unique_rankings': len(counts),
            'exact_ranking_agreement': exact,
            'pairwise_kendall_consistency': _finite_mean(kendall_values),
            'kendall_pairs_evaluated': len(kendall_values),
            'mean_model_only_nll': _finite_mean(
                frame['model_only_nll'].tolist()
            ),
        })
    details = (
        pd.concat(detail_frames, ignore_index=True)
        if detail_frames else pd.DataFrame()
    )
    return details, pd.DataFrame(summaries)


def _summarize(root: Path) -> None:

    runs, traces, rankings = _read_outputs(root)
    consistency = _consistency_table(traces)
    mean_rankings = _mean_rankings(rankings)
    summary = _final_summary(runs, consistency)
    traditional_runs, traditional_summary = _traditional_order_tables(root)

    runs.to_csv(root / 'runs.csv', index=False)
    traces.to_csv(root / 'trajectories.csv', index=False)
    rankings.to_csv(root / 'final_rankings.csv', index=False)
    consistency.to_csv(root / 'initialization_consistency.csv', index=False)
    mean_rankings.to_csv(root / 'mean_final_rankings.csv', index=False)
    if not traditional_summary.empty:
        order_metrics = traditional_summary[[
            'dataset_id',
            'method_id',
            'exact_ranking_agreement',
            'pairwise_kendall_consistency',
        ]].rename(columns={
            'exact_ranking_agreement': 'order_exact',
            'pairwise_kendall_consistency': 'order_kendall',
        })
        summary = summary.merge(
            order_metrics, on=['dataset_id', 'method_id'], how='left'
        )
        for target, source in (
            ('exact_ranking_agreement', 'order_exact'),
            ('pairwise_kendall_consistency', 'order_kendall'),
        ):
            if target not in summary:
                summary[target] = summary[source]
            else:
                summary[target] = summary[target].fillna(summary[source])
        summary = summary.drop(columns=['order_exact', 'order_kendall'])
    summary.to_csv(root / 'final_summary.csv', index=False)
    traditional_runs.to_csv(root / 'traditional_order_runs.csv', index=False)
    traditional_summary.to_csv(root / 'traditional_order_summary.csv', index=False)


def _tasks(
    method_ids: Sequence[str],
    seeds: Sequence[int],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    tasks = []
    for method_id in method_ids:
        method_seeds = (
            seeds
            if METHODS[method_id]['trajectory']
            else seeds[:1]
        )
        for seed in method_seeds:
            tasks.append({
                'method_id': method_id,
                'seed': int(seed),
                'overwrite': bool(args.overwrite),
                'traditional_shuffles': int(args.traditional_shuffles),
                'traditional_k': float(args.traditional_k),
                'initial_elo_std': float(args.initial_elo_std),
                'initial_quality_std': float(args.initial_quality_std),
                'initial_ability_std': float(args.initial_ability_std),
                'record_every': int(args.record_every),
            })
    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config',
        type=Path,
        default=PROJECT_ROOT / 'configs' / 'experiment_6_convergence.json',
    )
    parser.add_argument('--datasets', nargs='+', default=['all'])
    parser.add_argument('--methods', nargs='+', default=['all'])
    parser.add_argument(
        '--results-root',
        type=Path,
        default=PROJECT_ROOT / 'experiment_results' / 'experiment_6_convergence',
    )
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--traditional-shuffles', type=int, default=1000)
    parser.add_argument('--traditional-k', type=float, default=4.0)
    parser.add_argument('--initial-elo-std', type=float, default=100.0)
    parser.add_argument('--initial-quality-std', type=float, default=0.05)
    parser.add_argument('--initial-ability-std', type=float, default=0.1)
    parser.add_argument('--record-every', type=int, default=1)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--summarize-only', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.results_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        _summarize(root)
        print(f'Updated tables in {root}')
        return

    config = _load_config(args.config.resolve())
    dataset_ids = _select(args.datasets, config['datasets'])
    method_ids = _select(
        args.methods, METHODS, groups={'baseline', 'ours', 'shared'}
    )
    seeds = [int(seed) for seed in config['seeds']]
    _write_json(root / 'resolved_config.json', {
        'config': config,
        'selected_datasets': dataset_ids,
        'selected_methods': method_ids,
        'methods': METHODS,
        'seeds': seeds,
        'arguments': vars(args),
    })

    for dataset_id in dataset_ids:
        dataset_config = config['datasets'][dataset_id]
        frame, valid_users = _prepare_dataset(dataset_config)
        dataset_label = dataset_config.get('label', dataset_id)
        tasks = _tasks(method_ids, seeds, args)
        print(
            f'Dataset {dataset_label}: {len(frame)} rows, '
            f'{len(valid_users)} raters, {len(tasks)} fits'
        )

        worker_args = (
            frame,
            valid_users,
            str(root),
            dataset_id,
            dataset_label,
        )
        workers = max(1, min(int(args.workers), len(tasks)))
        if workers == 1:
            _init_worker(*worker_args)
            iterator = map(_fit_task, tasks)
            for _ in tqdm(iterator, total=len(tasks), desc=dataset_id):
                pass
        else:
            with Pool(
                processes=workers,
                initializer=_init_worker,
                initargs=worker_args,
            ) as pool:
                iterator = pool.imap_unordered(_fit_task, tasks, chunksize=1)
                for _ in tqdm(iterator, total=len(tasks), desc=dataset_id):
                    pass
        _summarize(root)
        print(f'Updated {root / "final_summary.csv"}')


if __name__ == '__main__':
    main()
