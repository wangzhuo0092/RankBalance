# Correctness BBQ: 为每个 rater 估计一个统一的标签正确率 eta_r。
# P(observed label | pi, eta_r) = eta_r * pi + (1 - eta_r) * (1 - pi)
# 本文件共享一套拟合代码，提供 posterior correctness 和 adaptive soft clipping 两类权重。
import math
import logging
from functools import partial
import scipy.stats
import numpy as np
from models import (
    Metric, QuerySet, ELO_INITIAL_SCORE, ELO_SCALE_FACTOR,
    ELO_MAX_UPDATES, ELO_CONVERGENCE_THRESHOLD,
    GAMMA_PRIOR_SHAPE, GAMMA_PRIOR_RATE,
    skill_to_elo, elo_to_skill, BETA_PRIOR_ALPHA, BETA_PRIOR_BETA,
    resolve_prior_config,
    initialize_optimization_state, optimization_trace_enabled,
    record_optimization_trace,
)
from torch_bayesian_backend import fit_correctness_bbq_cuda, resolve_device
import time

logger = logging.getLogger(__name__)

EPS = 1e-12
GLOBAL_RATER = 'global_rater'
ADAPTIVE_WARMUP_UPDATES = 10
ADAPTIVE_MAX_CONTAMINATION = 0.499
ADAPTIVE_TEMPERATURE = 1.0
ADAPTIVE_DECAY_POWER = 2.0


def _prior_mode(alpha, beta):
    """Beta(alpha, beta) 的众数；如果参数不满足众数条件，则退回均值。"""
    if alpha > 1 and beta > 1:
        return (alpha - 1) / (alpha + beta - 2)
    return alpha / (alpha + beta)


def _normalize_rater(rater):
    """没有标注者 ID 时，把所有比较归到同一个全局标注者。"""
    if rater is None:
        return GLOBAL_RATER
    if isinstance(rater, (float, np.floating)) and np.isnan(rater):
        return GLOBAL_RATER
    if isinstance(rater, str) and not rater.strip():
        return GLOBAL_RATER
    return rater


def _weighted_quantile(values, weights, quantile):
    """按比较次数计算分位数，避免把每个非零 wins 单元等权处理。"""
    values = np.asarray(values, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)

    if not np.any(valid):
        return None

    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]

    quantile = float(np.clip(quantile, 0.0, 1.0))
    target = quantile * np.sum(weights)
    index = np.searchsorted(np.cumsum(weights), target, side='left')
    return values[min(index, len(values) - 1)]


def _compute_correctness_expectations(
    wins, skill_scores, eta_t, reverse_update
):
    """计算统一标签正确率模型下的后验期望。

    对每个 canonical pair (i < j):
    - obs_pos = rater 说 i > j 的次数，即 wins[i, j, r]
    - obs_neg = rater 说 j > i 的次数，即 wins[j, i, r]

    Correctness BBQ 的观测模型是：
        P(obs_pos) = pi_ij * eta_r + (1 - pi_ij) * (1 - eta_r)
        P(obs_neg) = (1 - pi_ij) * eta_r + pi_ij * (1 - eta_r)

    reverse_update=False 时，只保留观测标签正确的权重 w，错误概率 1-w 被丢弃。
    reverse_update=True 时，正确概率 w 正向更新，错误概率 1-w 反向更新。
    """
    n, _, R = wins.shape
    expected_true_wins = np.zeros_like(wins, dtype=float)

    if n < 2 or R == 0:
        return expected_true_wins, None

    i_less_j = np.triu_indices(n, k=1)
    i_idx, j_idx = i_less_j

    obs_pos = wins[i_idx, j_idx, :]  # 观测到 i > j
    obs_neg = wins[j_idx, i_idx, :]  # 观测到 j > i

    pi = skill_scores[i_idx] / (skill_scores[i_idx] + skill_scores[j_idx])
    pi = pi[:, np.newaxis]
    eta = eta_t[np.newaxis, :]

    prob_pos = np.clip(pi * eta + (1 - pi) * (1 - eta), EPS, 1.0)
    prob_neg = np.clip((1 - pi) * eta + pi * (1 - eta), EPS, 1.0)

    # 给定观测结果，计算该标签标对的后验概率 w。
    correct_pos = pi * eta / prob_pos
    correct_neg = (1 - pi) * eta / prob_neg
    incorrect_pos = (1 - pi) * (1 - eta) / prob_pos
    incorrect_neg = pi * (1 - eta) / prob_neg

    expected_true_wins[i_idx, j_idx, :] = obs_pos * correct_pos
    expected_true_wins[j_idx, i_idx, :] = obs_neg * correct_neg

    if reverse_update:
        # 对判错部分进行反向更新：观测 j > i 但真实 i > j，反之亦然。
        expected_true_wins[i_idx, j_idx, :] += obs_neg * incorrect_neg
        expected_true_wins[j_idx, i_idx, :] += obs_pos * incorrect_pos

    quality_stats = {
        'correct': obs_pos * correct_pos + obs_neg * correct_neg,
        'total': obs_pos + obs_neg,
    }
    return expected_true_wins, quality_stats


def _compute_adaptive_expectations(
    wins,
    skill_scores,
    eta_t,
    reverse_update,
    max_contamination=ADAPTIVE_MAX_CONTAMINATION,
    temperature=ADAPTIVE_TEMPERATURE,
    decay_power=ADAPTIVE_DECAY_POWER,
):
    """估计污染率，并对 clean-likelihood 最低的一部分比较做 soft clipping。"""
    n, _, R = wins.shape
    expected_true_wins = np.zeros_like(wins, dtype=float)
    empty_diagnostics = {
        'contamination_rate_raw': 0.0,
        'contamination_rate': 0.0,
        'threshold_probability': None,
        'threshold_margin': None,
        'mean_clipping_weight': 1.0,
    }

    if n < 2 or R == 0:
        return expected_true_wins, None, empty_diagnostics

    i_idx, j_idx = np.triu_indices(n, k=1)
    obs_pos = wins[i_idx, j_idx, :]
    obs_neg = wins[j_idx, i_idx, :]
    total_comparisons = np.sum(obs_pos) + np.sum(obs_neg)

    if total_comparisons <= 0:
        return expected_true_wins, None, empty_diagnostics

    pi = skill_scores[i_idx] / (skill_scores[i_idx] + skill_scores[j_idx])
    pi = np.clip(pi[:, np.newaxis], EPS, 1 - EPS)
    eta = eta_t[np.newaxis, :]

    prob_pos = np.clip(pi * eta + (1 - pi) * (1 - eta), EPS, 1.0)
    prob_neg = np.clip((1 - pi) * eta + pi * (1 - eta), EPS, 1.0)
    correct_pos = pi * eta / prob_pos
    correct_neg = (1 - pi) * eta / prob_neg

    error_mass = (
        np.sum(obs_pos * (1 - correct_pos))
        + np.sum(obs_neg * (1 - correct_neg))
    )
    contamination_raw = float(error_mass / total_comparisons)
    contamination = float(np.clip(
        contamination_raw, 0.0, max_contamination
    ))

    score_pos = np.broadcast_to(pi, obs_pos.shape)
    score_neg = np.broadcast_to(1 - pi, obs_neg.shape)
    threshold_probability = _weighted_quantile(
        np.concatenate([score_pos.ravel(), score_neg.ravel()]),
        np.concatenate([obs_pos.ravel(), obs_neg.ravel()]),
        contamination,
    )
    threshold_probability = float(np.clip(
        threshold_probability, EPS, 1 - EPS
    ))
    threshold_margin = math.log(
        threshold_probability / (1 - threshold_probability)
    )

    margin_pos = np.broadcast_to(
        np.log(pi / (1 - pi)), obs_pos.shape
    )
    margin_neg = -margin_pos
    clip_pos = np.exp(
        -(
            np.maximum(threshold_margin - margin_pos, 0.0)
            / temperature
        ) ** decay_power
    )
    clip_neg = np.exp(
        -(
            np.maximum(threshold_margin - margin_neg, 0.0)
            / temperature
        ) ** decay_power
    )

    expected_true_wins[i_idx, j_idx, :] = obs_pos * clip_pos
    expected_true_wins[j_idx, i_idx, :] = obs_neg * clip_neg

    if reverse_update:
        expected_true_wins[i_idx, j_idx, :] += obs_neg * (1 - clip_neg)
        expected_true_wins[j_idx, i_idx, :] += obs_pos * (1 - clip_pos)

    mean_clipping_weight = float(
        (
            np.sum(obs_pos * clip_pos)
            + np.sum(obs_neg * clip_neg)
        )
        / total_comparisons
    )
    quality_stats = {
        'correct': obs_pos * correct_pos + obs_neg * correct_neg,
        'total': obs_pos + obs_neg,
    }
    diagnostics = {
        'contamination_rate_raw': contamination_raw,
        'contamination_rate': contamination,
        'threshold_probability': threshold_probability,
        'threshold_margin': threshold_margin,
        'mean_clipping_weight': mean_clipping_weight,
    }
    return expected_true_wins, quality_stats, diagnostics


def _update_bayesian_elo(
    metric,
    sessions,
    reverse_update,
    shared_rater=False,
    weighting_mode='posterior',
    device='auto',
    prior_config=None,
):
    """共享拟合核心：同时估计 method skill 和 rater 的统一正确率 eta。"""
    start = time.time()

    priors = resolve_prior_config(prior_config)
    a, b = priors['gamma_shape'], priors['gamma_rate']
    alpha, beta = priors['beta_alpha'], priors['beta_beta']
    metric.state['prior_config'] = priors
    default_eta = _prior_mode(alpha, beta)

    scores = metric.state.get('scores', {})
    qualities = metric.state.get('qualities', {})

    methods = metric.state.get('methods', [])
    method_to_index = dict(zip(methods, range(len(methods))))

    raters = metric.state.get('raters', [])
    rater_to_index = dict(zip(raters, range(len(raters))))

    wins = np.zeros((0, 0, 0))
    decisive_wins = np.zeros((0, 0, 0))
    tie_wins = np.zeros((0, 0, 0))

    # 让 Gamma 先验的默认 skill 对齐到初始 ELO 分数。
    elo_offset = math.log10((a - 1) / b) * ELO_SCALE_FACTOR
    elo_offset = ELO_INITIAL_SCORE - elo_offset
    s2e = partial(skill_to_elo, offset=elo_offset)
    e2s = partial(elo_to_skill, offset=elo_offset)

    # 先把原始 sessions 转成 wins[i][j][r]：rater r 认为 method i 赢 method j 的次数。
    for session in sessions:
        for slate in session.slates.all():
            ratings = slate.ratings.all()
            count = len(ratings)

            if count != 2:
                logger.error(
                    'Slate {} in session {} has {} ratings instead of 2.'.format(
                        slate.id, session.id, count
                    )
                )
                continue

            # shared_rater=True 时，即使原数据有 ID，也把全部评价合并到同一个 rater。
            rater = GLOBAL_RATER if shared_rater else _normalize_rater(session.rater)
            method_i = ratings[0].stimulus.name
            method_j = ratings[1].stimulus.name
            s_i, s_j = ratings[0].score, ratings[1].score

            if rater not in rater_to_index:
                index_rater = len(raters)
                raters.append(rater)
                rater_to_index[rater] = index_rater
                wins = np.pad(wins, ((0, 0), (0, 0), (0, 1)))
                decisive_wins = np.pad(
                    decisive_wins, ((0, 0), (0, 0), (0, 1))
                )
                tie_wins = np.pad(tie_wins, ((0, 0), (0, 0), (0, 1)))
            else:
                index_rater = rater_to_index[rater]

            if method_i not in method_to_index:
                index_i = len(methods)
                methods.append(method_i)
                method_to_index[method_i] = index_i
                wins = np.pad(wins, ((0, 1), (0, 0), (0, 0)))
                wins = np.pad(wins, ((0, 0), (0, 1), (0, 0)))
                decisive_wins = np.pad(
                    decisive_wins, ((0, 1), (0, 0), (0, 0))
                )
                decisive_wins = np.pad(
                    decisive_wins, ((0, 0), (0, 1), (0, 0))
                )
                tie_wins = np.pad(tie_wins, ((0, 1), (0, 0), (0, 0)))
                tie_wins = np.pad(tie_wins, ((0, 0), (0, 1), (0, 0)))
            else:
                index_i = method_to_index[method_i]

            if method_j not in method_to_index:
                index_j = len(methods)
                methods.append(method_j)
                method_to_index[method_j] = index_j
                wins = np.pad(wins, ((0, 1), (0, 0), (0, 0)))
                wins = np.pad(wins, ((0, 0), (0, 1), (0, 0)))
                decisive_wins = np.pad(
                    decisive_wins, ((0, 1), (0, 0), (0, 0))
                )
                decisive_wins = np.pad(
                    decisive_wins, ((0, 0), (0, 1), (0, 0))
                )
                tie_wins = np.pad(tie_wins, ((0, 1), (0, 0), (0, 0)))
                tie_wins = np.pad(tie_wins, ((0, 0), (0, 1), (0, 0)))
            else:
                index_j = method_to_index[method_j]

            if s_i > s_j:
                wins[index_i][index_j][index_rater] += 1
                decisive_wins[index_i][index_j][index_rater] += 1
            elif s_j > s_i:
                wins[index_j][index_i][index_rater] += 1
                decisive_wins[index_j][index_i][index_rater] += 1
            else:
                # 平局仍沿用原 BBQ 的处理：看作 forced choice 的期望，一边 0.5。
                wins[index_i][index_j][index_rater] += 0.5
                wins[index_j][index_i][index_rater] += 0.5
                tie_wins[index_i][index_j][index_rater] += 0.5
                tie_wins[index_j][index_i][index_rater] += 0.5

    metric.state['methods'] = methods
    metric.state['raters'] = raters
    metric.state['wins'] = wins.tolist()
    metric.state['tie_wins'] = tie_wins.tolist()

    n = len(methods)
    R = len(raters)

    is_adaptive = weighting_mode == 'adaptive'
    adaptive_diagnostics = None
    previous_contamination = None
    initialize_optimization_state(
        metric,
        methods,
        scores,
        raters=raters,
        qualities=qualities,
        default_quality=default_eta,
    )
    initial_elo = np.array([
        scores.get(m, {}).get('value', ELO_INITIAL_SCORE) for m in methods
    ])
    record_optimization_trace(metric, 0, methods, initial_elo, wins)
    requested_device = 'cpu' if optimization_trace_enabled(metric) else device
    actual_device = resolve_device(requested_device)
    metric.state['computation_backend'] = actual_device

    if actual_device.startswith('cuda'):
        initial_elo = np.array([
            scores.get(m, {}).get('value', ELO_INITIAL_SCORE) for m in methods
        ])
        initial_eta = np.array([
            qualities.get(r, {}).get('value', default_eta) for r in raters
        ])
        cuda_result = fit_correctness_bbq_cuda(
            wins,
            decisive_wins,
            tie_wins,
            e2s(initial_elo),
            initial_eta,
            reverse_update=reverse_update,
            weighting_mode=weighting_mode,
            a=a,
            b=b,
            alpha=alpha,
            beta=beta,
            max_updates=ELO_MAX_UPDATES * sessions.count(),
            convergence_threshold=ELO_CONVERGENCE_THRESHOLD,
            elo_scale_factor=ELO_SCALE_FACTOR,
            adaptive_warmup_updates=ADAPTIVE_WARMUP_UPDATES,
            adaptive_max_contamination=ADAPTIVE_MAX_CONTAMINATION,
            adaptive_temperature=ADAPTIVE_TEMPERATURE,
            adaptive_decay_power=ADAPTIVE_DECAY_POWER,
            eps=EPS,
            device=actual_device,
        )
        for i, method in enumerate(methods):
            scores[method] = {'value': float(s2e(cuda_result['skill'][i]))}
        for r, rater in enumerate(raters):
            qualities[rater] = {'value': float(cuda_result['quality'][r])}

        for i, method_i in enumerate(methods):
            gamma_dist = scipy.stats.gamma(
                a=cuda_result['posterior_shape'][i],
                scale=1 / cuda_result['posterior_rate'][i],
            )
            percentiles = gamma_dist.ppf(
                [0.005, 0.025, 0.05, 0.5, 0.95, 0.975, 0.995]
            )
            scores[method_i]['p005'] = s2e(percentiles[0])
            scores[method_i]['p025'] = s2e(percentiles[1])
            scores[method_i]['p05'] = s2e(percentiles[2])
            scores[method_i]['median'] = s2e(percentiles[3])
            scores[method_i]['p95'] = s2e(percentiles[4])
            scores[method_i]['p975'] = s2e(percentiles[5])
            scores[method_i]['p995'] = s2e(percentiles[6])

        adaptive_diagnostics = cuda_result['adaptive_diagnostics']
        metric.state['scores'] = scores
        metric.state['qualities'] = qualities
        metric.state['rater_qualities'] = {
            r: qualities[r]['value'] for r in raters
        }
        metric.state['rater_correctness'] = dict(
            metric.state['rater_qualities']
        )
        metric.state['update_mode'] = (
            'reverse' if reverse_update else 'downweight'
        )
        metric.state['rater_mode'] = (
            'shared' if shared_rater else 'per_rater'
        )
        metric.state['quality_parameter'] = 'eta'
        metric.state['weighting_mode'] = weighting_mode
        metric.state['iterations'] = cuda_result['iterations']
        metric.state['converged'] = cuda_result['converged']
        if adaptive_diagnostics is not None:
            metric.state['adaptive_clipping'] = {
                **adaptive_diagnostics,
                'warmup_updates': ADAPTIVE_WARMUP_UPDATES,
                'max_contamination': ADAPTIVE_MAX_CONTAMINATION,
                'temperature': ADAPTIVE_TEMPERATURE,
                'decay_power': ADAPTIVE_DECAY_POWER,
            }
        return time.time() - start

    # 核心 EM / 类 EM 迭代：
    # 1. 用当前 skill 和 eta 计算 latent true preference 的后验期望。
    # 2. 用后验期望更新 method skill / ELO。
    # 3. 用“标签标对”的后验期望更新每个 rater 的 eta。
    converged = False
    for t in range(ELO_MAX_UPDATES * sessions.count()):
        elo_scores_t = np.array([scores.get(m, {}).get('value', ELO_INITIAL_SCORE) for m in methods])
        skill_scores_t = e2s(elo_scores_t)

        eta_t = np.array([qualities.get(r, {}).get('value', default_eta) for r in raters])

        if is_adaptive and t < ADAPTIVE_WARMUP_UPDATES:
            # Warm-up 阶段使用全部原始比较拟合初始 Bayesian BT 排名。
            expected_true_wins = wins.copy()
            quality_stats = None
        elif is_adaptive:
            (
                expected_true_wins,
                quality_stats,
                adaptive_diagnostics,
            ) = _compute_adaptive_expectations(
                decisive_wins,
                skill_scores_t,
                eta_t,
                reverse_update,
            )
            expected_true_wins += tie_wins
        else:
            expected_true_wins, quality_stats = _compute_correctness_expectations(
                decisive_wins, skill_scores_t, eta_t, reverse_update
            )
            expected_true_wins += tie_wins

        # skill 更新：降权方法丢弃 1-w；反向方法把 1-w 加到相反胜负方向。
        nominator = a - 1 + np.sum(expected_true_wins, axis=(1, 2))
        denominator = b + np.sum(
            (expected_true_wins + np.transpose(expected_true_wins, (1, 0, 2))) /
            (skill_scores_t[:, np.newaxis, np.newaxis] + skill_scores_t[np.newaxis, :, np.newaxis]),
            axis=(1, 2)
        )

        skill_scores_new = nominator / denominator
        elo_scores_new = s2e(skill_scores_new)
        max_elo_change = np.max(np.abs(elo_scores_new - elo_scores_t)) if n else 0

        for i, method in enumerate(methods):
            scores[method] = {'value': elo_scores_new[i]}

        if quality_stats is not None:
            eta_new = (alpha - 1 + np.sum(quality_stats['correct'], axis=0)) / (
                alpha + beta - 2 + np.sum(quality_stats['total'], axis=0)
            )
            eta_new = np.clip(eta_new, EPS, 1 - EPS)

            for r, rater in enumerate(raters):
                qualities[rater] = {'value': eta_new[r]}

        iteration_converged = False
        if is_adaptive:
            if adaptive_diagnostics is not None:
                contamination = adaptive_diagnostics['contamination_rate']
                contamination_change = (
                    abs(contamination - previous_contamination)
                    if previous_contamination is not None
                    else np.inf
                )
                previous_contamination = contamination
                iteration_converged = (
                    max_elo_change < ELO_CONVERGENCE_THRESHOLD
                    and contamination_change < 1e-4
                )
        else:
            iteration_converged = (
                max_elo_change < ELO_CONVERGENCE_THRESHOLD
            )

        current_elo = np.array([scores[m]['value'] for m in methods])
        record_optimization_trace(
            metric, t + 1, methods, current_elo, wins,
            max_elo_change=max_elo_change, converged=iteration_converged,
        )
        if iteration_converged:
            converged = True
            break

    delta_time = time.time() - start

    metric.state['qualities'] = qualities
    metric.state['rater_qualities'] = {
        r: qualities.get(r, {}).get('value', default_eta).item()
        if hasattr(qualities.get(r, {}).get('value', default_eta), 'item')
        else qualities.get(r, {}).get('value', default_eta)
        for r in raters
    }
    metric.state['rater_correctness'] = dict(metric.state['rater_qualities'])

    # 用最终的后验期望，为每个 method 构造 Gamma 近似后验并输出分位数。
    elo_scores_t = np.array([scores.get(m, {}).get('value', ELO_INITIAL_SCORE) for m in methods])
    skill_scores_t = e2s(elo_scores_t)
    eta_t = np.array([qualities.get(r, {}).get('value', default_eta) for r in raters])
    if is_adaptive:
        (
            expected_true_wins,
            _,
            adaptive_diagnostics,
        ) = _compute_adaptive_expectations(
            decisive_wins,
            skill_scores_t,
            eta_t,
            reverse_update,
        )
    else:
        expected_true_wins, _ = _compute_correctness_expectations(
            decisive_wins, skill_scores_t, eta_t, reverse_update
        )
    expected_true_wins += tie_wins

    nominator = a - 1 + np.sum(expected_true_wins, axis=(1, 2))
    denominator = b + np.sum(
        (expected_true_wins + np.transpose(expected_true_wins, (1, 0, 2))) /
        (skill_scores_t[:, np.newaxis, np.newaxis] + skill_scores_t[np.newaxis, :, np.newaxis]),
        axis=(1, 2)
    )

    for i, method_i in enumerate(methods):
        gamma_dist = scipy.stats.gamma(a=nominator[i], scale=1 / denominator[i])
        percentiles = gamma_dist.ppf([0.005, 0.025, 0.05, 0.5, 0.95, 0.975, 0.995])

        scores[method_i]['p005'] = s2e(percentiles[0])
        scores[method_i]['p025'] = s2e(percentiles[1])
        scores[method_i]['p05'] = s2e(percentiles[2])
        scores[method_i]['median'] = s2e(percentiles[3])
        scores[method_i]['p95'] = s2e(percentiles[4])
        scores[method_i]['p975'] = s2e(percentiles[5])
        scores[method_i]['p995'] = s2e(percentiles[6])

    metric.state['scores'] = scores
    metric.state['update_mode'] = 'reverse' if reverse_update else 'downweight'
    metric.state['rater_mode'] = 'shared' if shared_rater else 'per_rater'
    metric.state['quality_parameter'] = 'eta'
    metric.state['weighting_mode'] = weighting_mode
    metric.state['iterations'] = t + 1 if n else 0
    metric.state['converged'] = converged
    if adaptive_diagnostics is not None:
        metric.state['adaptive_clipping'] = {
            **adaptive_diagnostics,
            'warmup_updates': ADAPTIVE_WARMUP_UPDATES,
            'max_contamination': ADAPTIVE_MAX_CONTAMINATION,
            'temperature': ADAPTIVE_TEMPERATURE,
            'decay_power': ADAPTIVE_DECAY_POWER,
        }
    return delta_time


def update_bayesian_elo(
    metric, sessions, shared_rater=False, device='auto', prior_config=None
):
    """Correctness-BBQ-Downweight：用 w 正向更新，直接丢弃 1-w。"""
    return _update_bayesian_elo(
        metric,
        sessions,
        reverse_update=False,
        shared_rater=shared_rater,
        device=device,
        prior_config=prior_config,
    )


def update_bayesian_elo_reverse(
    metric, sessions, shared_rater=False, device='auto', prior_config=None
):
    """Correctness-BBQ-Reverse：用 w 正向更新，用 1-w 反向更新。"""
    return _update_bayesian_elo(
        metric,
        sessions,
        reverse_update=True,
        shared_rater=shared_rater,
        device=device,
        prior_config=prior_config,
    )


def update_bayesian_elo_adaptive_clip(
    metric, sessions, shared_rater=False, device='auto', prior_config=None
):
    """Adaptive-Clip：自适应估计阈值，对可疑比较做 soft clipping。"""
    return _update_bayesian_elo(
        metric,
        sessions,
        reverse_update=False,
        shared_rater=shared_rater,
        weighting_mode='adaptive',
        device=device,
        prior_config=prior_config,
    )


def update_bayesian_elo_adaptive_flip(
    metric, sessions, shared_rater=False, device='auto', prior_config=None
):
    """Adaptive-Flip：soft clipping 后把剩余权重反向更新。"""
    return _update_bayesian_elo(
        metric,
        sessions,
        reverse_update=True,
        shared_rater=shared_rater,
        weighting_mode='adaptive',
        device=device,
        prior_config=prior_config,
    )
