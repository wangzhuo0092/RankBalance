"""Optimization routines used only by Experiment 6.

The regular project entry points remain unchanged. These wrappers add random
initialization and per-iteration callbacks for m-ELO and normalized am-ELO, and
provide the order-sensitive Traditional ELO baseline used by the am-ELO paper.
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from am_elo import (
    AM_ELO_MAX_ITERATIONS,
    _collect_comparisons,
    _fit_bt_mle,
    _point_estimate_scores,
    _unpack_parameters,
)
from models import (
    ELO_INITIAL_SCORE,
    ELO_SCALE_FACTOR,
    Metric,
    QuerySet,
    model_only_nll,
    record_optimization_trace,
)


def _indexed_wins(
    method_i: np.ndarray,
    method_j: np.ndarray,
    outcomes: np.ndarray,
    method_count: int,
) -> np.ndarray:
    wins = np.zeros((method_count, method_count), dtype=float)
    np.add.at(wins, (method_i, method_j), outcomes)
    np.add.at(wins, (method_j, method_i), 1.0 - outcomes)
    return wins


def _trace_config(metric: Metric) -> Dict:
    return metric.state.get('_optimization_trace_config', {})


def _random_bt_initialization(metric: Metric, method_count: int) -> np.ndarray:
    config = _trace_config(metric)
    if not config.get('random_initialization', True):
        return np.zeros(max(0, method_count - 1), dtype=float)
    rng = np.random.default_rng(int(config.get('seed', 0)))
    elo_std = float(config.get('initial_elo_std', 100.0))
    logit_std = elo_std * np.log(10.0) / ELO_SCALE_FACTOR
    scores = rng.normal(0.0, logit_std, size=method_count)
    scores -= scores[0]
    return scores[1:]


def _elo_from_logit(logit_scores: np.ndarray) -> np.ndarray:
    centered = logit_scores - np.mean(logit_scores)
    return (
        ELO_INITIAL_SCORE
        + ELO_SCALE_FACTOR * centered / np.log(10.0)
    )


def _finalize_trace(metric: Metric, success: bool) -> None:
    trace = metric.state.get('optimization_trace', [])
    if trace:
        trace[-1]['converged'] = bool(success)


def update_m_elo_traced(metric: Metric, sessions: QuerySet) -> float:
    """Fit m-ELO with a seeded initialization and an L-BFGS trajectory."""
    started = time.time()
    methods, raters, method_i, method_j, _, outcomes = _collect_comparisons(
        sessions
    )
    method_count = len(methods)
    wins = _indexed_wins(method_i, method_j, outcomes, method_count)

    if not len(outcomes) or method_count < 2:
        metric.state.update({
            'methods': methods,
            'raters': raters,
            'scores': {},
            'qualities': {},
            'converged': False,
            'iterations': 0,
        })
        return time.time() - started

    def objective(free_scores: np.ndarray):
        scores = np.concatenate(([0.0], free_scores))
        logits = scores[method_i] - scores[method_j]
        residual = expit(logits) - outcomes
        loss = np.logaddexp(0.0, logits).sum() - np.dot(outcomes, logits)
        gradient = np.zeros(method_count, dtype=float)
        np.add.at(gradient, method_i, residual)
        np.add.at(gradient, method_j, -residual)
        return float(loss), gradient[1:]

    initial = _random_bt_initialization(metric, method_count)
    previous_elo = _elo_from_logit(np.concatenate(([0.0], initial)))
    record_optimization_trace(
        metric, 0, methods, previous_elo, wins,
        objective=objective(initial)[0] / len(outcomes),
    )
    iteration = 0

    def callback(free_scores: np.ndarray):
        nonlocal iteration, previous_elo
        iteration += 1
        logit_scores = np.concatenate(([0.0], free_scores))
        elo_scores = _elo_from_logit(logit_scores)
        max_change = float(np.max(np.abs(elo_scores - previous_elo)))
        loss = objective(free_scores)[0] / len(outcomes)
        record_optimization_trace(
            metric, iteration, methods, elo_scores, wins,
            max_elo_change=max_change, objective=loss,
        )
        previous_elo = elo_scores

    result = minimize(
        objective,
        initial,
        method='L-BFGS-B',
        jac=True,
        callback=callback,
        options={
            'maxiter': AM_ELO_MAX_ITERATIONS,
            'ftol': 1e-10,
            'gtol': 1e-7,
        },
    )
    logit_scores = np.concatenate(([0.0], result.x))
    logit_scores -= logit_scores.mean()
    elo_scores = _elo_from_logit(logit_scores)
    record_optimization_trace(
        metric, int(result.nit), methods, elo_scores, wins,
        max_elo_change=float(np.max(np.abs(elo_scores - previous_elo))),
        objective=float(result.fun / len(outcomes)),
        converged=bool(result.success),
    )

    metric.state.update({
        'methods': methods,
        'raters': raters,
        'scores': _point_estimate_scores(methods, elo_scores, logit_scores),
        'qualities': {},
        'rater_qualities': {},
        'iterations': int(result.nit),
        'converged': bool(result.success),
        'computation_backend': 'cpu',
        'm_elo_optimization': {
            'success': bool(result.success),
            'message': str(result.message),
            'iterations': int(result.nit),
            'comparisons': int(len(outcomes)),
            'log_likelihood': float(-result.fun),
        },
    })
    return time.time() - started


def update_am_elo_traced(metric: Metric, sessions: QuerySet) -> float:
    """Fit normalized am-ELO with seeded model and rater initialization."""
    started = time.time()
    methods, raters, method_i, method_j, rater_indices, outcomes = (
        _collect_comparisons(sessions)
    )
    method_count = len(methods)
    rater_count = len(raters)
    wins = _indexed_wins(method_i, method_j, outcomes, method_count)

    if not len(outcomes) or method_count < 2 or rater_count < 1:
        metric.state.update({
            'methods': methods,
            'raters': raters,
            'scores': {},
            'qualities': {},
            'converged': False,
            'iterations': 0,
        })
        return time.time() - started

    config = _trace_config(metric)
    if config.get('random_initialization', True):
        rng = np.random.default_rng(int(config.get('seed', 0)))
        free_scores = _random_bt_initialization(metric, method_count)
        if rater_count > 1:
            ability_std = float(config.get('initial_ability_std', 0.1))
            abilities = rng.normal(1.0, ability_std, size=rater_count)
            abilities -= abilities.mean() - 1.0
            initial = np.concatenate((free_scores, abilities[:-1]))
        else:
            initial = free_scores
    else:
        bt_scores, _ = _fit_bt_mle(
            method_i, method_j, outcomes, method_count
        )
        parts = [bt_scores[1:]]
        if rater_count > 1:
            parts.append(np.ones(rater_count - 1, dtype=float))
        initial = np.concatenate(parts)

    def objective(parameters: np.ndarray):
        scores, relative_abilities = _unpack_parameters(
            parameters, method_count, rater_count
        )
        differences = scores[method_i] - scores[method_j]
        logits = relative_abilities[rater_indices] * differences
        residual = expit(logits) - outcomes
        loss = np.logaddexp(0.0, logits).sum() - np.dot(outcomes, logits)

        score_gradient = np.zeros(method_count, dtype=float)
        weighted = residual * relative_abilities[rater_indices]
        np.add.at(score_gradient, method_i, weighted)
        np.add.at(score_gradient, method_j, -weighted)

        ability_gradient = np.zeros(rater_count, dtype=float)
        np.add.at(
            ability_gradient,
            rater_indices,
            residual * differences,
        )
        parts = [score_gradient[1:]]
        if rater_count > 1:
            parts.append(ability_gradient[:-1] - ability_gradient[-1])
        return float(loss), np.concatenate(parts)

    initial_scores, _ = _unpack_parameters(
        initial, method_count, rater_count
    )
    previous_elo = _elo_from_logit(initial_scores)
    record_optimization_trace(
        metric, 0, methods, previous_elo, wins,
        objective=objective(initial)[0] / len(outcomes),
    )
    iteration = 0

    def callback(parameters: np.ndarray):
        nonlocal iteration, previous_elo
        iteration += 1
        scores, _ = _unpack_parameters(
            parameters, method_count, rater_count
        )
        elo_scores = _elo_from_logit(scores)
        max_change = float(np.max(np.abs(elo_scores - previous_elo)))
        loss = objective(parameters)[0] / len(outcomes)
        record_optimization_trace(
            metric, iteration, methods, elo_scores, wins,
            max_elo_change=max_change, objective=loss,
        )
        previous_elo = elo_scores

    result = minimize(
        objective,
        initial,
        method='L-BFGS-B',
        jac=True,
        callback=callback,
        options={
            'maxiter': AM_ELO_MAX_ITERATIONS,
            'ftol': 1e-10,
            'gtol': 1e-7,
            'maxls': 50,
        },
    )
    logit_scores, relative_abilities = _unpack_parameters(
        result.x, method_count, rater_count
    )
    logit_scores -= logit_scores.mean()
    abilities = relative_abilities / rater_count
    raw_scores = logit_scores * rater_count
    elo_scores = _elo_from_logit(logit_scores)
    record_optimization_trace(
        metric, int(result.nit), methods, elo_scores, wins,
        max_elo_change=float(np.max(np.abs(elo_scores - previous_elo))),
        objective=float(result.fun / len(outcomes)),
        converged=bool(result.success),
    )

    qualities = {
        rater: {
            'value': float(ability),
            'relative_value': float(relative),
        }
        for rater, ability, relative in zip(
            raters, abilities, relative_abilities
        )
    }
    metric.state.update({
        'methods': methods,
        'raters': raters,
        'scores': _point_estimate_scores(methods, elo_scores, raw_scores),
        'qualities': qualities,
        'rater_qualities': {
            rater: item['value'] for rater, item in qualities.items()
        },
        'iterations': int(result.nit),
        'converged': bool(result.success),
        'computation_backend': 'cpu',
        'am_elo_optimization': {
            'success': bool(result.success),
            'message': str(result.message),
            'iterations': int(result.nit),
            'comparisons': int(len(outcomes)),
            'negative_ability_raters': int(np.sum(abilities < 0)),
            'log_likelihood': float(-result.fun),
        },
    })
    return time.time() - started


def update_traditional_elo(
    metric: Metric,
    sessions: QuerySet,
    shuffles: int = 1000,
    k_factor: float = 4.0,
) -> float:
    """Average Traditional ELO over shuffled data orders, as in am-ELO."""
    started = time.time()
    methods, raters, method_i, method_j, _, outcomes = _collect_comparisons(
        sessions
    )
    method_count = len(methods)
    config = _trace_config(metric)
    seed = int(config.get('seed', 0))
    rng = np.random.default_rng(seed)
    final_scores: List[np.ndarray] = []
    order_runs: List[Dict] = []

    for shuffle_index in range(int(shuffles)):
        scores = np.full(method_count, ELO_INITIAL_SCORE, dtype=float)
        order = rng.permutation(len(outcomes))
        for row in order:
            i = method_i[row]
            j = method_j[row]
            expected_i = 1.0 / (
                1.0 + 10.0 ** ((scores[j] - scores[i]) / ELO_SCALE_FACTOR)
            )
            residual = outcomes[row] - expected_i
            scores[i] += k_factor * residual
            scores[j] -= k_factor * residual
        final_scores.append(scores)
        order_runs.append({
            'shuffle': shuffle_index,
            'model_only_nll': model_only_nll(
                scores,
                _indexed_wins(method_i, method_j, outcomes, method_count),
            ),
            'ranking': [
                methods[index]
                for index in np.argsort(-scores, kind='stable')
            ],
            'scores': {
                method: float(value)
                for method, value in zip(methods, scores)
            },
        })

    stacked = np.vstack(final_scores)
    mean_scores = stacked.mean(axis=0)
    metric.state.update({
        'methods': methods,
        'raters': raters,
        'scores': _point_estimate_scores(
            methods, mean_scores, mean_scores - mean_scores.mean()
        ),
        'qualities': {},
        'rater_qualities': {},
        'traditional_elo_runs': order_runs,
        'iterations': int(shuffles),
        'converged': True,
        'computation_backend': 'cpu',
    })
    return time.time() - started
