"""Maximum-likelihood am-ELO baseline.

The model follows Liu et al. (2025):

    P(i beats j | rater k) = sigmoid(theta_k * (R_i - R_j))

Model scores and rater abilities are fitted jointly without Bayesian priors.
The rater abilities are constrained to sum to one, as in the paper.
"""

import logging
import time
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from models import ELO_INITIAL_SCORE, ELO_SCALE_FACTOR, Metric, QuerySet


logger = logging.getLogger(__name__)

AM_ELO_MAX_ITERATIONS = 10000
DEFAULT_LABEL_SMOOTHING = 0.10


def _collect_comparisons(
    sessions: QuerySet,
) -> Tuple[List[str], List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert Session/Slate objects into indexed am-ELO observations."""
    methods: List[str] = []
    raters: List[str] = []
    method_to_index: Dict[str, int] = {}
    rater_to_index: Dict[str, int] = {}
    method_i: List[int] = []
    method_j: List[int] = []
    rater_indices: List[int] = []
    outcomes: List[float] = []

    for session in sessions:
        rater = str(session.rater if session.rater is not None else session.id)
        if rater not in rater_to_index:
            rater_to_index[rater] = len(raters)
            raters.append(rater)
        rater_index = rater_to_index[rater]

        for slate in session.slates.all():
            ratings = slate.ratings.all()
            if len(ratings) != 2:
                logger.warning(
                    "Skipping slate %s: expected two ratings, found %d",
                    slate.id,
                    len(ratings),
                )
                continue

            names = [rating.stimulus.name for rating in ratings]
            for name in names:
                if name not in method_to_index:
                    method_to_index[name] = len(methods)
                    methods.append(name)

            if ratings[0].score > ratings[1].score:
                outcome = 1.0
            elif ratings[0].score < ratings[1].score:
                outcome = 0.0
            else:
                # The paper treats a tie as a fractional binary target.
                outcome = 0.5

            method_i.append(method_to_index[names[0]])
            method_j.append(method_to_index[names[1]])
            rater_indices.append(rater_index)
            outcomes.append(outcome)

    return (
        methods,
        raters,
        np.asarray(method_i, dtype=np.int64),
        np.asarray(method_j, dtype=np.int64),
        np.asarray(rater_indices, dtype=np.int64),
        np.asarray(outcomes, dtype=np.float64),
    )


def _fit_bt_mle(
    method_i: np.ndarray,
    method_j: np.ndarray,
    outcomes: np.ndarray,
    method_count: int,
) -> Tuple[np.ndarray, object]:
    """Fit the ordinary Bradley-Terry full-data MLE used by m-ELO."""
    if method_count <= 1:
        return np.zeros(method_count, dtype=np.float64), None

    def objective(free_scores: np.ndarray):
        scores = np.concatenate(([0.0], free_scores))
        logits = scores[method_i] - scores[method_j]
        residual = expit(logits) - outcomes
        loss = np.logaddexp(0.0, logits).sum() - np.dot(outcomes, logits)

        score_gradient = np.zeros(method_count, dtype=np.float64)
        np.add.at(score_gradient, method_i, residual)
        np.add.at(score_gradient, method_j, -residual)
        return float(loss), score_gradient[1:]

    result = minimize(
        objective,
        np.zeros(method_count - 1, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": AM_ELO_MAX_ITERATIONS, "ftol": 1e-10, "gtol": 1e-7},
    )
    scores = np.concatenate(([0.0], result.x))
    return scores - scores.mean(), result


def _point_estimate_scores(
    methods: List[str], elo_scores: np.ndarray, raw_scores: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """Build the metric.state score records used by the shared processor."""
    scores: Dict[str, Dict[str, float]] = {}
    for method, elo_score, raw_score in zip(methods, elo_scores, raw_scores):
        scores[method] = {
            "value": float(elo_score),
            "raw_value": float(raw_score),
            # MLE methods are point estimates; Bayesian intervals are unavailable.
            "p005": np.nan,
            "p025": np.nan,
            "p05": np.nan,
            "median": np.nan,
            "p95": np.nan,
            "p975": np.nan,
            "p995": np.nan,
        }
    return scores


def update_m_elo(metric: Metric, sessions: QuerySet) -> float:
    """Fit the paper's m-ELO baseline: ordinary BT full-data MLE."""
    start_time = time.time()
    methods, raters, method_i, method_j, _, outcomes = _collect_comparisons(
        sessions
    )
    method_count = len(methods)

    if not len(outcomes) or method_count < 2:
        metric.state["methods"] = methods
        metric.state["raters"] = raters
        metric.state["scores"] = {}
        metric.state["qualities"] = {}
        return time.time() - start_time

    logit_scores, result = _fit_bt_mle(
        method_i, method_j, outcomes, method_count
    )
    elo_scores = (
        ELO_INITIAL_SCORE
        + ELO_SCALE_FACTOR * logit_scores / np.log(10.0)
    )

    metric.state["methods"] = methods
    metric.state["raters"] = raters
    metric.state["scores"] = _point_estimate_scores(
        methods, elo_scores, logit_scores
    )
    metric.state["qualities"] = {}
    metric.state["rater_qualities"] = {}
    metric.state["m_elo_optimization"] = {
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(result.nit),
        "comparisons": int(len(outcomes)),
        "log_likelihood": float(-result.fun),
    }
    if not result.success:
        logger.warning("m-ELO optimizer stopped without convergence: %s", result.message)
    return time.time() - start_time


def update_label_smoothed_bt(
    metric: Metric,
    sessions: QuerySet,
    label_smoothing: float = DEFAULT_LABEL_SMOOTHING,
) -> float:
    """Fit Bradley--Terry MLE with fixed symmetric label smoothing.

    A win/loss target of 1/0 becomes ``1-epsilon``/``epsilon`` while a
    draw remains 0.5. Setting epsilon to zero recovers m-ELO exactly.
    """
    start_time = time.time()
    epsilon = float(label_smoothing)
    if not np.isfinite(epsilon) or not 0.0 <= epsilon < 0.5:
        raise ValueError("label_smoothing must be finite and in [0, 0.5)")

    methods, raters, method_i, method_j, _, outcomes = _collect_comparisons(
        sessions
    )
    method_count = len(methods)
    if not len(outcomes) or method_count < 2:
        metric.state["methods"] = methods
        metric.state["raters"] = raters
        metric.state["scores"] = {}
        metric.state["qualities"] = {}
        return time.time() - start_time

    smoothed_outcomes = epsilon + (1.0 - 2.0 * epsilon) * outcomes
    logit_scores, result = _fit_bt_mle(
        method_i, method_j, smoothed_outcomes, method_count
    )
    elo_scores = (
        ELO_INITIAL_SCORE
        + ELO_SCALE_FACTOR * logit_scores / np.log(10.0)
    )

    metric.state["methods"] = methods
    metric.state["raters"] = raters
    metric.state["scores"] = _point_estimate_scores(
        methods, elo_scores, logit_scores
    )
    metric.state["qualities"] = {}
    metric.state["rater_qualities"] = {}
    metric.state["label_smoothing"] = epsilon
    metric.state["ls_bt_optimization"] = {
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(result.nit),
        "comparisons": int(len(outcomes)),
        "label_smoothing": epsilon,
        "log_likelihood": float(-result.fun),
    }
    if not result.success:
        logger.warning(
            "LS-BT optimizer stopped without convergence: %s", result.message
        )
    return time.time() - start_time


def _unpack_parameters(
    parameters: np.ndarray,
    method_count: int,
    rater_count: int,
) -> Tuple[np.ndarray, np.ndarray]:
    # Fix the first model score to remove the additive BT indeterminacy.
    scores = np.concatenate(([0.0], parameters[: method_count - 1]))
    if rater_count == 1:
        relative_abilities = np.ones(1, dtype=np.float64)
    else:
        free_relative_abilities = parameters[method_count - 1 :]
        relative_abilities = np.concatenate(
            (
                free_relative_abilities,
                [rater_count - free_relative_abilities.sum()],
            )
        )
    return scores, relative_abilities


def update_am_elo(metric: Metric, sessions: QuerySet) -> float:
    """Fit am-ELO model scores and rater abilities by point-estimate MLE."""
    start_time = time.time()
    methods, raters, method_i, method_j, rater_indices, outcomes = (
        _collect_comparisons(sessions)
    )
    method_count = len(methods)
    rater_count = len(raters)

    if not len(outcomes) or method_count < 2 or rater_count < 1:
        metric.state["methods"] = methods
        metric.state["raters"] = raters
        metric.state["scores"] = {}
        metric.state["qualities"] = {}
        return time.time() - start_time

    bt_scores, _ = _fit_bt_mle(
        method_i, method_j, outcomes, method_count
    )
    # We optimize alpha_k = M * theta_k, whose mean is one. This is exactly
    # equivalent to the paper's sum(theta)=1 constraint but better conditioned.
    initial_parameters = [bt_scores[1:]]
    if rater_count > 1:
        initial_parameters.append(
            np.ones(rater_count - 1, dtype=np.float64)
        )
    x0 = np.concatenate(initial_parameters)

    def objective(parameters: np.ndarray):
        scores, relative_abilities = _unpack_parameters(
            parameters, method_count, rater_count
        )
        score_differences = scores[method_i] - scores[method_j]
        logits = relative_abilities[rater_indices] * score_differences
        residual = expit(logits) - outcomes
        loss = np.logaddexp(0.0, logits).sum() - np.dot(outcomes, logits)

        score_gradient = np.zeros(method_count, dtype=np.float64)
        weighted_residual = residual * relative_abilities[rater_indices]
        np.add.at(score_gradient, method_i, weighted_residual)
        np.add.at(score_gradient, method_j, -weighted_residual)

        ability_gradient = np.zeros(rater_count, dtype=np.float64)
        np.add.at(
            ability_gradient,
            rater_indices,
            residual * score_differences,
        )

        gradient_parts = [score_gradient[1:]]
        if rater_count > 1:
            # alpha_last = M - sum(alpha_free).
            gradient_parts.append(
                ability_gradient[:-1] - ability_gradient[-1]
            )
        return float(loss), np.concatenate(gradient_parts)

    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": AM_ELO_MAX_ITERATIONS,
            "ftol": 1e-10,
            "gtol": 1e-7,
            "maxls": 50,
        },
    )
    if not result.success:
        logger.warning("am-ELO optimizer stopped without convergence: %s", result.message)

    logit_scores, relative_abilities = _unpack_parameters(
        result.x, method_count, rater_count
    )
    logit_scores = logit_scores - logit_scores.mean()
    abilities = relative_abilities / rater_count
    raw_scores = logit_scores * rater_count

    # Convert raw am-ELO scores to the usual Elo display scale. Since the
    # normalized average ability is 1/M, this preserves all fitted probabilities
    # for an average rater and does not change the ranking.
    elo_scores = (
        ELO_INITIAL_SCORE
        + ELO_SCALE_FACTOR
        * logit_scores
        / np.log(10.0)
    )

    scores = _point_estimate_scores(methods, elo_scores, raw_scores)

    qualities = {
        rater: {
            "value": float(ability),
            "relative_value": float(relative_ability),
        }
        for rater, ability, relative_ability in zip(
            raters, abilities, relative_abilities
        )
    }
    metric.state["methods"] = methods
    metric.state["raters"] = raters
    metric.state["scores"] = scores
    metric.state["qualities"] = qualities
    metric.state["rater_qualities"] = {
        rater: quality["value"] for rater, quality in qualities.items()
    }
    metric.state["quality_parameter"] = "ability"
    metric.state["am_elo_optimization"] = {
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(result.nit),
        "comparisons": int(len(outcomes)),
        "negative_ability_raters": int(np.sum(abilities < 0)),
        "log_likelihood": float(-result.fun),
    }
    return time.time() - start_time
