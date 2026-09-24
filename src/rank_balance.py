"""RankBalance joint ability and rater-reliability estimator.

This implements Equations (1)--(5) from the RankBalance method description.
The zero-mean ability constraint is enforced by centering abilities at every
objective evaluation. A tie is represented by two half-weighted opposite
orientations so every original comparison has total weight one.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Iterable, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from models import ELO_INITIAL_SCORE, ELO_SCALE_FACTOR


DEFAULT_CONFIG: Dict[str, float] = {
    "lambda_theta": 1.0,
    "lambda_u": 1.0,
    "mu": 2.0,
    "max_iter": 1000,
    "tol": 1e-8,
}


def _validate_config(config: Dict[str, Any] | None) -> Dict[str, float]:
    if config is None or "gamma" not in config:
        raise ValueError("RankBalance gamma must be supplied explicitly")
    resolved = dict(DEFAULT_CONFIG)
    if config:
        unknown = sorted(set(config) - (set(resolved) | {"gamma"}))
        if unknown:
            raise ValueError(f"Unknown RankBalance parameters: {unknown}")
        resolved.update({key: float(value) for key, value in config.items()})

    for key in ("gamma", "lambda_theta", "lambda_u"):
        if not np.isfinite(resolved[key]) or resolved[key] < 0.0:
            raise ValueError(f"RankBalance {key} must be finite and nonnegative")
    if resolved["lambda_theta"] <= 0.0 or resolved["lambda_u"] <= 0.0:
        raise ValueError("RankBalance shrinkage parameters must be positive")
    if not np.isfinite(resolved["mu"]):
        raise ValueError("RankBalance mu must be finite")
    if resolved["max_iter"] < 1 or resolved["tol"] <= 0.0:
        raise ValueError("RankBalance max_iter and tol must be positive")
    return resolved


def _comparison_arrays(sessions: Iterable[Any]) -> Tuple[list[str], list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    records = []
    model_names = set()
    rater_names = set()

    for session in sessions:
        rater = str(getattr(session, "rater", None) or session.id)
        rater_names.add(rater)
        for slate in session.slates.all():
            ratings = list(slate.ratings.all())
            if len(ratings) != 2:
                continue
            model_a = str(ratings[0].stimulus.name)
            model_b = str(ratings[1].stimulus.name)
            if model_a == model_b:
                continue
            model_names.update((model_a, model_b))
            score_a = float(ratings[0].score)
            score_b = float(ratings[1].score)
            if score_a > score_b:
                records.append((model_a, model_b, rater, 1.0, 1.0))
            elif score_a < score_b:
                records.append((model_a, model_b, rater, -1.0, 1.0))
            else:
                records.append((model_a, model_b, rater, 1.0, 0.5))
                records.append((model_a, model_b, rater, -1.0, 0.5))

    models = sorted(model_names)
    raters = sorted(rater_names)
    if len(models) < 2 or not records:
        raise ValueError("RankBalance requires at least one valid comparison between two objects")

    model_index = {name: index for index, name in enumerate(models)}
    rater_index = {name: index for index, name in enumerate(raters)}
    index_a = np.fromiter((model_index[row[0]] for row in records), dtype=np.int64)
    index_b = np.fromiter((model_index[row[1]] for row in records), dtype=np.int64)
    rater_ids = np.fromiter((rater_index[row[2]] for row in records), dtype=np.int64)
    signs = np.fromiter((row[3] for row in records), dtype=float)
    weights = np.fromiter((row[4] for row in records), dtype=float)
    return models, raters, index_a, index_b, rater_ids, signs, weights


def _accumulate_gradient(
    derivative_m: np.ndarray,
    derivative_u: np.ndarray,
    index_a: np.ndarray,
    index_b: np.ndarray,
    rater_ids: np.ndarray,
    signs: np.ndarray,
    model_count: int,
    rater_count: int,
) -> np.ndarray:
    theta_gradient = np.zeros(model_count, dtype=float)
    signed = derivative_m * signs
    np.add.at(theta_gradient, index_a, signed)
    np.add.at(theta_gradient, index_b, -signed)
    rater_gradient = np.zeros(rater_count, dtype=float)
    np.add.at(rater_gradient, rater_ids, derivative_u)
    return np.concatenate((theta_gradient, rater_gradient))


def _fit_initial_bt(
    model_count: int,
    index_a: np.ndarray,
    index_b: np.ndarray,
    signs: np.ndarray,
    weights: np.ndarray,
    lambda_theta: float,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    def objective(raw_theta: np.ndarray) -> Tuple[float, np.ndarray]:
        theta = raw_theta - raw_theta.mean()
        margin = signs * (theta[index_a] - theta[index_b])
        value = np.sum(weights * np.logaddexp(0.0, -margin))
        value += 0.5 * lambda_theta * np.dot(theta, theta)
        derivative = -weights * expit(-margin)
        gradient = np.zeros(model_count, dtype=float)
        signed = derivative * signs
        np.add.at(gradient, index_a, signed)
        np.add.at(gradient, index_b, -signed)
        gradient += lambda_theta * theta
        gradient -= gradient.mean()
        return float(value), gradient

    result = minimize(
        objective,
        np.zeros(model_count, dtype=float),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": max_iter, "ftol": tol, "gtol": tol},
    )
    if not np.all(np.isfinite(result.x)):
        raise RuntimeError(f"RankBalance BT initialization failed: {result.message}")
    return result.x - result.x.mean()


def update_rank_balance(metric, sessions, config=None):
    """Fit RankBalance and store ELO-scaled abilities in ``metric.state``."""
    started = time.time()
    settings = _validate_config(config)
    (
        models,
        raters,
        index_a,
        index_b,
        rater_ids,
        signs,
        weights,
    ) = _comparison_arrays(sessions)
    model_count = len(models)
    rater_count = len(raters)
    max_iter = int(settings["max_iter"])
    tol = float(settings["tol"])

    initial_theta = _fit_initial_bt(
        model_count,
        index_a,
        index_b,
        signs,
        weights,
        settings["lambda_theta"],
        max_iter,
        tol,
    )
    initial = np.concatenate(
        (initial_theta, np.full(rater_count, settings["mu"], dtype=float))
    )

    gamma = settings["gamma"]
    lambda_theta = settings["lambda_theta"]
    lambda_u = settings["lambda_u"]
    mu = settings["mu"]

    def objective(parameters: np.ndarray) -> Tuple[float, np.ndarray]:
        theta = parameters[:model_count]
        theta = theta - theta.mean()
        u = parameters[model_count:]
        margin = signs * (theta[index_a] - theta[index_b])
        row_u = u[rater_ids]

        sigmoid_m = expit(margin)
        sigmoid_u = expit(row_u)
        sigmoid_sum = expit(margin + row_u)
        loss = (
            np.logaddexp(0.0, margin)
            + np.logaddexp(0.0, row_u)
            - np.logaddexp(0.0, margin + row_u)
        )
        derivative_m = sigmoid_m - sigmoid_sum
        derivative_u = sigmoid_u - sigmoid_sum

        if gamma:
            sigmoid_difference = expit(row_u - margin)
            a = derivative_m
            b = derivative_u
            c = sigmoid_sum + sigmoid_difference - 1.0
            d = sigmoid_sum - sigmoid_difference

            q_m = sigmoid_m * (1.0 - sigmoid_m)
            q_u = sigmoid_u * (1.0 - sigmoid_u)
            q_sum = sigmoid_sum * (1.0 - sigmoid_sum)
            q_difference = sigmoid_difference * (1.0 - sigmoid_difference)

            regularizer = a * a + 0.5 * b * b + c * c + 0.5 * d * d
            regularizer_m = (
                2.0 * a * (q_m - q_sum)
                - b * q_sum
                + 2.0 * c * (q_sum - q_difference)
                + d * (q_sum + q_difference)
            )
            regularizer_u = (
                -2.0 * a * q_sum
                + b * (q_u - q_sum)
                + 2.0 * c * (q_sum + q_difference)
                + d * (q_sum - q_difference)
            )
            loss = loss + gamma * regularizer
            derivative_m = derivative_m + gamma * regularizer_m
            derivative_u = derivative_u + gamma * regularizer_u

        value = np.sum(weights * loss)
        value += 0.5 * lambda_theta * np.dot(theta, theta)
        centered_u = u - mu
        value += 0.5 * lambda_u * np.dot(centered_u, centered_u)

        gradient = _accumulate_gradient(
            weights * derivative_m,
            weights * derivative_u,
            index_a,
            index_b,
            rater_ids,
            signs,
            model_count,
            rater_count,
        )
        gradient[:model_count] += lambda_theta * theta
        gradient[:model_count] -= gradient[:model_count].mean()
        gradient[model_count:] += lambda_u * centered_u
        return float(value), gradient

    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": max_iter, "ftol": tol, "gtol": tol, "maxls": 50},
    )
    if not np.all(np.isfinite(result.x)) or not np.isfinite(result.fun):
        raise RuntimeError(f"RankBalance optimization failed: {result.message}")

    theta = result.x[:model_count]
    theta -= theta.mean()
    u = result.x[model_count:]
    elo_scale = ELO_SCALE_FACTOR / np.log(10.0)
    metric.state["scores"] = {
        model: {
            "value": float(ELO_INITIAL_SCORE + elo_scale * ability),
            "p005": float("nan"),
            "p995": float("nan"),
        }
        for model, ability in zip(models, theta)
    }
    metric.state["qualities"] = {
        rater: {"value": float(reliability)}
        for rater, reliability in zip(raters, expit(u))
    }
    metric.state["quality_parameter"] = "agreement_probability"
    metric.state["rank_balance_config"] = settings
    metric.state["rank_balance_optimization"] = {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(result.nit),
        "evaluations": int(result.nfev),
        "objective": float(result.fun),
        "gradient_inf_norm": float(np.linalg.norm(result.jac, ord=np.inf)),
    }
    metric.state["converged"] = bool(result.success)
    metric.state["computation_backend"] = "cpu"
    return time.time() - started
