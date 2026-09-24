"""Matched RankBalance ablations and method-specific influence attacks."""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit


ELO_OFFSET = 2000.0
ELO_SCALE = 400.0 / math.log(10.0)


def _arrays(
    frame: pd.DataFrame,
    models: Optional[Sequence[str]] = None,
    raters: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    models = list(models or sorted(set(frame["methodA"].astype(str)) | set(frame["methodB"].astype(str))))
    raters = list(raters or sorted(frame["answerer"].astype(str).unique()))
    model_index = {name: i for i, name in enumerate(models)}
    rater_index = {name: i for i, name in enumerate(raters)}
    rows = []
    for row_id, row in enumerate(frame.itertuples(index=False)):
        a = str(row.methodA)
        b = str(row.methodB)
        rater = str(row.answerer)
        if a not in model_index or b not in model_index or rater not in rater_index or a == b:
            continue
        label = str(row.answerValue).strip().lower()
        if label == "a":
            rows.append((row_id, model_index[a], model_index[b], rater_index[rater], 1.0, 1.0))
        elif label == "b":
            rows.append((row_id, model_index[a], model_index[b], rater_index[rater], -1.0, 1.0))
        elif label in {"draw", "tie"}:
            rows.append((row_id, model_index[a], model_index[b], rater_index[rater], 1.0, 0.5))
            rows.append((row_id, model_index[a], model_index[b], rater_index[rater], -1.0, 0.5))
    if not rows or len(models) < 2:
        raise ValueError("RankBalance needs valid comparisons between at least two objects")
    columns = list(zip(*rows))
    return {
        "models": models,
        "raters": raters,
        "model_index": model_index,
        "rater_index": rater_index,
        "source_row": np.asarray(columns[0], dtype=np.int64),
        "a": np.asarray(columns[1], dtype=np.int64),
        "b": np.asarray(columns[2], dtype=np.int64),
        "r": np.asarray(columns[3], dtype=np.int64),
        "sign": np.asarray(columns[4], dtype=float),
        "weight": np.asarray(columns[5], dtype=float),
    }


def _row_terms(margin, row_u, gamma: float, mode: str):
    sm, su, ss = expit(margin), expit(row_u), expit(margin + row_u)
    loss = np.logaddexp(0.0, margin) + np.logaddexp(0.0, row_u) - np.logaddexp(0.0, margin + row_u)
    dm, du = sm - ss, su - ss
    if gamma == 0.0:
        return loss, dm, du
    qm, qu, qs = sm * (1.0 - sm), su * (1.0 - su), ss * (1.0 - ss)
    regularizer = np.zeros_like(dm)
    rm = np.zeros_like(dm)
    ru = np.zeros_like(du)
    if mode in {"delete_only", "full", "ability_only"}:
        regularizer += dm * dm
        rm += 2.0 * dm * (qm - qs)
        ru += -2.0 * dm * qs
    if mode in {"delete_only", "full", "reliability_only"}:
        regularizer += 0.5 * du * du
        rm += -du * qs
        ru += du * (qu - qs)
    if mode in {"full", "ability_only", "reliability_only", "flip_only"}:
        sd = expit(row_u - margin)
        qd = sd * (1.0 - sd)
        c, d = ss + sd - 1.0, ss - sd
        if mode in {"full", "ability_only", "flip_only"}:
            regularizer += c * c
            rm += 2.0 * c * (qs - qd)
            ru += 2.0 * c * (qs + qd)
        if mode in {"full", "reliability_only", "flip_only"}:
            regularizer += 0.5 * d * d
            rm += d * (qs + qd)
            ru += d * (qs - qd)
    return loss + gamma * regularizer, dm + gamma * rm, du + gamma * ru


def _gradient(dm, du, design):
    m = len(design["models"])
    result = np.zeros(m + len(design["raters"]), dtype=float)
    signed = dm * design["sign"]
    np.add.at(result, design["a"], signed)
    np.add.at(result, design["b"], -signed)
    np.add.at(result, m + design["r"], du)
    return result


def fit_rank_balance(
    frame: pd.DataFrame,
    config: Dict[str, Any],
    models: Optional[Sequence[str]] = None,
    raters: Optional[Sequence[str]] = None,
    initial_fit: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    started = time.time()
    gamma = float(config["gamma"])
    lambda_theta = float(config["lambda_theta"])
    lambda_u = float(config["lambda_u"])
    mu = float(config["mu"])
    max_iter = int(config.get("max_iter", 1000))
    tol = float(config.get("tol", 1e-8))
    mode = str(config.get("regularizer", "full"))
    if (
        gamma < 0
        or lambda_theta <= 0
        or lambda_u <= 0
        or mode not in {
            "delete_only",
            "flip_only",
            "ability_only",
            "reliability_only",
            "full",
        }
    ):
        raise ValueError("Invalid RankBalance ablation configuration")
    design = _arrays(frame, models=models, raters=raters)
    m, r = len(design["models"]), len(design["raters"])

    def bt_objective(raw_theta):
        theta = raw_theta - raw_theta.mean()
        margin = design["sign"] * (theta[design["a"]] - theta[design["b"]])
        value = np.sum(design["weight"] * np.logaddexp(0.0, -margin))
        value += 0.5 * lambda_theta * theta.dot(theta)
        derivative = -design["weight"] * expit(-margin)
        gradient = np.zeros(m, dtype=float)
        signed = derivative * design["sign"]
        np.add.at(gradient, design["a"], signed)
        np.add.at(gradient, design["b"], -signed)
        gradient += lambda_theta * theta
        gradient -= gradient.mean()
        return float(value), gradient

    if initial_fit is None:
        initial_bt = minimize(
            bt_objective,
            np.zeros(m),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": max_iter, "ftol": tol, "gtol": tol},
        )
        initial_theta = initial_bt.x - initial_bt.x.mean()
        initial = np.concatenate([initial_theta, np.full(r, mu)])
    else:
        initial = np.concatenate(
            [
                np.asarray(initial_fit["theta"], dtype=float),
                np.asarray(initial_fit["u"], dtype=float),
            ]
        )

    def objective(parameters):
        theta = parameters[:m] - parameters[:m].mean()
        u = parameters[m:]
        margin = design["sign"] * (theta[design["a"]] - theta[design["b"]])
        loss, dm, du = _row_terms(margin, u[design["r"]], gamma, mode)
        weight = design["weight"]
        value = np.sum(weight * loss)
        value += 0.5 * lambda_theta * theta.dot(theta)
        value += 0.5 * lambda_u * np.square(u - mu).sum()
        gradient = _gradient(weight * dm, weight * du, design)
        gradient[:m] += lambda_theta * theta
        gradient[:m] -= gradient[:m].mean()
        gradient[m:] += lambda_u * (u - mu)
        return float(value), gradient

    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": max_iter, "ftol": tol, "gtol": tol, "maxls": 50},
    )
    if not np.isfinite(result.fun) or not np.all(np.isfinite(result.x)):
        raise RuntimeError(f"RankBalance failed: {result.message}")
    theta = result.x[:m] - result.x[:m].mean()
    u = result.x[m:]
    ranking = pd.DataFrame(
        {"Method": design["models"], "ELO Score": ELO_OFFSET + ELO_SCALE * theta}
    ).sort_values("ELO Score", ascending=False, kind="stable").reset_index(drop=True)
    return {
        "ranking": ranking,
        "theta": theta,
        "u": u,
        "design": design,
        "inverse_hessian": result.hess_inv,
        "config": dict(config),
        "success": bool(result.success),
        "message": str(result.message),
        "elapsed": time.time() - started,
    }


def directional_changes(frame, fit, target_a: str, target_b: str, attack_type: str):
    design = fit["design"]
    model_index = design["model_index"]
    output = np.full(len(frame), np.nan)
    if target_a not in model_index or target_b not in model_index:
        return output
    m = len(design["models"])
    contrast = np.zeros(m + len(design["raters"]))
    contrast[model_index[target_a]] = 1.0
    contrast[model_index[target_b]] = -1.0
    direction = np.asarray(fit["inverse_hessian"].matvec(contrast))
    config = fit["config"]

    def projection(row_signs):
        margin = row_signs * (
            fit["theta"][design["a"]] - fit["theta"][design["b"]]
        )
        _, dm, du = _row_terms(
            margin,
            fit["u"][design["r"]],
            float(config["gamma"]),
            str(config.get("regularizer", "full")),
        )
        return design["weight"] * (
            dm * row_signs * (
                direction[design["a"]] - direction[design["b"]]
            )
            + du * direction[m + design["r"]]
        )

    original = projection(design["sign"])
    if attack_type == "delete":
        output[:] = 0.0
        np.add.at(output, design["source_row"], original)
    else:
        labels = frame["answerValue"].astype(str).str.lower()
        decisive = labels.isin({"a", "b"}).to_numpy()
        flipped = projection(-design["sign"])
        per_source = np.zeros(len(frame), dtype=float)
        np.add.at(per_source, design["source_row"], -(flipped - original))
        output[decisive] = per_source[decisive]
    return output

def select_adaptive_attack(frame, fit, attack_type: str, top_k: int, budget: int):
    ranking = fit["ranking"]["Method"].astype(str).tolist()
    if top_k >= len(ranking):
        return None
    scores = fit["ranking"].set_index("Method")["ELO Score"] / ELO_SCALE
    best = None
    for target_a in ranking[:top_k]:
        for target_b in ranking[top_k:]:
            changes = directional_changes(frame, fit, target_a, target_b, attack_type)
            eligible = np.flatnonzero(np.isfinite(changes))
            if not len(eligible):
                continue
            order = eligible[np.argsort(changes[eligible], kind="stable")[:budget]]
            predicted_gap = float(scores[target_a] - scores[target_b] + changes[order].sum())
            candidate = (predicted_gap, target_a, target_b, order)
            if best is None or candidate[0] < best[0]:
                best = candidate
    if best is None:
        return None
    return {"predicted_gap": best[0], "target_a": best[1], "target_b": best[2], "positions": best[3]}


def apply_attack(frame: pd.DataFrame, attack_type: str, positions: np.ndarray) -> pd.DataFrame:
    selected = np.zeros(len(frame), dtype=bool)
    selected[np.asarray(positions, dtype=int)] = True
    if attack_type == "delete":
        return frame.loc[~selected].reset_index(drop=True)
    result = frame.copy()
    result.loc[selected, "answerValue"] = result.loc[selected, "answerValue"].map({"A": "B", "B": "A"})
    return result.reset_index(drop=True)


def preference_nll(test: pd.DataFrame, fit: Dict[str, Any]) -> Tuple[float, int]:
    design = fit["design"]
    model_index, rater_index = design["model_index"], design["rater_index"]
    ia = test["methodA"].astype(str).map(model_index)
    ib = test["methodB"].astype(str).map(model_index)
    valid = ia.notna() & ib.notna()
    work = test.loc[valid]
    a, b = ia.loc[valid].to_numpy(int), ib.loc[valid].to_numpy(int)
    base = expit(fit["theta"][a] - fit["theta"][b])
    mapped_rater = work["answerer"].astype(str).map(rater_index)
    eta = np.full(len(work), expit(float(fit["config"]["mu"])))
    seen = mapped_rater.notna().to_numpy()
    eta[seen] = expit(fit["u"][mapped_rater.loc[seen].to_numpy(int)])
    probability = eta * base + (1.0 - eta) * (1.0 - base)
    labels = work["answerValue"].astype(str).str.lower()
    target = np.select([labels.eq("a"), labels.eq("b"), labels.isin({"draw", "tie"})], [1.0, 0.0, 0.5], default=np.nan)
    finite = np.isfinite(target)
    probability = np.clip(probability[finite], 1e-12, 1.0 - 1e-12)
    target = target[finite]
    loss = -target * np.log(probability) - (1.0 - target) * np.log(1.0 - probability)
    return float(loss.mean()), int(len(loss))


def decisive_preference_metrics(
    test: pd.DataFrame,
    fit: Dict[str, Any],
) -> Dict[str, float]:
    """Evaluate Bernoulli NLL and accuracy on decisive held-out labels."""
    design = fit["design"]
    model_index = design["model_index"]
    rater_index = design["rater_index"]
    labels = test["answerValue"].astype(str).str.lower()
    index_a = test["methodA"].astype(str).map(model_index)
    index_b = test["methodB"].astype(str).map(model_index)
    valid = labels.isin({"a", "b"}) & index_a.notna() & index_b.notna()
    work = test.loc[valid]
    if work.empty:
        return {"nll": math.nan, "accuracy": math.nan, "rows": 0}

    a = index_a.loc[valid].to_numpy(dtype=int)
    b = index_b.loc[valid].to_numpy(dtype=int)
    base = expit(fit["theta"][a] - fit["theta"][b])
    mapped_rater = work["answerer"].astype(str).map(rater_index)
    eta = np.full(len(work), expit(float(fit["config"]["mu"])))
    seen = mapped_rater.notna().to_numpy()
    eta[seen] = expit(
        fit["u"][mapped_rater.loc[seen].to_numpy(dtype=int)]
    )
    probability = eta * base + (1.0 - eta) * (1.0 - base)
    probability = np.clip(probability, 1e-12, 1.0 - 1e-12)
    target = labels.loc[valid].eq("a").to_numpy(dtype=float)
    loss = -target * np.log(probability) - (1.0 - target) * np.log(
        1.0 - probability
    )
    accuracy = np.mean((probability >= 0.5) == target.astype(bool))
    return {
        "nll": float(loss.mean()),
        "accuracy": float(accuracy),
        "rows": int(len(loss)),
    }
