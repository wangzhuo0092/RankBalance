"""Optional PyTorch/CUDA kernels for the Bayesian ranking methods.

The public functions in this module accept and return NumPy arrays. PyTorch is
imported lazily, so the existing CPU-only environment does not depend on it.
All CUDA calculations use float64 to stay close to the NumPy implementations.
"""

from __future__ import annotations

from functools import lru_cache
import math
from typing import Any, Dict, Optional

import numpy as np


@lru_cache(maxsize=None)
def _load_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for CUDA execution. Install a CUDA-enabled "
            "PyTorch build or use --device cpu/auto."
        ) from exc
    return torch


@lru_cache(maxsize=None)
def resolve_device(requested: str = "auto") -> str:
    """Resolve auto/cpu/cuda without making PyTorch a CPU-side dependency."""
    requested = str(requested or "auto").strip().lower()
    if requested == "cpu":
        return "cpu"
    if requested != "auto" and not requested.startswith("cuda"):
        raise ValueError("device must be 'auto', 'cpu', 'cuda', or 'cuda:N'")

    try:
        torch = _load_torch()
    except RuntimeError:
        if requested == "auto":
            return "cpu"
        raise

    if not torch.cuda.is_available():
        if requested == "auto":
            return "cpu"
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    if requested.startswith("cuda:"):
        index = int(requested.split(":", 1)[1])
        if index < 0 or index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device {index} does not exist; found {torch.cuda.device_count()} device(s)"
            )
    return "cuda" if requested == "auto" else requested


def cuda_metadata(device: str) -> Dict[str, Any]:
    """Return lightweight backend metadata for result files."""
    actual = resolve_device(device)
    metadata: Dict[str, Any] = {"backend": actual}
    if actual.startswith("cuda"):
        torch = _load_torch()
        index = torch.device(actual).index
        if index is None:
            index = torch.cuda.current_device()
        metadata.update(
            {
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "cuda_device": index,
                "cuda_device_name": torch.cuda.get_device_name(index),
            }
        )
    return metadata


def _tensor(values, torch, device):
    return torch.as_tensor(values, dtype=torch.float64, device=device)


def _numpy(tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _elo_change(old_skill, new_skill, scale_factor, torch):
    return scale_factor * torch.max(
        torch.abs(torch.log10(new_skill) - torch.log10(old_skill))
    )


def fit_bayes_bt_cuda(
    wins: np.ndarray,
    initial_skill: np.ndarray,
    *,
    a: float,
    b: float,
    max_updates: int,
    convergence_threshold: float,
    elo_scale_factor: float,
    device: str,
) -> Dict[str, Any]:
    """Run the Bayes-BT MAP updates on CUDA."""
    torch = _load_torch()
    with torch.no_grad():
        wins_t = _tensor(wins, torch, device)
        skill = _tensor(initial_skill, torch, device)
        win_totals = wins_t.sum(dim=1)
        pair_totals = wins_t + wins_t.transpose(0, 1)
        shape = (a - 1) + win_totals
        converged = False

        for iteration in range(max_updates):
            rate = b + (
                pair_totals / (skill[:, None] + skill[None, :])
            ).sum(dim=1)
            new_skill = shape / rate
            change = _elo_change(skill, new_skill, elo_scale_factor, torch)
            skill = new_skill
            if float(change.item()) < convergence_threshold:
                converged = True
                break

        # The original CPU uncertainty code uses rate = shape / final skill.
        posterior_rate = shape / skill
        return {
            "skill": _numpy(skill),
            "posterior_shape": _numpy(shape),
            "posterior_rate": _numpy(posterior_rate),
            "iterations": iteration + 1,
            "converged": converged,
        }


def _original_bbq_expectations(wins, skill, quality, torch):
    ratios = skill[:, None] / (skill[:, None] + skill[None, :])
    quality_expanded = quality[None, None, :]
    bt_probability = quality_expanded * ratios[:, :, None]
    gamma = bt_probability / (bt_probability + (1 - quality_expanded) / 2)
    off_diagonal = (~torch.eye(
        len(skill), dtype=torch.bool, device=skill.device
    ))[:, :, None]
    return gamma * off_diagonal


def fit_original_bbq_cuda(
    wins: np.ndarray,
    initial_skill: np.ndarray,
    initial_quality: np.ndarray,
    *,
    a: float,
    b: float,
    alpha: float,
    beta: float,
    max_updates: int,
    convergence_threshold: float,
    elo_scale_factor: float,
    device: str,
) -> Dict[str, Any]:
    """Run the original BBQ random-response EM updates on CUDA."""
    torch = _load_torch()
    with torch.no_grad():
        wins_t = _tensor(wins, torch, device)
        skill = _tensor(initial_skill, torch, device)
        quality = _tensor(initial_quality, torch, device)
        n = len(skill)
        upper_i, upper_j = torch.triu_indices(n, n, offset=1, device=device)
        quality_denominator = alpha + beta - 2 + (
            wins_t[upper_i, upper_j, :] + wins_t[upper_j, upper_i, :]
        ).sum(dim=0)
        converged = False

        for iteration in range(max_updates):
            gamma = _original_bbq_expectations(wins_t, skill, quality, torch)
            effective = wins_t * gamma
            shape = (a - 1) + effective.sum(dim=(1, 2))
            rate = b + (
                (effective + effective.transpose(0, 1))
                / (skill[:, None, None] + skill[None, :, None])
            ).sum(dim=(1, 2))
            new_skill = shape / rate

            quality_numerator = alpha - 1 + (
                effective[upper_i, upper_j, :]
                + effective[upper_j, upper_i, :]
            ).sum(dim=0)
            new_quality = quality_numerator / quality_denominator
            change = _elo_change(skill, new_skill, elo_scale_factor, torch)
            skill, quality = new_skill, new_quality
            if float(change.item()) < convergence_threshold:
                converged = True
                break

        gamma = _original_bbq_expectations(wins_t, skill, quality, torch)
        effective = wins_t * gamma
        shape = (a - 1) + effective.sum(dim=(1, 2))
        rate = b + (
            (effective + effective.transpose(0, 1))
            / (skill[:, None, None] + skill[None, :, None])
        ).sum(dim=(1, 2))
        return {
            "skill": _numpy(skill),
            "quality": _numpy(quality),
            "posterior_shape": _numpy(shape),
            "posterior_rate": _numpy(rate),
            "iterations": iteration + 1,
            "converged": converged,
        }


def _correctness_expectations(
    wins, skill, eta, reverse_update, torch, eps
):
    n, _, rater_count = wins.shape
    expected = torch.zeros_like(wins)
    if n < 2 or rater_count == 0:
        return expected, None

    i_idx, j_idx = torch.triu_indices(n, n, offset=1, device=wins.device)
    obs_pos = wins[i_idx, j_idx, :]
    obs_neg = wins[j_idx, i_idx, :]
    pi = skill[i_idx] / (skill[i_idx] + skill[j_idx])
    pi = pi[:, None]
    eta_row = eta[None, :]
    prob_pos = torch.clamp(
        pi * eta_row + (1 - pi) * (1 - eta_row), min=eps, max=1.0
    )
    prob_neg = torch.clamp(
        (1 - pi) * eta_row + pi * (1 - eta_row), min=eps, max=1.0
    )
    correct_pos = pi * eta_row / prob_pos
    correct_neg = (1 - pi) * eta_row / prob_neg
    incorrect_pos = (1 - pi) * (1 - eta_row) / prob_pos
    incorrect_neg = pi * (1 - eta_row) / prob_neg

    expected[i_idx, j_idx, :] = obs_pos * correct_pos
    expected[j_idx, i_idx, :] = obs_neg * correct_neg
    if reverse_update:
        expected[i_idx, j_idx, :] += obs_neg * incorrect_neg
        expected[j_idx, i_idx, :] += obs_pos * incorrect_pos
    quality_stats = {
        "correct": obs_pos * correct_pos + obs_neg * correct_neg,
        "total": obs_pos + obs_neg,
    }
    return expected, quality_stats


def _weighted_quantile(values, weights, quantile, torch):
    values = values.reshape(-1)
    weights = weights.reshape(-1)
    valid = torch.isfinite(values) & torch.isfinite(weights) & (weights > 0)
    if not bool(torch.any(valid).item()):
        return None
    values = values[valid]
    weights = weights[valid]
    order = torch.argsort(values)
    values = values[order]
    weights = weights[order]
    target = quantile * weights.sum()
    index = torch.searchsorted(torch.cumsum(weights, dim=0), target, right=False)
    index = torch.clamp(index, max=len(values) - 1)
    return values[index]


def _adaptive_expectations(
    wins,
    skill,
    eta,
    reverse_update,
    torch,
    *,
    eps,
    max_contamination,
    temperature,
    decay_power,
):
    n, _, rater_count = wins.shape
    expected = torch.zeros_like(wins)
    empty = {
        "contamination_rate_raw": 0.0,
        "contamination_rate": 0.0,
        "threshold_probability": None,
        "threshold_margin": None,
        "mean_clipping_weight": 1.0,
    }
    if n < 2 or rater_count == 0:
        return expected, None, empty

    i_idx, j_idx = torch.triu_indices(n, n, offset=1, device=wins.device)
    obs_pos = wins[i_idx, j_idx, :]
    obs_neg = wins[j_idx, i_idx, :]
    total = obs_pos.sum() + obs_neg.sum()
    if float(total.item()) <= 0:
        return expected, None, empty

    pi = skill[i_idx] / (skill[i_idx] + skill[j_idx])
    pi = torch.clamp(pi[:, None], min=eps, max=1 - eps)
    eta_row = eta[None, :]
    prob_pos = torch.clamp(
        pi * eta_row + (1 - pi) * (1 - eta_row), min=eps, max=1.0
    )
    prob_neg = torch.clamp(
        (1 - pi) * eta_row + pi * (1 - eta_row), min=eps, max=1.0
    )
    correct_pos = pi * eta_row / prob_pos
    correct_neg = (1 - pi) * eta_row / prob_neg
    error_mass = (
        (obs_pos * (1 - correct_pos)).sum()
        + (obs_neg * (1 - correct_neg)).sum()
    )
    contamination_raw_t = error_mass / total
    contamination_t = torch.clamp(
        contamination_raw_t, min=0.0, max=max_contamination
    )
    score_pos = torch.broadcast_to(pi, obs_pos.shape)
    score_neg = torch.broadcast_to(1 - pi, obs_neg.shape)
    threshold = _weighted_quantile(
        torch.cat([score_pos.reshape(-1), score_neg.reshape(-1)]),
        torch.cat([obs_pos.reshape(-1), obs_neg.reshape(-1)]),
        contamination_t,
        torch,
    )
    if threshold is None:
        return expected, None, empty
    threshold = torch.clamp(threshold, min=eps, max=1 - eps)
    threshold_margin = torch.log(threshold / (1 - threshold))
    margin_pos = torch.broadcast_to(torch.log(pi / (1 - pi)), obs_pos.shape)
    margin_neg = -margin_pos
    clip_pos = torch.exp(
        -(torch.clamp(threshold_margin - margin_pos, min=0.0) / temperature)
        ** decay_power
    )
    clip_neg = torch.exp(
        -(torch.clamp(threshold_margin - margin_neg, min=0.0) / temperature)
        ** decay_power
    )
    expected[i_idx, j_idx, :] = obs_pos * clip_pos
    expected[j_idx, i_idx, :] = obs_neg * clip_neg
    if reverse_update:
        expected[i_idx, j_idx, :] += obs_neg * (1 - clip_neg)
        expected[j_idx, i_idx, :] += obs_pos * (1 - clip_pos)

    diagnostics = {
        "contamination_rate_raw": float(contamination_raw_t.item()),
        "contamination_rate": float(contamination_t.item()),
        "threshold_probability": float(threshold.item()),
        "threshold_margin": float(threshold_margin.item()),
        "mean_clipping_weight": float(
            ((obs_pos * clip_pos).sum() + (obs_neg * clip_neg).sum()).item()
            / total.item()
        ),
    }
    quality_stats = {
        "correct": obs_pos * correct_pos + obs_neg * correct_neg,
        "total": obs_pos + obs_neg,
    }
    return expected, quality_stats, diagnostics


def fit_correctness_bbq_cuda(
    wins: np.ndarray,
    decisive_wins: np.ndarray,
    tie_wins: np.ndarray,
    initial_skill: np.ndarray,
    initial_eta: np.ndarray,
    *,
    reverse_update: bool,
    weighting_mode: str,
    a: float,
    b: float,
    alpha: float,
    beta: float,
    max_updates: int,
    convergence_threshold: float,
    elo_scale_factor: float,
    adaptive_warmup_updates: int,
    adaptive_max_contamination: float,
    adaptive_temperature: float,
    adaptive_decay_power: float,
    eps: float,
    device: str,
) -> Dict[str, Any]:
    """Run Correctness/Adaptive BBQ updates on CUDA."""
    torch = _load_torch()
    with torch.no_grad():
        wins_t = _tensor(wins, torch, device)
        decisive_t = _tensor(decisive_wins, torch, device)
        ties_t = _tensor(tie_wins, torch, device)
        skill = _tensor(initial_skill, torch, device)
        eta = _tensor(initial_eta, torch, device)
        is_adaptive = weighting_mode == "adaptive"
        previous_contamination: Optional[float] = None
        diagnostics = None
        converged = False

        for iteration in range(max_updates):
            if is_adaptive and iteration < adaptive_warmup_updates:
                expected = wins_t.clone()
                quality_stats = None
            elif is_adaptive:
                expected, quality_stats, diagnostics = _adaptive_expectations(
                    decisive_t,
                    skill,
                    eta,
                    reverse_update,
                    torch,
                    eps=eps,
                    max_contamination=adaptive_max_contamination,
                    temperature=adaptive_temperature,
                    decay_power=adaptive_decay_power,
                )
                expected = expected + ties_t
            else:
                expected, quality_stats = _correctness_expectations(
                    decisive_t, skill, eta, reverse_update, torch, eps
                )
                expected = expected + ties_t

            shape = (a - 1) + expected.sum(dim=(1, 2))
            rate = b + (
                (expected + expected.transpose(0, 1))
                / (skill[:, None, None] + skill[None, :, None])
            ).sum(dim=(1, 2))
            new_skill = shape / rate
            change = _elo_change(skill, new_skill, elo_scale_factor, torch)
            skill = new_skill

            if quality_stats is not None:
                eta = (
                    alpha - 1 + quality_stats["correct"].sum(dim=0)
                ) / (
                    alpha + beta - 2 + quality_stats["total"].sum(dim=0)
                )
                eta = torch.clamp(eta, min=eps, max=1 - eps)

            if is_adaptive and diagnostics is not None:
                contamination = diagnostics["contamination_rate"]
                contamination_change = (
                    abs(contamination - previous_contamination)
                    if previous_contamination is not None
                    else math.inf
                )
                previous_contamination = contamination
                if (
                    float(change.item()) < convergence_threshold
                    and contamination_change < 1e-4
                ):
                    converged = True
                    break
            elif not is_adaptive and float(change.item()) < convergence_threshold:
                converged = True
                break

        if is_adaptive:
            expected, _, diagnostics = _adaptive_expectations(
                decisive_t,
                skill,
                eta,
                reverse_update,
                torch,
                eps=eps,
                max_contamination=adaptive_max_contamination,
                temperature=adaptive_temperature,
                decay_power=adaptive_decay_power,
            )
        else:
            expected, _ = _correctness_expectations(
                decisive_t, skill, eta, reverse_update, torch, eps
            )
        expected = expected + ties_t
        shape = (a - 1) + expected.sum(dim=(1, 2))
        rate = b + (
            (expected + expected.transpose(0, 1))
            / (skill[:, None, None] + skill[None, :, None])
        ).sum(dim=(1, 2))
        return {
            "skill": _numpy(skill),
            "quality": _numpy(eta),
            "posterior_shape": _numpy(shape),
            "posterior_rate": _numpy(rate),
            "adaptive_diagnostics": diagnostics,
            "iterations": iteration + 1,
            "converged": converged,
        }
