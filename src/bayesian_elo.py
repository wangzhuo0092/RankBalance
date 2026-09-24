import math
import logging
from functools import partial
import scipy.special
import scipy.stats
import numpy as np
from models import (
    Metric, QuerySet, ELO_INITIAL_SCORE, ELO_SCALE_FACTOR,
    ELO_MAX_UPDATES, ELO_CONVERGENCE_THRESHOLD,
    GAMMA_PRIOR_SHAPE, GAMMA_PRIOR_RATE,
    skill_to_elo, elo_to_skill,
    resolve_prior_config,
    initialize_optimization_state, optimization_trace_enabled,
    record_optimization_trace,
)
from torch_bayesian_backend import fit_bayes_bt_cuda, resolve_device
import time

logger = logging.getLogger(__name__)

def update_bayesian_elo(
    metric: Metric,
    sessions: QuerySet,
    device: str = 'auto',
    prior_config=None,
):
    """Implements an iterative procedure for computing Elo scores.

    This algorithm is based on Caron & Doucet's (2010) Bayesian interpretation of an algorithm by
    Hunter (2004). However, where Caron & Doucet view it as expectation maximization algorithm for
    computing a MAP estimate, we here go a step further and interpret it as mean-field variational
    inference steps. This allows us to additionally compute uncercainty estimates.

    References:
        Caron & Doucet (2010), Efficient Bayesian Inference for the Bradley-Terry Model
        https://www.stats.ox.ac.uk/~doucet/caron_doucet_bayesianbradleyterry.pdf
    """
    start = time.time()
    # Parameters of a Gamma prior over skill values. The parameter "b" only determines the scale of
    # skill values. It should be of some numerical interest but should otherwise have no influence
    # on resulting Elo scores.
    priors = resolve_prior_config(prior_config)
    a, b = priors['gamma_shape'], priors['gamma_rate']
    metric.state['prior_config'] = priors

    # Elo scores.
    scores = metric.state.get('scores', {})

    # List of methods indicates indices in the win matrix.
    methods = metric.state.get('methods', [])
    method_to_index = dict(zip(methods, range(len(methods))))

    # Pairwise win matrix.
    wins = metric.state.get('wins', [])

    # Makes it so that the mode of the gamma distribution corresponds to an Elo score of
    # ELO_INITIAL_SCORE.
    elo_offset = math.log10((a - 1) / b) * ELO_SCALE_FACTOR
    elo_offset = ELO_INITIAL_SCORE - elo_offset

    s2e = partial(skill_to_elo, offset=elo_offset)
    e2s = partial(elo_to_skill, offset=elo_offset)

    # Collect and update win statistics.
    for session in sessions:
        for slate in session.slates.all():
            ratings = slate.ratings.all()
            count = len(ratings)

            if count != 2:
                logger.error(
                    f'Slate {slate.id} in session {session.id} has {count} ratings instead of 2.'
                )
                continue

            method_i = ratings[0].stimulus.name
            method_j = ratings[1].stimulus.name

            index_i = method_to_index.get(method_i)
            index_j = method_to_index.get(method_j)

            if index_i is None:
                index_i = len(methods)
                methods.append(method_i)
                method_to_index[method_i] = index_i
                for row in wins:
                    row.append(0)
                wins.append([0] * len(methods))

            if index_j is None:
                index_j = len(methods)
                methods.append(method_j)
                method_to_index[method_j] = index_j
                for row in wins:
                    row.append(0)
                wins.append([0] * len(methods))

            s_i = ratings[0].score
            s_j = ratings[1].score

            if s_i > s_j:
                wins[index_i][index_j] += 1
            elif s_j > s_i:
                wins[index_j][index_i] += 1
            else:
                # We don't model ties explicitly. Instead, we consider the expected outcome of a
                # forced choice.
                wins[index_i][index_j] += 0.5
                wins[index_j][index_i] += 0.5

    metric.state['methods'] = methods
    metric.state['wins'] = wins

    n = len(methods)
    win_totals = [0] * n

    for i in range(n):
        for j in range(n):
            win_totals[i] += wins[i][j]

    initialize_optimization_state(metric, methods, scores)
    initial_elo = np.array([
        scores.get(m, {}).get('value', ELO_INITIAL_SCORE) for m in methods
    ])
    record_optimization_trace(metric, 0, methods, initial_elo, wins)

    # Traced runs use CPU because the trace stores every NumPy iteration.
    requested_device = 'cpu' if optimization_trace_enabled(metric) else device
    actual_device = resolve_device(requested_device)
    metric.state['computation_backend'] = actual_device
    if actual_device.startswith('cuda'):
        initial_elo = np.array([
            scores.get(m, {}).get('value', ELO_INITIAL_SCORE) for m in methods
        ])
        cuda_result = fit_bayes_bt_cuda(
            np.asarray(wins, dtype=float),
            e2s(initial_elo),
            a=a,
            b=b,
            max_updates=ELO_MAX_UPDATES * sessions.count(),
            convergence_threshold=ELO_CONVERGENCE_THRESHOLD,
            elo_scale_factor=ELO_SCALE_FACTOR,
            device=actual_device,
        )
        for i, method in enumerate(methods):
            scores[method] = {'value': float(s2e(cuda_result['skill'][i]))}

        for i, method_i in enumerate(methods):
            gamma = scipy.stats.gamma(
                a=cuda_result['posterior_shape'][i],
                scale=1 / cuda_result['posterior_rate'][i],
            )
            percentiles = gamma.ppf(
                [0.005, 0.025, 0.05, 0.5, 0.95, 0.975, 0.995]
            )
            # Preserve the original Bayes-BT field mapping for reproducibility.
            scores[method_i]['p005'] = s2e(percentiles[0])
            scores[method_i]['p025'] = s2e(percentiles[0])
            scores[method_i]['p05'] = s2e(percentiles[1])
            scores[method_i]['median'] = s2e(percentiles[2])
            scores[method_i]['p95'] = s2e(percentiles[3])
            scores[method_i]['p975'] = s2e(percentiles[4])
            scores[method_i]['p995'] = s2e(percentiles[5])
        metric.state['scores'] = scores
        metric.state['iterations'] = cuda_result['iterations']
        metric.state['converged'] = cuda_result['converged']
        return time.time() - start

    # Update scores.
    converged = False
    for iteration in range(ELO_MAX_UPDATES * sessions.count()):
        max_elo_change = 0

        elo_scores = [scores.get(m, {}).get('value', ELO_INITIAL_SCORE) for m in methods]

        for i in range(n):
            elo_i = elo_scores[i]
            skill_i = e2s(elo_i)

            # Parameters of the Gamma distribution modeling the score (shape, rate).
            a_i = a - 1 + win_totals[i]
            b_i = b

            for j in range(n):
                if i != j:
                    skill_j = e2s(elo_scores[j])
                    n_ij = (wins[i][j] + wins[j][i])
                    b_i += n_ij / (skill_i + skill_j)


            # Update Elo score.
            skill_i_new = a_i / b_i
            elo_i_new = s2e(skill_i_new)

            max_elo_change = max(max_elo_change, abs(elo_i_new - elo_i))

            scores[methods[i]] = {'value': elo_i_new}

        # Check for convergence.
        iteration_converged = max_elo_change < ELO_CONVERGENCE_THRESHOLD
        current_elo = np.array([scores[m]['value'] for m in methods])
        record_optimization_trace(
            metric, iteration + 1, methods, current_elo, wins,
            max_elo_change=max_elo_change, converged=iteration_converged,
        )
        if iteration_converged:
            converged = True
            break
    end = time.time()
    delta_time = end - start
    # Estimate uncertainty.
    for i in range(n):
        method_i = methods[i]
        skill_i = e2s(scores.get(method_i, {}).get('value', 1))
        a_i = a - 1 + win_totals[i]

        # Gamma distribution approximating the posterior distribution over the score.
        # The scale is the reciprocal of the rate `b_i = a_i / skill_i`.
        gamma = scipy.stats.gamma(a=a_i, scale=skill_i / a_i)
        percentiles = gamma.ppf([0.005, 0.025, 0.05, 0.5, 0.95, 0.975, 0.995])

        # Convert percentiles to Elo scale.
        scores[method_i]['p005'] = s2e(percentiles[0])
        scores[method_i]['p025'] = s2e(percentiles[0])
        scores[method_i]['p05'] = s2e(percentiles[1])
        scores[method_i]['median'] = s2e(percentiles[2])
        scores[method_i]['p95'] = s2e(percentiles[3])
        scores[method_i]['p975'] = s2e(percentiles[4])
        scores[method_i]['p995'] = s2e(percentiles[5])
    metric.state['scores'] = scores
    metric.state['iterations'] = iteration + 1 if n else 0
    metric.state['converged'] = converged
    return delta_time
