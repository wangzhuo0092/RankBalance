import math
import numpy as np

# Constants
ELO_INITIAL_SCORE = 2000
ELO_SCALE_FACTOR = 400
ELO_MAX_UPDATES = 1000
ELO_CONVERGENCE_THRESHOLD = 1
MIN_COMPARISONS_PER_USER = 1  # Minimum number of comparisons required per user 答数少于这个数的人会被丢掉。现在设成 1，等于基本不过滤

# Gamma prior parameters for skill values
GAMMA_PRIOR_SHAPE = 5  # a
GAMMA_PRIOR_RATE = 0.1   # b

BETA_PRIOR_ALPHA = 10
BETA_PRIOR_BETA = 2 

PRIOR_PARAMETER_NAMES = (
    'gamma_shape',
    'gamma_rate',
    'beta_alpha',
    'beta_beta',
)


def resolve_prior_config(prior_config=None):
    """Return validated Bayesian prior parameters without mutating globals.

    Normal experiment paths pass ``None`` and retain the original BBQ defaults.
    Prior-sensitivity experiments pass a mapping so multiple configurations can
    be fitted safely in the same Python process.
    """
    resolved = {
        'gamma_shape': float(GAMMA_PRIOR_SHAPE),
        'gamma_rate': float(GAMMA_PRIOR_RATE),
        'beta_alpha': float(BETA_PRIOR_ALPHA),
        'beta_beta': float(BETA_PRIOR_BETA),
    }
    if prior_config is not None:
        unknown = sorted(set(prior_config) - set(PRIOR_PARAMETER_NAMES))
        if unknown:
            raise ValueError(f'Unknown prior parameters: {unknown}')
        resolved.update({key: float(value) for key, value in prior_config.items()})

    if not np.isfinite(resolved['gamma_shape']) or resolved['gamma_shape'] <= 1:
        raise ValueError('gamma_shape must be finite and greater than 1')
    if not np.isfinite(resolved['gamma_rate']) or resolved['gamma_rate'] <= 0:
        raise ValueError('gamma_rate must be finite and positive')
    # The current CPU/CUDA updates use the Beta MAP equations with alpha - 1
    # and beta - 1 pseudo-counts, so both parameters must exceed one.
    for name in ('beta_alpha', 'beta_beta'):
        if not np.isfinite(resolved[name]) or resolved[name] <= 1:
            raise ValueError(f'{name} must be finite and greater than 1')
    return resolved

# Utility functions
def skill_to_elo(skill, offset=0):
    return (np.log(skill) / np.log(10)) * ELO_SCALE_FACTOR + offset

def elo_to_skill(elo, offset=0):
    return np.power(10, (elo - offset) / ELO_SCALE_FACTOR)


def optimization_trace_enabled(metric):
    """Return whether an experiment requested per-iteration diagnostics."""
    config = metric.state.get('_optimization_trace_config', {})
    return bool(config.get('enabled', False))


def initialize_optimization_state(
    metric,
    methods,
    scores,
    raters=None,
    qualities=None,
    default_quality=0.5,
):
    """Apply deterministic random initialization for convergence experiments.

    Normal runs do not set ``_optimization_trace_config`` and therefore retain
    the algorithms' original initialization exactly.
    """
    config = metric.state.get('_optimization_trace_config', {})
    if not config.get('enabled') or not config.get('random_initialization', True):
        return

    rng = np.random.default_rng(int(config.get('seed', 0)))
    elo_std = float(config.get('initial_elo_std', 100.0))
    if methods and not scores:
        offsets = rng.normal(0.0, elo_std, size=len(methods))
        offsets -= offsets.mean()
        for method, offset in zip(methods, offsets):
            scores[method] = {'value': float(ELO_INITIAL_SCORE + offset)}

    if raters is None or qualities is None or not raters or qualities:
        return

    quality_std = float(config.get('initial_quality_std', 0.05))
    initial = np.clip(
        rng.normal(default_quality, quality_std, size=len(raters)),
        1e-6,
        1.0 - 1e-6,
    )
    for rater, value in zip(raters, initial):
        qualities[rater] = {'value': float(value)}


def model_only_nll(elo_scores, wins):
    """Return mean BT negative log-likelihood for a win-count matrix."""
    elo_scores = np.asarray(elo_scores, dtype=float)
    wins = np.asarray(wins, dtype=float)
    if wins.ndim == 3:
        wins = wins.sum(axis=2)
    if wins.size == 0 or wins.sum() <= 0:
        return np.nan

    logits = (
        np.log(10.0)
        * (elo_scores[:, np.newaxis] - elo_scores[np.newaxis, :])
        / ELO_SCALE_FACTOR
    )
    total_loss = np.sum(wins * np.logaddexp(0.0, -logits))
    return float(total_loss / wins.sum())


def record_optimization_trace(
    metric,
    iteration,
    methods,
    elo_scores,
    wins,
    max_elo_change=np.nan,
    objective=np.nan,
    converged=False,
):
    """Append one compact optimization snapshot when tracing is enabled."""
    config = metric.state.get('_optimization_trace_config', {})
    if not config.get('enabled'):
        return

    interval = max(1, int(config.get('record_every', 1)))
    if iteration != 0 and not converged and iteration % interval:
        return

    elo_scores = np.asarray(elo_scores, dtype=float)
    order = np.argsort(-elo_scores, kind='stable')
    metric.state.setdefault('optimization_trace', []).append({
        'iteration': int(iteration),
        'model_only_nll': model_only_nll(elo_scores, wins),
        'objective': float(objective) if np.isfinite(objective) else np.nan,
        'max_elo_change': (
            float(max_elo_change) if np.isfinite(max_elo_change) else np.nan
        ),
        'converged': bool(converged),
        'ranking': [str(methods[index]) for index in order],
        'scores': {
            str(method): float(score)
            for method, score in zip(methods, elo_scores)
        },
    })


# Data models
class QuerySet:
    def __init__(self, items):
        self._items = items
        
    def all(self):
        return self._items
        
    def count(self):
        return len(self._items)
        
    def __iter__(self):
        return iter(self._items)

class Metric:
    def __init__(self):
        self.state = {
            'scores': {},
            'qualities': {},
            'methods': [],
            'raters': [],
            'wins': []
        }

class Session:
    def __init__(self, slates, rater=None):
        self._slates = slates
        self.id = 0  # Dummy ID since we don't need it
        self.rater = rater
        
    @property
    def slates(self):
        return QuerySet(self._slates)

class Slate:
    def __init__(self, ratings):
        self._ratings = ratings
        self.id = 0  # Dummy ID since we don't need it
        
    @property
    def ratings(self):
        return QuerySet(self._ratings)

class Rating:
    def __init__(self, score, stimulus):
        self.score = score
        self.stimulus = stimulus

class Stimulus:
    def __init__(self, name):
        self.name = name 
