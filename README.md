# RankBalance

RankBalance is a reliability-aware pairwise ranking model with
operation-specific regularization for ranking stability under comparison
deletion and label reversal. This repository contains the implementation,
paper experiment runners, configurations, and the dataset files used by the
included configurations.

This source release intentionally contains **no computed results**. All CSV
summaries, checkpoints, logs, tables, and figures must be regenerated under
`experiment_results/`, which is ignored by Git except for `.gitkeep`.

## Layout

```text
.
├── src/                         # RankBalance and baseline implementations
├── experiments/                 # Paper-facing experiment entry points
├── configs/                     # Paper configurations
├── metadata/                    # Rater mappings, dataset statistics, and settings
├── projects/*/data/             # Input comparison datasets
├── tests/                       # Core numerical test
└── experiment_results/          # Empty generated-output directory
```

## Installation

```bash
conda env create -f environment.yml
conda activate rank
```

Some baseline experiments use the optional Google ELO executable:

```bash
bash src/google_elo/compile.sh
```

The build script downloads pinned third-party C++ dependencies. RankBalance
itself is implemented in Python/SciPy and does not require this executable.

## Tests

```bash
python -m unittest tests/test_rank_balance.py
```

## Main reproduction workflow

Limit nested BLAS parallelism when using process workers:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

Select one global perturbation weight using decisive validation NLL:

```bash
python experiments/tune_rank_balance_gamma.py
```

Run the matched ablation and held-out utility experiments:

```bash
python experiments/run_rank_balance_ablation.py
python experiments/evaluate_rank_balance_tuned_utility.py
python experiments/run_paper_baseline_utility.py
```

Run method-specific Delete/Flip attacks, followed by fixed-budget transfer
attacks. These experiments are resumable and write only to
`experiment_results/`.

```bash
python experiments/run_victim_specific_robustness.py --workers 8
python experiments/run_cross_attack_transfer.py --condition-workers 8
```

Run the known-truth synthetic check:

```bash
python experiments/run_synthetic_truth_attack.py --workers 8
```

## Hyperparameter selection

The candidate grid is

```text
0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1
```

For each candidate, validation NLL is first averaged over seeds within each
dataset. The selected value is the largest gamma whose paired dataset-level
NLL increase relative to the NLL-minimizing candidate does not exceed one
standard error of that paired increase. The released source contains no
precomputed selected value. See
[`metadata/hyperparameter_search.yaml`](metadata/hyperparameter_search.yaml)
for the complete search space and selection rule.

## Tie handling

A tie is expanded into two opposite binary outcomes with weight 0.5 each.
The same weights enter the likelihood and perturbation regularizer. Attack
budgets count original comparison records; the two expanded rows are always
edited together.

## Data and licensing

See [DATASETS.md](DATASETS.md) for included paths and basic provenance notes.
The datasets originate from third-party projects and may retain their original
terms; consult the corresponding upstream sources before reuse or
redistribution.

The source code is distributed under the terms in [LICENSE](LICENSE).

Dataset/rater mappings, dataset-size summaries, random seeds, and optimizer
settings are under the [metadata directory](metadata/). Experiment runners
record convergence status when executed; this source-only repository does not
include outcomes.

## Reproducibility notes

- The standard split is seeded record-level 60/20/20, with rows moved back to
  training only when needed to preserve model coverage.
- Failed and non-converged fits must be reported; they are not attack failures.
- Source-conditioned attacks use common edit sequences across victim methods.
  Victim-specific attacks recompute scores from each method at reported budget
  points.
- RankBalance runners read the validation-selected gamma generated at runtime;
  the source release does not prescribe or bundle a selected value.
