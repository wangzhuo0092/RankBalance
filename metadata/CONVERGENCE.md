# Convergence reporting

Experiment runners save a `fit_status` for every refit. Depending on the
backend, values distinguish converged fits, completed fits whose backend does
not expose a convergence flag, and optimizer warnings. Exceptions are written
to failure or traceback files by the corresponding resumable runner.

Non-convergence is never encoded as a successful defense or a zero ranking
change. Summary scripts retain status counts so a paper table can report the
number of valid fits and warnings alongside each result.

This source-only release contains no convergence outcomes. Running the paper
experiments generates those outcomes under `experiment_results/`.

