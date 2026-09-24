PYTHON ?= python
WORKERS ?= 8

.PHONY: test gamma ablation utility synthetic clean-results

test:
	$(PYTHON) -m unittest tests/test_rank_balance.py

gamma:
	$(PYTHON) experiments/tune_rank_balance_gamma.py

ablation:
	$(PYTHON) experiments/run_rank_balance_ablation.py
	$(PYTHON) experiments/run_component_ablation_utility.py

utility:
	$(PYTHON) experiments/evaluate_rank_balance_tuned_utility.py
	$(PYTHON) experiments/run_paper_baseline_utility.py

synthetic:
	$(PYTHON) experiments/run_synthetic_truth_attack.py --workers $(WORKERS)

clean-results:
	@echo "Generated outputs are under experiment_results/. Remove them manually if intended."
