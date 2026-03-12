.PHONY: setup test test-cov lint preprocess run clean status

setup:
	python3.11 -m venv .venv
	.venv/bin/python -m pip install -U pip setuptools wheel
	.venv/bin/python -m pip install -e ".[dev,extra]"
	@echo "Activate: source .venv/bin/activate"

test:
	python -m pytest tests/ -x --tb=short -q

test-cov:
	python -m pytest tests/ --cov=midi_vae --cov-report=html --cov-report=term

lint:
	ruff check midi_vae/ tests/

status:
	@echo "=== Worktrees ===" && git worktree list
	@echo "" && echo "=== Branches ===" && git branch -a --sort=-committerdate | head -15
	@echo "" && echo "=== Main log ===" && git log --oneline -5 main 2>/dev/null || echo "(empty)"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage dist build *.egg-info
