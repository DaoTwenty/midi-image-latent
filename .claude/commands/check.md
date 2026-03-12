# Run Tests and Checks

Run the full test suite and code quality checks.

Steps:
1. Run `python -m pytest tests/ -x --tb=short -q` and report results
2. If tests fail, show the first failure with full traceback
3. Run `python -c "from midi_vae.config import load_config; print('Config OK')"` to verify config loads
4. Run `python -c "from midi_vae.registry import ComponentRegistry; print('Registry:', ComponentRegistry.list_all())"` to verify registry
5. Report: total tests, passed, failed, and any import errors
