# Prepare Pull Request

Prepare the current worktree branch for a PR to main. Validate everything before push.

Steps:
1. Run `git diff --stat origin/main` to show what changed
2. Run `pytest -x --tb=short` to verify tests pass. If tests fail, stop and fix them.
3. Run `python -c "import midi_vae"` to verify the package imports cleanly
4. Check that no files outside your role's ownership were modified by comparing the changed file list against the module ownership table in CLAUDE.md
5. Ensure all new public classes/methods have docstrings by grepping for `def ` and `class ` without preceding docstrings
6. Show a summary: files changed, tests passed, any warnings
7. If everything passes, run `git push origin HEAD` and report the branch name for PR creation
