# Sync Branch with Main

Rebase the current worktree branch onto the latest main. This keeps your feature branch up to date with other agents' merged work.

Steps:
1. Run `git fetch origin` to get latest changes
2. Run `git stash` if there are uncommitted changes
3. Run `git rebase origin/main` to apply main's changes under your work
4. If stash was created, run `git stash pop`
5. If rebase conflicts occur, show the conflicting files and stop — do NOT auto-resolve. List the conflicts and ask for guidance.
6. After successful rebase, run `git log --oneline -5` to show the current state
