# Team Status Overview

Show the current state of all agent worktrees and branches.

Steps:
1. Run `git worktree list` to show all active worktrees
2. Run `git branch -a --sort=-committerdate | head -20` to show recent branch activity
3. For each feature branch matching `feature/*`, show the last commit with `git log --oneline -1 <branch>`
4. Run `git log --oneline -10 main` to show recent merges to main
5. Check for any uncommitted changes in the current worktree with `git status --short`
6. Summarize: which roles have active work, which have merged recently, any stale branches (>3 days old)
