# Kickoff Sprint Work

## Variables
ROLE: $ARGUMENTS

Dispatch work to a specific agent role by preparing their worktree and providing their sprint task.

Steps:
1. Identify the role from ROLE (alpha, bravo, charlie, delta, or echo)
2. Read the agent's task brief from `specs/task_ROLE.md` (if it exists) or from `.claude/agents/ROLE.md`
3. Check the current sprint by looking at what's already on main (`git log --oneline -10 main`)
4. Based on what's already merged, determine the next deliverable for this role
5. Create or verify the worktree exists: `git worktree list | grep ROLE`
6. Print a clear task assignment with:
   - What to build next (specific files and classes)
   - What interfaces to implement against (specific ABC methods)
   - Acceptance criteria (specific tests that should pass)
   - Estimated scope (number of files, approximate lines)
