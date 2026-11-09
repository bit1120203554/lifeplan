# AGENTS.md — LifePlan

Not a code project. No build, test, lint, or package scripts.

## User context
- Chinese graduate student (BIT CS, 2027fall PhD applicant), extremely busy.
- Communicate in **Chinese** unless otherwise specified.
- Keep responses concrete and actionable; avoid generic advice.
- `docs/prompt.md` has full background — read it first.

## Historical files
- Put superseded files under the root `legacy/`, mirroring their original repository-relative paths. For example, `internship/1/archive/x` becomes `legacy/internship/1/archive/x`.
- `/legacy/` is Git-ignored and is a local archive, not a Git backup. Preserve archived file contents and update links in active documents when moving files.
- Archive only material clearly superseded or completed; retain current plans, user records, source documents, and reusable materials in their active areas.

## Layout
- Each subdirectory is a self-contained task area with its own `README.md`.
- Active: `phd-apply/` (2027fall PhD), `algorithm-competition/` (Aug 2-3 contest), `internship/` (fallback).
- Each task dir may have task-specific files (study plans, logs, trackers, templates, materials).
- Mark completed tasks in the root `README.md` status table.
- Memory references use `[[user-profile]]`, `[[key-dates]]` wiki-style links (resolved from memory files).

## Memory & state
- User memory (gitignored, local only):
  `/home/lys/.claude/projects/-home-lys-task-lifeplan/memory/` (user-profile, key-dates, feedback)
- `.claude/` is gitignored.
- Git history: single init commit. No CI, no hooks.
- `CLAUDE.md` at root has additional Chinese-language context notes — check it.

## For uncertain claims
State your source and confidence level. When evaluating advisors/mentors, prioritize objective evidence and reputation verification.
