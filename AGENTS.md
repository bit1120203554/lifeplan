# AGENTS.md — LifePlan

## Project
Personal life planning & task management for lys. Not a code project — no build, test, lint, or package scripts.

## Layout
- Each subdirectory is a self-contained task with its own `README.md`.
- Active tasks: `phd-apply/` (2027fall PhD), `algorithm-competition/` (Aug 2-3 contest), `internship/` (fallback).
- `docs/prompt.md` has the full user background and goals — read it first.

## Key paths
- User memory: `/home/lys/.claude/projects/-home-lys-task-lifeplan/memory/` (user-profile, key-dates, feedback)
- Existing agent instructions: `CLAUDE.md` (18 lines, still relevant)

## Workflow
- Read the subdirectory's `README.md` and linked files before acting.
- Mark completed tasks in the root `README.md` table.
- Consider the user is **very busy** (grad student, paper deadline, PhD prep).
- Keep responses concrete and actionable; avoid generic advice.

## Quirks
- The `.claude/` directory is gitignored — memory files are local only.
- Git history is minimal (single init commit). No CI, no hooks.
