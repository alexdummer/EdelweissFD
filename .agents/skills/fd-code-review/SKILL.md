---
name: fd-code-review
description: >-
  Quality assurance, static check, and architectural review checklist for EdelweissFD changes and pull requests.
  Use when reviewing code, checking pre-commit compliance, inspecting free-threading safety, verifying dual-path execution, or preparing PR submissions.
---

# Code Review & QA Checklist for EdelweissFD

## 1. Material Ownership & Free-Threading Safety
- [ ] **Material Isolation**: Stencils must never share material instances across entities or threads. Use `MaterialProvider`.
- [ ] **Minimal State Variables**: Only strictly path-dependent historical variables in `stateVars` (no instantaneous/elastic quantities).
- [ ] **Voigt Order**: Marmot Voigt ordering `(11, 22, 33, 12, 13, 23)` with engineering shear strain $\gamma_{ij} = 2\varepsilon_{ij}$ used consistently.

## 2. Stencils & Dual-Path Verification
- [ ] **Flux & Tangent Sign**: Internal flux $P$ in positive sense; $K = \partial P/\partial U$.
- [ ] **Sparse Matrix View**: $K$ treated as Fortran-order view into sparse block without manual zeroing between iterations.
- [ ] **Dual-Path Parity**: When a compiled Cython kernel is supported, both compiled and pure-Python paths (`EDELWEISSFD_NO_COMPILED_KERNEL=1`) give identical results within round-off ($< 10^{-11}$).
- [ ] **Graceful Fallback**: `_createCompiledKernel()` returns `None` for unsupported configurations rather than failing.

## 3. Formatting, Linting & Hygiene
```bash
pre-commit run --all-files
```
- [ ] **No Stray Agent Artifacts**: No comments or notes referencing agent chats, plans, prompts, or temp instructions.
- [ ] **No Debug Print/Dumps**: Clean up temporary print statements and scratch test artifacts.
- [ ] **Clean Code Standards**: Enforce `black --line-length 120`, `isort`, `autoflake`, and `flake8`.

## 4. Tests & Documentation
- [ ] **Tests Pass**: `pytest` and `pytest -m "not marmot"` both pass cleanly.
- [ ] **Analytic Tangents**: All new stencils verified against central differences in `tests/test_tangents.py`.
- [ ] **Sphinx Documentation**: `sphinx-build ./doc/source/ ./docs -b html` builds without warnings.

## 5. Maintenance & PR Targeting
- [ ] **Agent Guidance**: Update `AGENTS.md` and `.agents/skills/` if workflows or conventions evolve.
- [ ] **Conventional Commits**: `<type>(<scope>): <summary>` format (e.g. `feat(stencils): add ...`, `fix(operators): ...`).
- [ ] **PR Target**: Match the target branch conventions of the repository (`master` for bug fixes, `next_v<YY>.<MM>` for features).
