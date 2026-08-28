---
name: fd-add-module
description: >-
  Universal workflow and architectural lifecycle for adding or extending any subsystem in EdelweissFD.
  Use when implementing new stencils, differential operators, grid generators, drivers, step actions, or routing to specialized skills.
---

# Module Development Lifecycle in EdelweissFD

Universal guide for adding, modifying, or extending subsystems in EdelweissFD.

---

## 1. Routing to Dedicated Skills

| Task | Skill |
| :--- | :--- |
| Finite difference stencils (cell-based & GFDM) | [`fd-add-stencil`](../fd-add-stencil/SKILL.md) |
| Material models & MaterialProvider | [`fd-add-material`](../fd-add-material/SKILL.md) |
| Pytest suites, tangents & references | [`fd-create-test`](../fd-create-test/SKILL.md) |
| Sphinx documentation & API docs | [`fd-documentation`](../fd-documentation/SKILL.md) |
| QA, static checks & code review | [`fd-code-review`](../fd-code-review/SKILL.md) |

---

## 2. Subsystem Directory Map

| Subsystem | Directory | Description |
| :--- | :--- | :--- |
| Stencils | `edelweissfd/stencils/` | Cell-based and nodal finite difference stencils (`BaseStencil`) |
| Operators | `edelweissfd/operators/` | Finite difference operators (`differences.py`, `gfdm.py`) |
| Grids | `edelweissfd/grids/` | Structured grid representations and node set topologies (`StructuredGrid`) |
| Drivers | `edelweissfd/drivers/` | High-level simulation and step scripting drivers (`FDSimulation`, `FDStep`) |
| Models | `edelweissfd/models/` | Model containers extending EdelweissFE model tree (`FDModel`) |
| Materials | `edelweissfd/materials/` | Material management and isolated instance providers (`MaterialProvider`) |
| Step Actions | `edelweissfd/stepactions/` | Boundary conditions, loads, and traction handlers (`TractionNodalLoad`) |
| Sets | `edelweissfd/sets/` | Stencil sets (`StencilSet`) |

---

## 3. The "Thin Package" Principle

EdelweissFD focuses strictly on the finite difference discretization. Whenever adding functionality:
1. Check if the underlying capability already exists in `EdelweissFE` (`edelweissfe.numerics`, `edelweissfe.solvers`, `edelweissfe.linsolve`, `edelweissfe.timesteppers`, `edelweissfe.steps`, `edelweissfe.stepactions`, `edelweissfe.utils.fieldoutput`, `edelweissfe.outputmanagers`).
2. Reuse existing abstractions directly rather than reimplementing them in EdelweissFD.
3. Keep the scripting API in `FDSimulation` clean, intuitive, and Pythonic.

---

## 4. Implementation Checklist

1. **Architecture**: Extend appropriate base class (`BaseStencil`, `FEModel`, etc.).
2. **Conventions**: Adhere to Marmot Voigt ordering `(11, 22, 33, 12, 13, 23)` and positive flux sign $K = \partial P/\partial U$.
3. **Thread Safety**: Ensure material isolation via `MaterialProvider`.
4. **Dual Execution Path**: For accelerated stencils, provide both compiled Cython kernel and pure-Python fallback.
5. **Testing**: Add pytest test cases in `tests/`, verify tangents in `tests/test_tangents.py`, and test with/without compiled kernels.
6. **Documentation**: Update `.rst` files in `doc/source/documentation/`.
7. **Quality Assurance**: Run `pre-commit run --all-files` and follow [`fd-code-review`](../fd-code-review/SKILL.md).
