# AGENTS.md

This file provides guidance to AI Agents when working with code in this repository.

## What this is

EdelweissFD is a light-weight finite difference framework for **multifield** (coupled physics)
problems. It is the finite difference sibling of [EdelweissFE](https://github.com/EdelweissFE/EdelweissFE)
and reuses its complete solution machinery (Newton-Raphson/arc-length solvers, DOF management, sparse
assembly, linear solvers, time stepping, boundary conditions, field output); constitutive models come from
[Marmot](https://github.com/MAteRialMOdelingToolbox/Marmot). What EdelweissFD adds is the discretization
itself: structured grids, finite difference operators, stencils, and a Python scripting simulation driver.

Simulations are written as **Python scripts** that build an `FDSimulation` object — there are no input files.

Requires Python >= 3.14 and targets the free-threaded ("nogil") CPython build (`python-freethreading`).

## Setup & Installation

EdelweissFE (including its compiled Marmot material interfaces) must already be installed. Then:

```bash
pip install -e .
```

Optional dependencies for testing and documentation:
```bash
pip install -e ".[test,doc]"
```

## Running tests

The test suite is driven by `pytest`:

```bash
pytest                                          # full suite
pytest -m "not marmot"                          # tier needing no compiled Marmot extensions
pytest tests/test_gradientplasticity.py         # a single file
pytest tests/test_gradientplasticity.py::test_name   # a single test
EDELWEISSFD_NO_COMPILED_KERNEL=1 pytest         # force the pure-Python kernel fallback path

python examples/TensionBar1D/run.py             # run an example script directly
python examples/SimpleBeam2D/run.py             # run 2D beam bending example
```

Every stencil that has a compiled kernel is tested through **both** paths (native Cython kernel and
`EDELWEISSFD_NO_COMPILED_KERNEL=1`), and `test_compiledkernel.py` additionally asserts they agree to
round-off (~1e-11/1e-13 relative) on identical inputs. When modifying a stencil's `computeKernels`, run
the suite both ways.

`tests/regenerate_references.py` regenerates the stored reference arrays under `tests/references/` when
a reconstructed test case's expected results legitimately change.

## Linting, formatting & commit conventions

Formatting and static checks are enforced by pre-commit hooks (`autoflake`, `black --line-length 120`,
`isort`, `flake8`):

```bash
pre-commit run --all-files
```

See [CONTRIBUTING.md](CONTRIBUTING.md#pre-commit-hooks) for hook details, and [CONTRIBUTING.md](CONTRIBUTING.md#conventional-commits) for Conventional Commit types and scopes (`feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `chore`).

See [CONTRIBUTING.md](CONTRIBUTING.md#pull-requests) for PR workflows and branch conventions (`master` for bugfixes, `next_v<YY>.<MM>` for features).

## Workspace Skills (`.agents/skills/`)

Specialized runbooks and checklists for development workflows are available under `.agents/skills/`.
Agents should follow the corresponding `SKILL.md` when tasked with:

| Skill | Description | Entry Point |
| :--- | :--- | :--- |
| **fd-add-stencil** | Implementing, registering, testing, and documenting finite difference stencils (cell-based and GFDM) | [SKILL.md](.agents/skills/fd-add-stencil/SKILL.md) |
| **fd-add-material** | Material configuration, point-wise ownership, free-threading safety & `MaterialProvider` | [SKILL.md](.agents/skills/fd-add-material/SKILL.md) |
| **fd-add-module** | Universal workflow and lifecycle for adding/extending any subsystem in EdelweissFD | [SKILL.md](.agents/skills/fd-add-module/SKILL.md) |
| **fd-create-test** | Creating pytest suites, tangent checks, reconstructed EdelweissFE tests & references | [SKILL.md](.agents/skills/fd-create-test/SKILL.md) |
| **fd-documentation** | Sphinx documentation guide, subsystem `.rst` docs, docstrings, and local builds | [SKILL.md](.agents/skills/fd-documentation/SKILL.md) |
| **fd-code-review** | Quality assurance, static check, free-threading safety, and PR review checklist | [SKILL.md](.agents/skills/fd-code-review/SKILL.md) |

---

## Architecture & Code Organization

### The "thin package" principle

EdelweissFE's solution stack is discretization-agnostic. Its central abstraction, `BaseNodeCouplingEntity`,
describes anything that couples nodes — finite elements, constraints, cells, particles. A finite difference
**stencil** is exactly such an entity, so EdelweissFD imports rather than reimplements:

- Degrees of freedom, sparsity pattern, and sparse assembly (`edelweissfe.numerics`)
- Solvers (`edelweissfe.solvers`)
- Linear solvers (`edelweissfe.linsolve`)
- Time stepping and step management (`edelweissfe.timesteppers`, `edelweissfe.steps`)
- Boundary conditions and loads (`edelweissfe.stepactions`)
- Field output, CSV export, and monitors (`edelweissfe.utils.fieldoutput`, `edelweissfe.outputmanagers`)
- Phenomena & tolerance registry (`edelweissfe.config.phenomena`)
- Constitutive models (`edelweissfe.materials`, native and Marmot)

Keep this split in mind: a missing piece of solver/matrix/output functionality usually already exists on the
EdelweissFE side and should be reused directly.

### The Scripting Driver

`edelweissfd/drivers/pythonscriptedsimulation.py::FDSimulation` builds objects programmatically:
`createStructuredGrid`, `createMaterial`, `assignStencils`/`assignNodalStencils`, `createSolver`,
`createStep`, `addNodeFieldOutput`, then `run()`. `FDStep` wraps `addDirichlet`, `addNodeForces`,
`addNeumann`, `addBodyForce`, and `addIndirectControl` (arc-length control).

### Grids

`edelweissfd/grids/structuredgrid.py::StructuredGrid` is a uniform, structured, **collocated** grid
(every field lives on every grid point) in 1D/2D/3D. Grid points are plain `edelweissfe.points.node.Node`
instances, so all of EdelweissFE's node-set-based machinery applies unchanged. Node set names mirror
EdelweissFE's `planeRectQuad`/`boxGen` generators (`left`/`right`, `bottom`/`top`, `back`/`front`, and
pairwise intersections like `leftBottom`).

### Stencils

`edelweissfd/stencils/base/basestencil.py::BaseStencil` is a `BaseNodeCouplingEntity` + `VIJEntityBase`;
from the solver's point of view a stencil is indistinguishable from a finite element. Implementing one
means implementing `computeKernels(K, P, U, dU, time, dT)` plus descriptive properties:

- DOFs are ordered **node-major, field-minor**.
- `P` is the internal flux in *positive* sense; solvers form $R = -P + P_{\text{ext}}$, so $K = \partial P/\partial U$.
- `K` arrives as a 2D column-major view into the global sparse matrix block and is **not** zeroed by the
  stencil between iterations — the solver zeroes the whole value array itself.
- A stencil owns exactly **one** material point (`getNumberOfQuadraturePoints() == 1`).

Two families of stencils exist:
- **Cell-based** (`DisplacementStencil`, `GradientEnhancedDisplacementStencil`, `GradientPlasticityStencil`):
  The material is evaluated once per cell corner using compact one-sided differences (`cellCornerGradientOperators`)
  rather than wide node-centred quotients, preventing even/odd grid point decoupling and hourglass modes.
  `GradientPlasticityStencil` widens its molecule for the plastic multiplier Laplacian (`grid.laplacianAt`), and
  supports optional Flanagan-Belytschko `hourglassControl="stabilized"` mode.
- **Nodal** (`GFDMGradientPlasticityStencil`): A Generalized Finite Difference (GFDM) scheme assigned via
  `assignNodalStencils`. Fits a weighted-least-squares local Taylor model over a neighbour cloud
  (`edelweissfd/operators/gfdm.py`: `gatherCloud`, `gfdmWeights`) with condition-number singularity checks.

`FDModel` (`edelweissfd/models/fdmodel.py`) is an `FEModel` subclass that holds `grids`; stencils live in
`elements` (aliased as `.stencils`), which is what the solvers iterate over.

### Material Ownership: One Instance per Stencil

A point-wise material carries the state-variable buffer it currently operates on as mutable state
(`assignCurrentStateVars`). Sharing one material instance across stencils corrupts state once the solver
distributes stencils across threads (`edelweissfe.solvers.base.parallelelementcomputation`); on a free-threaded
interpreter this aborts the process outright. `edelweissfd/materials/provider.py::MaterialProvider` mints
a fresh instance per stencil; never pass the same material instance to more than one stencil.

### Compiled Kernel Fallback

Cell-based stencils can delegate their inner loop to EdelweissFE's compiled Cython kernel (e.g.
`GradientPlasticityKernel`) when available, falling back to an equivalent pure-Python `computeKernels`
otherwise. A stencil configuration unsupported by the compiled kernel (e.g. `hourglassControl="stabilized"`)
must make `_createCompiledKernel` return `None` to force the Python path.

### Voigt Convention

Strain/stress vectors throughout EdelweissFD use **Marmot's** Voigt order `(11, 22, 33, 12, 13, 23)` with
engineering shear strains — deliberately *not* `edelweissfe.utils.voigtnotation` (`12, 23, 13`).

### Documentation

Sphinx docs live in `doc/source/documentation/*.rst`, one file per subsystem (`stencils.rst`, `operators.rst`,
`grids.rst`, `drivers.rst`, `materials.rst`, `models.rst`). Build locally with:

```bash
sphinx-build ./doc/source/ ./docs -b html
```
