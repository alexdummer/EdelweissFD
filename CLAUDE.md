# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

EdelweissFD is a light-weight finite difference framework for multifield (coupled physics)
problems. It is the finite difference sibling of
[EdelweissFE](https://github.com/EdelweissFE/EdelweissFE) and reuses its complete solution
machinery (Newton-Raphson/arc-length solvers, DOF management, sparse assembly, linear
solvers, time stepping, boundary conditions, field output); constitutive models come from
[Marmot](https://github.com/MAteRialMOdelingToolbox/Marmot). What EdelweissFD adds is the
discretization itself: structured grids, finite difference operators, and stencils.

Simulations are plain **Python scripts** that build an `FDSimulation` object — there are no
input files.

## Setup

EdelweissFE (including its compiled Marmot material interfaces) must already be installed.
Then:

```bash
pip install -e .
```

Both EdelweissFD and EdelweissFE currently require Python >= 3.14.

## Common commands

```bash
pytest                                          # full suite
pytest -m "not marmot"                          # tier needing no compiled Marmot extensions
pytest tests/test_gradientplasticity.py         # a single file
pytest tests/test_gradientplasticity.py::test_name   # a single test
EDELWEISSFD_NO_COMPILED_KERNEL=1 pytest         # force the pure-Python kernel fallback path

python examples/TensionBar1D/run.py             # run an example script directly
python examples/SimpleBeam2D/run.py

pre-commit run --all-files                      # black, isort, autoflake, flake8, yaml/whitespace checks
```

Every stencil that has a compiled kernel is tested through **both** paths (native and
`EDELWEISSFD_NO_COMPILED_KERNEL=1`), and `test_compiledkernel.py` additionally asserts they
agree to round-off (~1e-11/1e-13 relative) on identical inputs. When touching a stencil's
`computeKernels`, run the suite both ways.

`tests/regenerate_references.py` regenerates the stored reference arrays the tests compare
against, for when a reconstructed EdelweissFE test case's expected numbers legitimately
change.

## Architecture

### The "thin package" principle

EdelweissFE's solution stack is discretization-agnostic. Its central abstraction,
`BaseNodeCouplingEntity`, describes anything that couples nodes — finite elements,
constraints, cells, particles. A finite difference **stencil** is exactly such an entity, so
EdelweissFD imports rather than reimplements: DOF management and sparse assembly
(`edelweissfe.numerics`), solvers (`edelweissfe.solvers`), linear solvers
(`edelweissfe.linsolve`), time stepping (`edelweissfe.timesteppers`, `edelweissfe.steps`),
boundary conditions/loads (`edelweissfe.stepactions`), field output
(`edelweissfe.utils.fieldoutput`, `edelweissfe.outputmanagers`), the phenomena/tolerance
registry (`edelweissfe.config.phenomena`), and constitutive models
(`edelweissfe.materials`, native and Marmot). Keep this split in mind: a missing piece of
functionality usually already exists on the EdelweissFE side and should be reused, not
reimplemented here.

### The scripting driver

`edelweissfd/drivers/pythonscriptedsimulation.py::FDSimulation` mirrors EdelweissFE's
input-file-driven simulation, but builds objects directly: `createStructuredGrid`,
`createMaterial`, `assignStencils`/`assignNodalStencils`, `createSolver`, `createStep`,
`addNodeFieldOutput`, then `run()`. `FDStep` wraps `addDirichlet`/`addNodeForces`/
`addNeumann`/`addBodyForce`/`addIndirectControl` (arc-length control).

### Grids

`edelweissfd/grids/structuredgrid.py::StructuredGrid` is a uniform, structured, **collocated**
grid (every field lives on every grid point) in 1D/2D/3D. Grid points are plain
`edelweissfe.points.node.Node` instances, so all of EdelweissFE's node-set-based machinery
applies unchanged. Node set names mirror EdelweissFE's `planeRectQuad`/`boxGen` generators
(`left`/`right`, `bottom`/`top`, `back`/`front`, and pairwise intersections like
`leftBottom`), so boundary conditions read the same in both packages.

### Stencils

`edelweissfd/stencils/base/basestencil.py::BaseStencil` is a `BaseNodeCouplingEntity` +
`VIJEntityBase`; from the solver's point of view a stencil is indistinguishable from a
finite element. Implementing one means implementing `computeKernels(K, P, U, dU, time, dT)`
plus a few descriptive properties. Conventions that matter everywhere:

- DOFs are ordered **node-major, field-minor**.
- `P` is the internal flux in *positive* sense; solvers form `R = -P + P_ext`, so
  `K = ∂P/∂U`.
- `K` arrives as a view into the global sparse matrix block and is **not** zeroed by the
  stencil between iterations — the solver zeroes the whole value array itself.

Two families of stencils exist:

- **Cell-based** (one stencil per grid cell): `DisplacementStencil`,
  `GradientEnhancedDisplacementStencil`, `GradientPlasticityStencil`. The material is
  evaluated once per cell corner using a compact one-sided quotient
  (`cellCornerGradientOperators`) rather than the wide node-centred quotient, because the
  latter decouples even/odd grid points and admits checkerboard (hourglass) modes.
  `GradientPlasticityStencil` additionally needs the plastic multiplier's Laplacian, which
  widens its molecule one grid point beyond the cell (`setCell` → `grid.laplacianAt`), and
  supports an optional Flanagan-Belytschko `hourglassControl="stabilized"` mode as an
  alternative to corner sampling (2D only, off by default).
- **Nodal** (one stencil per grid *node*, no cell — assigned via `assignNodalStencils`
  instead of `assignStencils`): `GFDMGradientPlasticityStencil`, a Generalized Finite
  Difference (GFDM) scheme. It fits a weighted-least-squares local Taylor model over a
  neighbour cloud (`edelweissfd/operators/gfdm.py`: `gatherCloud`, `gfdmWeights`) rather
  than differencing corners, at the cost of a wider, denser molecule. `gfdmWeights` guards
  against silently rank-deficient neighbour clouds via an explicit condition-number check.

`FDModel` (`edelweissfd/models/fdmodel.py`) is an `FEModel` subclass that additionally holds
`grids`; stencils live in `elements` (aliased as `.stencils`), matching what the solvers
iterate over.

### Material ownership: one instance per stencil, never shared

A point-wise material is not a pure function — it carries the state-variable storage it
currently operates on as mutable state (`assignCurrentStateVars`). Sharing one material
instance across stencils corrupts state once the solver distributes stencils across threads
(`edelweissfe.solvers.base.parallelelementcomputation`); on a free-threaded interpreter this
aborts the process outright. `edelweissfd/materials/provider.py::MaterialProvider` mints a
fresh instance per stencil (`assignStencils`/`assignNodalStencils` do this automatically when
given a callable or provider); never pass the same material instance to more than one
stencil.

### Compiled kernel fallback

Cell-based stencils can delegate their inner loop to EdelweissFE's compiled Cython kernel
(e.g. `GradientPlasticityKernel`) when it's available, falling back to an equivalent
pure-Python `computeKernels` otherwise. Controlled by `useCompiledKernel` / the
`EDELWEISSFD_NO_COMPILED_KERNEL` env var. A stencil configuration that the compiled kernel
doesn't support (e.g. `hourglassControl="stabilized"`, or a stress state whose material
routine the kernel can't drive consistently) must make `_createCompiledKernel` return `None`
to force the Python path — see `GradientPlasticityStencil._createCompiledKernel` for the
pattern and its documented exceptions.

### Voigt convention

Strain/stress vectors throughout EdelweissFD use **Marmot's** Voigt order
`(11, 22, 33, 12, 13, 23)` with engineering shear strains — deliberately *not*
`edelweissfe.utils.voigtnotation`, whose shear order is `12, 23, 13`.

### Test suite: reconstructed EdelweissFE regression cases

The suite is built by reconstructing EdelweissFE regression tests as finite difference
problems, so the two discretizations can be compared on the same physics (uniaxial
stress/plane strain/plane stress/3D, shear, bending, J2 plasticity, AT2 phase field
fracture, body forces, displacement and indirect/arc-length control). See the table in
`README.md` for which EdelweissFD test reconstructs which EdelweissFE `testfiles/` case.
`tests/test_tangents.py`/`test_operators.py` check analytic tangents against numerical ones
and operator accuracy directly, independent of any reconstructed case.

### CI

Two independent test workflows: `run_tests_with_marmot.yml` builds the full dependency chain
(Eigen → autodiff → Fastor → Marmot → EdelweissFE → EdelweissFD) and runs the whole suite
both with and without the compiled kernel; `run_tests_without_marmot.yml` is the fast
feedback loop that skips everything needing compiled Marmot extensions
(`pytest -m "not marmot"`). Both clone EdelweissFE/Marmot at `github.base_ref` (the PR's
target branch), falling back to `next_v26.11` — a branch name mismatch is treated as a hard
error rather than silently building against an incompatible sibling revision. `format.yml`
runs the pre-commit hooks.
