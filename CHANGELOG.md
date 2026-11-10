# Changelog

All notable changes to EdelweissFD. Releases are tagged `v<YY>.<MM>`, in step with
[EdelweissFE](https://github.com/Edelweiss-Numerics/EdelweissFE) and
[Marmot](https://github.com/MAteRialMOdelingToolbox/Marmot), and the commits follow
[Conventional Commits](https://www.conventionalcommits.org).

## v26.11

The first release: a finite difference framework for multifield problems that reuses the
complete solution machinery of EdelweissFE and the constitutive models of Marmot.

### Requirements

- Python >= 3.14, preferably the free-threaded build.
- EdelweissFE v26.11 and, for the Marmot materials and the compiled kernels, Marmot v26.11.
  The dependency is pinned to `EdelweissFE ~= 26.11`.

### Features

- **Scripting driver.** `FDSimulation` builds and runs a simulation from a Python script:
  `createStructuredGrid`, `createNodeSet`, `createMaterial`, `assignStencils`,
  `createSolver`, `createStep`, `addNodeFieldOutput`, `addStencilFieldOutput`,
  `addOutputManager` and `run`. `FDStep` adds Dirichlet conditions, nodal forces, tractions
  (`addNeumann`), body forces, solver options and indirect (arc-length) control.
- **Structured, collocated grids** in 1D, 2D and 3D. The grid points are EdelweissFE nodes, and
  the node sets are named like those of EdelweissFE's `planeRectQuad` and `boxGen` generators.
- **Cell-based stencils.** They evaluate the material at the cell corners with compact
  one-sided differences, so even and odd grid points don't decouple and hourglass modes don't
  form:
  - `DisplacementStencil`: balance of linear momentum in uniaxial stress, plane strain, plane
    stress and 3D, with optional B-bar volumetric averaging against locking.
  - `GradientEnhancedDisplacementStencil`: displacement plus a nonlocal field, for example for
    AT2 phase field fracture.
- **Traction loads** (`TractionNodalLoad`), which convert a traction on a boundary plane into
  consistent nodal forces.
- **Materials** from EdelweissFE, native or from Marmot. Every stencil owns its material
  instance (`MaterialProvider`), so EdelweissFE's parallel solvers can run stencils on
  several threads of a free-threaded interpreter.
- **Optional compiled Cython kernels** for both stencils. If a kernel isn't built, the stencil
  falls back to an equivalent pure-Python implementation, and `EDELWEISSFD_NO_COMPILED_KERNEL=1`
  forces that fallback. The test suite runs both ways and checks that they agree to round-off.
- **Numerical tangent check** (`numericaltangent`) for verifying a stencil's analytic tangent.
- **Examples**: `TensionBar1D`, `SimpleBeam2D` and `AT2PhaseFieldBar`.

### Tests

Regression tests of EdelweissFE reconstructed as finite difference problems: tension bar,
fixed displacement shear, simple beam, von Mises plasticity, 3D tension and wall shear with
body forces, plane stress and the AT2 phase field bar. Further tests cover tangents, operators,
compiled kernels, B-bar averaging, material ownership and output managers. The tests marked
`marmot` are skipped when the Marmot material interfaces are not compiled.

### Known limitations

- **Gradient plasticity is not included.** Its material family is not part of EdelweissFE and
  Marmot v26.11. The cell-based and GFDM gradient plasticity stencils are kept on the
  `feat/gradient-plasticity` branch.
- **Not compatible with EdelweissFE after v26.11.** EdelweissFE's `master` changed the sign
  convention of element body forces (Edelweiss-Numerics/EdelweissFE#259), so body forces act
  in the wrong direction against it. The next release adapts to it.
- **Source distribution only.** The compiled kernels link against the user's own Marmot
  install, so there are no prebuilt wheels.
