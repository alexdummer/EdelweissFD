# EdelweissFD

EdelweissFD is a light-weight finite difference framework for **multifield** (coupled
physics) problems. It is the finite difference sibling of
[EdelweissFE](https://github.com/EdelweissFE/EdelweissFE) and reuses its complete
solution machinery; the constitutive models come from
[Marmot](https://github.com/MAteRialMOdelingToolbox/Marmot).

Simulations are set up and executed by **Python scripts**, not by input files.

## Why this is a thin package

EdelweissFE's solution stack is discretization agnostic. Its central abstraction,
`BaseNodeCouplingEntity`, describes *anything* that couples nodes — finite elements,
constraints, cells, particles. A finite difference **stencil** is exactly such an entity:
it couples a grid point with its neighbours, reports which fields it operates on, and
returns a residual and a tangent.

Consequently EdelweissFD does not reimplement any of the following, it imports it:

| Concern | Provided by |
| --- | --- |
| Degrees of freedom, sparsity pattern, sparse assembly | `edelweissfe.numerics` (`DofManager`, `DofVector`, `VIJSystemMatrix`, `CSRGenerator`) |
| Newton-Raphson, convergence checks, arc-length, explicit dynamics | `edelweissfe.solvers` |
| Linear solvers (PARDISO, KLU, MUMPS, AMGCL, SuperLU, …) | `edelweissfe.linsolve` |
| Adaptive time stepping, steps, cutbacks | `edelweissfe.timesteppers`, `edelweissfe.steps` |
| Boundary conditions and loads | `edelweissfe.stepactions` |
| Field output, monitors, status files, csv export | `edelweissfe.utils.fieldoutput`, `edelweissfe.outputmanagers` |
| The registry of physical fields and their tolerances | `edelweissfe.config.phenomena` |
| Constitutive models | `edelweissfe.materials` (native and Marmot) |

What EdelweissFD adds is the discretization: structured grids, finite difference
operators, stencils, and a Python scripting driver.

## Installation

EdelweissFE (including its compiled Marmot material interfaces) must be installed first.

```bash
pip install -e .
```

## Usage

```python
from edelweissfd.drivers.pythonscriptedsimulation import FDSimulation
from edelweissfd.stencils.displacementstencil import DisplacementStencil

sim = FDSimulation(domainSize=1, name="TensionBar1D")

grid = sim.createStructuredGrid(lengths=[100.0], nGridPoints=[101])
material = sim.createMaterial("LinearElastic", [20000.0, 0.2])

sim.assignStencils(DisplacementStencil, grid, material=material)

step = sim.createStep(stepLength=1.0)
step.addDirichlet("clamped", grid.nodeSets["left"], "displacement", {0: 0.0})
step.addNeumann("pulled", grid, "right", "displacement", [10.0])

sim.addNodeFieldOutput("displacement", "displacement", "U", nodeSet=grid.nodeSets["all"])

model, fieldOutputs = sim.run()
```

Run it as any Python script:

```bash
python examples/TensionBar1D/run.py
```

## Tests

```bash
pytest
```

Tests which require the compiled Marmot material interfaces are marked `marmot` and are
skipped automatically if those extensions are unavailable, mirroring the split of the
EdelweissFE regression suite into `testfiles/edelweiss-only` and `testfiles/marmot`:

```bash
pytest -m "not marmot"      # the tier that needs no Marmot at all
```

### Reconstructed EdelweissFE test cases

The suite is built by reconstructing regression tests of EdelweissFE as finite difference
problems, so that the two discretizations can be compared on the same physics.

| EdelweissFD test | reconstructed from | what it adds |
| --- | --- | --- |
| `test_tensionbar.py` | `edelweiss-only/TensionBarQuad4`, `marmot/LinearElasticIsotropic` | uniaxial stress, plane strain; exact vs. closed form |
| `test_fixeddisplacement.py` | `edelweiss-only/FixedDisplacementQuad4` | **shear**, purely displacement driven, native material |
| `test_simplebeam.py` | `edelweiss-only/SimpleBeamQuad4` | **bending**, convergence to beam theory, tractions |
| `test_vonmisesplasticity.py` | `edelweiss-only/VonMises` | **plasticity**, state variables, multiple steps, unloading |
| `test_threedimensional.py` | `edelweiss-only/TensionBarHexa8`, `WallShearHexa8` | **3D**, `boxGen` grids, **body forces** |
| `test_planestress.py` | `marmot/CPS4` | **plane stress**, out-of-plane condensation |
| `test_at2phasefieldbar.py` | `marmot/AT2PhaseField` | **multifield**, phase field fracture, arc-length control |
| `test_tangents.py`, `test_operators.py` | — | analytic vs. numerical tangents, operator accuracy |

Coverage that follows from it:

- dimensions 1D, 2D and 3D
- stress states uniaxial stress, plane strain, plane stress and 3D
- deformation modes stretch, shear and bending
- materials linear elastic, J2 plasticity and AT2 phase field, from both providers
- loads prescribed displacements, nodal forces, tractions and body forces
- single and multiple steps, displacement control and indirect (arc-length) control

## License

LGPL-2.1, see `LICENSE.md`.
