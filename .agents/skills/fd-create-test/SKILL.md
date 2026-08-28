---
name: fd-create-test
description: >-
  Step-by-step instructions for constructing, validating, and maintaining pytest test suites and reference solutions in EdelweissFD.
  Use when adding new test cases, reconstructing EdelweissFE benchmarks, validating tangents, or updating reference results.
---

# Creating and Managing Tests in EdelweissFD

The test suite validates finite difference discretizations by reconstructing corresponding EdelweissFE regression tests, checking analytic tangents against central numerical differences, and verifying compiled kernel parity with the pure-Python fallback.

---

## 1. Test Categories & Organization

| Category | File | Description |
| :--- | :--- | :--- |
| **Reconstructed FE Tests** | `test_tensionbar.py`, `test_fixeddisplacement.py`, `test_simplebeam.py`, `test_vonmisesplasticity.py`, `test_threedimensional.py`, `test_planestress.py`, `test_at2phasefieldbar.py` | Full simulation tests comparing FD solutions against closed-form or EdelweissFE FE reference solutions. |
| **Tangent Verification** | `test_tangents.py` | Compares analytic $K = \partial P/\partial U$ against central finite-difference perturbations of $P$. |
| **Operator Tests** | `test_operators.py`, `test_laplacian.py`, `test_gfdm.py` | Direct mathematical accuracy tests for differential operators. |
| **Kernel Parity** | `test_compiledkernel.py` | Asserts compiled Cython kernel and pure-Python fallback agree to round-off ($< 10^{-11}$). |
| **Material Ownership** | `test_materialownership.py` | Asserts thread-safe isolation of material instances across stencils. |
| **Convergence & Mechanics** | `test_widthconvergence.py`, `test_hourglassstabilization.py`, `test_volumetricaveraging.py`, `test_fischerburmeistersmoothing.py` | Tests for localization, mesh independence, stabilization, and smoothing. |

---

## 2. Pytest Execution Commands

```bash
# Run complete test suite
pytest

# Run tests that do not require compiled Marmot extensions
pytest -m "not marmot"

# Run a specific test file or test function
pytest tests/test_gradientplasticity.py
pytest tests/test_gradientplasticity.py::test_name

# Force the pure-Python fallback path across all stencils
EDELWEISSFD_NO_COMPILED_KERNEL=1 pytest

# Run example scripts directly
python examples/TensionBar1D/run.py
python examples/SimpleBeam2D/run.py
```

---

## 3. Test Structure Template (Scripted Simulation)

```python
import numpy as np
import pytest

from edelweissfd.drivers.pythonscriptedsimulation import FDSimulation
from edelweissfd.stencils.displacementstencil import DisplacementStencil


@pytest.mark.marmot  # Add mark if Marmot material is used
def test_my_feature():
    sim = FDSimulation(domainSize=2, name="MyTest")
    grid = sim.createStructuredGrid(lengths=[10.0, 10.0], nGridPoints=[5, 5])
    material = sim.createMaterial("LinearElastic", [20000.0, 0.2])

    sim.assignStencils(DisplacementStencil, grid, material=material)

    step = sim.createStep(stepLength=1.0)
    step.addDirichlet("fixed_bottom", grid.nodeSets["bottom"], "displacement", {1: 0.0})
    step.addDirichlet("fixed_left",   grid.nodeSets["bottom"], "displacement", {0: 0.0})
    step.addDirichlet("pull_top",     grid.nodeSets["top"],    "displacement", {1: 0.1})

    sim.addNodeFieldOutput("displacement", "displacement", "U", nodeSet=grid.nodeSets["all"])

    model, fieldOutputs = sim.run()

    # Assert expected solution
    U = fieldOutputs["displacement"].values
    assert np.all(np.isfinite(U))
```

---

## 4. Stored References & Regeneration

For tests that compare against stored reference arrays:
- Reference files are stored in `tests/references/`.
- If an expected solution legitimately changes due to an intentional algorithmic or formulation update, run:
  ```bash
  python tests/regenerate_references.py
  ```
- Always inspect the diff of generated reference files before committing.
