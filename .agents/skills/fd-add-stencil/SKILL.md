---
name: fd-add-stencil
description: >-
  Procedure for implementing, registering, testing, and documenting a new finite difference stencil (cell-based or nodal/GFDM) in EdelweissFD.
  Use when creating continuum, gradient-enhanced, or multi-field stencils, adding compiled kernel acceleration, or implementing GFDM stencils.
---

# Implementing a Finite Difference Stencil in EdelweissFD

A stencil is a `BaseNodeCouplingEntity` (`edelweissfe.nodecouplingentity.base.nodecouplingentity.BaseNodeCouplingEntity`) and `VIJEntityBase` (`edelweissfe.numerics.vijentitybase.VIJEntityBase`). From the solver's point of view, a stencil is indistinguishable from a finite element: the `DofManager` derives global DOFs and CSR sparsity from the stencil's nodes and fields, and the solvers evaluate it via `computeKernels(K, P, U, dU, time, dT)`.

---

## 1. Core Mathematical & Solvers Conventions

1. **DOF Ordering**: Ordered **node-major, field-minor** (e.g. for displacement in 2D: $u_{x0}, u_{y0}, u_{x1}, u_{y1}, \dots$; for coupled fields: $u_{x0}, u_{y0}, p_0, \dots$).
2. **Residual & Flux Sign**: $P$ is the internal flux in **positive** sense. Solvers form residual $R = -P + P_{\text{ext}}$, so the tangent is $K = \frac{\partial P}{\partial U}$.
3. **Sparse Tangent View**: $K$ arrives as a 2D column-major (Fortran-order) slice `(nDof, nDof)` into the global sparse matrix block. It is **not** zeroed by the stencil between iterations — the solver zeroes the global matrix.
4. **Single Material Point Ownership**: A stencil owns exactly **one** material point. `getNumberOfQuadraturePoints()` returns `1`, and `getResultArray(result, quadraturePoint=0)` only accepts `quadraturePoint=0`.
5. **Voigt Order**: All stress/strain tensors use **Marmot's** Voigt convention: `(11, 22, 33, 12, 13, 23)` with engineering shear strains ($\gamma_{ij} = 2\varepsilon_{ij}$).

---

## 2. Stencil Architecture: Cell-Based vs. Nodal (GFDM)

### A. Cell-Based Stencils (`DisplacementStencil`, `GradientEnhancedDisplacementStencil`, `GradientPlasticityStencil`)
- **Corner Sampling**: Materials are evaluated at cell corners using compact one-sided differences (`cellCornerGradientOperators`), avoiding even/odd decoupling and checkerboard/hourglass modes.
- **Molecule Widening**: First-derivative stencils couple exactly the $2^d$ corner nodes of the cell via `setCell(grid, cellIndex)`. Stencils needing second derivatives (e.g. Laplacian in `GradientPlasticityStencil`) widen their molecule via `grid.laplacianAt` during `setCell`.
- **Hourglass Control**: Optional Flanagan-Belytschko stabilization (`hourglassControl="stabilized"`) for 2D cell-centred formulations.

### B. Nodal Stencils (`GFDMGradientPlasticityStencil`)
- **Generalized Finite Differences**: One stencil per grid node (no cell mesh). Fitted via weighted least-squares local Taylor expansion over a star/cloud of neighbouring nodes (`gatherCloud`, `gfdmWeights`).
- **Singularity Protection**: `gfdmWeights` validates the condition number of the moment matrix $A^T W A$ to protect against rank-deficient neighbour clouds.
- **Assignment**: Assigned using `sim.assignNodalStencils(...)` instead of `sim.assignStencils(...)`.

---

## 3. Implementation Skeleton

Create `edelweissfd/stencils/<name>stencil.py`:

```python
import numpy as np
from edelweissfe.points.node import Node

from edelweissfd.materials.provider import MaterialProvider
from edelweissfd.operators.differences import (
    cellCornerGradientOperators,
    cellCornerOffsets,
)
from edelweissfd.stencils.base.basestencil import BaseStencil


class MyNewStencil(BaseStencil):
    """Description of the stencil formulation."""

    def __init__(self, stencilNumber: int, spacings: np.ndarray, **options):
        super().__init__()
        self._stencilNumber = int(stencilNumber)
        self.spacings = np.asarray(spacings, dtype=float)
        self.domainSize = len(self.spacings)
        self.material = None
        self._options = options

    @property
    def stencilNumber(self) -> int:
        return self._stencilNumber

    @property
    def hasMaterial(self) -> bool:
        return self.material is not None

    def setMaterial(self, material):
        """Assign the isolated material instance for this stencil."""
        self.material = material

    @property
    def cornerOffsets(self) -> np.ndarray:
        return cellCornerOffsets(self.domainSize)

    def initializeStencil(self):
        """Precompute gradient operators, volume weights, and allocate local buffers."""
        self.gradientOperators = cellCornerGradientOperators(self.spacings)
        self.volume = np.prod(self.spacings)
        # Register DOF fields on nodes (e.g. 'displacement', 'plasticMultiplier')
        ...

    def computeKernels(
        self,
        K: np.ndarray,
        P: np.ndarray,
        U: np.ndarray,
        dU: np.ndarray,
        time: float,
        dT: float,
    ):
        """Evaluate internal flux P and tangent K."""
        # 1. Compute strain / gradients from U and dU
        # 2. Evaluate material response at point
        # 3. Accumulate P and K
        ...

    def acceptLastState(self):
        if self.material is not None and hasattr(self.material, "acceptLastState"):
            self.material.acceptLastState()

    def resetToLastValidState(self):
        if self.material is not None and hasattr(self.material, "resetToLastValidState"):
            self.material.resetToLastValidState()

    def getResultArray(self, result: str, quadraturePoint: int = 0, getPersistentView: bool = True) -> np.ndarray:
        if quadraturePoint != 0:
            raise ValueError(f"A stencil owns exactly one point, got quadraturePoint={quadraturePoint}")
        # Return requested field/result from material or stencil state
        ...

    def getCoordinatesAtCenter(self) -> np.ndarray:
        return np.mean([n.coordinates for n in self.nodes], axis=0)
```

---

## 4. Compiled Kernel Acceleration & Fallback

When a fast Cython kernel is available in `EdelweissFE` (e.g. `GradientPlasticityKernel`):
1. **Delegation**: If `useCompiledKernel=True` and available, delegate `computeKernels` to the compiled kernel.
2. **Fallback**: Always provide the full pure-Python evaluation path when `EDELWEISSFD_NO_COMPILED_KERNEL=1` or compiled kernel is unavailable.
3. **Unsupported Modes**: If a feature or stress state is unsupported by the compiled kernel, `_createCompiledKernel()` must return `None` to gracefully force the Python fallback path without failing.
4. **Verification**: Verify both paths agree to numerical round-off ($< 10^{-11}$) via `test_compiledkernel.py`.

---

## 5. Testing & Verification Checklist

1. **Analytic vs. Numerical Tangents**: Add test in `tests/test_tangents.py` comparing analytic $K$ against central finite-difference perturbations of $P$.
2. **Material Ownership**: Ensure stencil works with `MaterialProvider` (no shared material instances across stencils).
3. **Dual-Path Agreement**: If compiled kernel acceleration is implemented, add parity tests in `tests/test_compiledkernel.py`.
4. **Regression Physics**: Reconstruct corresponding EdelweissFE test cases in `tests/test_<physics>.py`.
5. **Documentation**: Document stencil in `doc/source/documentation/stencils.rst`.
