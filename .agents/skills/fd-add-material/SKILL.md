---
name: fd-add-material
description: >-
  Procedure for implementing, configuring, and assigning constitutive material models in EdelweissFD.
  Use when connecting Marmot or native EdelweissFE materials, managing point-wise material ownership, and ensuring free-threaded safety via MaterialProvider.
---

# Material Models & MaterialProvider in EdelweissFD

EdelweissFD relies on EdelweissFE and Marmot for constitutive material laws (linear elastic, J2 von Mises plasticity, AT2 phase field fracture, etc.).

---

## 1. The Material Ownership Principle (Thread & Free-Threading Safety)

A point-wise material is not a stateless pure function. It stores the state-variable buffer it operates on as mutable state (`assignCurrentStateVars`).

- **Single Point at a Time**: An instance can only serve one material point at a time.
- **Multithreading / Free-Threading**: Solvers (`edelweissfe.solvers.base.parallelelementcomputation`) distribute stencil chunks across threads. If two threads point the same material instance to their state variables, memory corruption occurs. Under free-threaded Python (`python-freethreading`), the Cython memoryview acquisition count is corrupted, aborting with `Fatal Python error: Acquisition count is -1`.
- **Solution**: **One unique material instance per stencil**. `MaterialProvider` mints a fresh instance per stencil.

---

## 2. Using `MaterialProvider`

In `edelweissfd/materials/provider.py`:

```python
from edelweissfd.materials.provider import MaterialProvider

# Create provider from material class and parameters
provider = MaterialProvider(LinearElastic, [20000.0, 0.2])

# Or create via FDSimulation helper
material = sim.createMaterial("LinearElastic", [20000.0, 0.2])
```

When passing `material` to `sim.assignStencils(StencilClass, grid, material=material)` or `sim.assignNodalStencils(...)`, the simulation automatically wraps callables / providers and instantiates an isolated material per stencil.

---

## 3. Voigt Convention

All material models and stencils in EdelweissFD follow **Marmot's Voigt ordering**:
- Order: `(11, 22, 33, 12, 13, 23)`
- Engineering shear strain: $\gamma_{ij} = 2\varepsilon_{ij}$
- *Caution*: Do not use `edelweissfe.utils.voigtnotation`, which uses `(11, 22, 33, 12, 23, 13)`.

---

## 4. Minimal State Variables Requirement

When defining or wrapping materials:
- Store **only strictly path-dependent historical quantities** in `stateVars` (e.g. plastic strain, back-stress, damage history).
- Never store instantaneous, purely elastic, or algebraically reconstructible variables in `stateVars` unless explicitly requested.

---

## 5. Verification Checklist

1. **Material Isolation Test**: Ensure `tests/test_materialownership.py` passes.
2. **Stress States**: Validate against 1D uniaxial stress, 2D plane strain, 2D plane stress, and 3D states.
3. **Marmot Tier**: Decorate Marmot-dependent material tests with `@pytest.mark.marmot`.
