---
name: fd-documentation
description: >-
  Procedure for writing, updating, structuring, and building documentation in EdelweissFD.
  Use when modifying Sphinx docs (.rst files), docstrings, driver APIs, or building local documentation.
---

# Sphinx Documentation Guide for EdelweissFD

All Sphinx documentation source files live under `doc/source/`. Subsystem documentation lives under `doc/source/documentation/*.rst`.

---

## 1. Documentation Structure

```
doc/
└── source/
    ├── conf.py                 # Sphinx configuration
    ├── index.rst               # Documentation home & table of contents
    ├── installation.rst        # Installation guide & prerequisites
    └── documentation/
        ├── index.rst           # Subsystem index
        ├── drivers.rst         # Scripted simulation API (FDSimulation, FDStep)
        ├── grids.rst           # Grid generators & node set topology
        ├── materials.rst       # MaterialProvider & Marmot integration
        ├── models.rst          # FDModel container
        ├── operators.rst       # Difference & GFDM operators
        └── stencils.rst        # Finite difference stencils
```

---

## 2. Building Documentation Locally

```bash
sphinx-build ./doc/source/ ./docs -b html
```

Always ensure the documentation builds cleanly without warnings or broken cross-references.

---

## 3. Formatting Guidelines

### Subsystem Documentation Template (`.rst`)
```rst
Stencil Name
------------

Description of the mathematical model and discrete equations.

Governing Equations:

.. math::
   \boldsymbol{\nabla} \cdot \boldsymbol{\sigma} + \mathbf{b} = \mathbf{0}

Parameters:
- ``param1``: Description of parameter (:math:`\text{unit}`).
- ``param2``: Description of parameter.

Usage in Python Script:

.. code-block:: python

   sim = FDSimulation(domainSize=2, name="MySim")
   sim.assignStencils(MyStencil, grid, material=material)
```

### Docstring Conventions
- Use **NumPy-style docstrings** (`Parameters`, `Returns`, `Yields`, `Raises`, `Notes`, `Examples`).
- Document all public classes, methods, properties, and module-level functions.
- Specify array shapes and coordinate/field conventions in docstrings.
