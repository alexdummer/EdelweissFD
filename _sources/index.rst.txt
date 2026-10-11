EdelweissFD
===========

EdelweissFD is a light-weight finite difference framework for **multifield** (coupled
physics) problems. It is the finite difference sibling of `EdelweissFE
<https://github.com/EdelweissFE/EdelweissFE>`_ and reuses its complete solution machinery;
the constitutive models come from `Marmot
<https://github.com/MAteRialMOdelingToolbox/Marmot>`_.

Simulations are set up and executed by **Python scripts**, not by input files.

.. code-block:: python

    from edelweissfd.drivers.pythonscriptedsimulation import FDSimulation
    from edelweissfd.stencils.displacementstencil import DisplacementStencil

    sim = FDSimulation(domainSize=1, name="TensionBar1D")

    grid = sim.createStructuredGrid(lengths=[100.0], nGridPoints=[101])
    material = sim.createMaterial("LinearElastic", [20000.0, 0.2])

    sim.assignStencils(DisplacementStencil, grid, material=material,
                       stressState="uniaxial stress")

    step = sim.createStep(stepLength=1.0)
    step.addDirichlet("clamped", grid.nodeSets["left"], "displacement", {0: 0.0})
    step.addNeumann("pulled", grid, "right", "displacement", [10.0])

    sim.addNodeFieldOutput("displacement", "displacement", "U",
                           nodeSet=grid.nodeSets["all"])

    model, fieldOutputs = sim.run()

Why this is a thin package
--------------------------

EdelweissFE's solution stack is discretization agnostic. Its central abstraction,
:class:`~edelweissfe.nodecouplingentity.base.nodecouplingentity.BaseNodeCouplingEntity`,
describes *anything* that couples nodes -- finite elements, constraints, cells, particles.
A finite difference **stencil** is exactly such an entity: it couples a grid point with its
neighbours, reports which fields it operates on, and returns a residual and a tangent.

Consequently EdelweissFD does not reimplement the degrees of freedom and the sparse
assembly, the Newton-Raphson and arc-length solvers, the linear solvers, the adaptive time
stepping, the boundary conditions, the field output, or the constitutive models. It imports
all of them. What it adds is the discretization: structured grids, finite difference
operators, stencils, and the Python scripting driver.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   documentation/index
