#!/usr/bin/env python3
"""
A linear elastic tension bar, solved by finite differences.

This is the finite difference counterpart of the EdelweissFE regression tests
``testfiles/edelweiss-only/TensionBarQuad4`` and ``testfiles/marmot/LinearElasticIsotropic``:
a bar, clamped at one end, loaded by a traction at the other. It is the simplest complete
EdelweissFD simulation and has a closed form solution,

    sigma = t,      u(x) = t * x / E

which the script checks against.

Run it with::

    python run.py
"""

import numpy as np

from edelweissfd.drivers.pythonscriptedsimulation import FDSimulation
from edelweissfd.stencils.displacementstencil import DisplacementStencil

youngsModulus = 20000.0
poissonsRatio = 0.2

length = 100.0
nGridPoints = 101

traction = 10.0


def run(nGridPoints: int = nGridPoints, verbose: bool = True):
    """Solve the tension bar and return the model and the field outputs.

    Parameters
    ----------
    nGridPoints
        The number of grid points along the bar.
    verbose
        Be verbose during the simulation.

    Returns
    -------
    tuple
        The model tree and the field output controller.
    """

    sim = FDSimulation(domainSize=1, name="TensionBar1D", verbose=verbose)

    grid = sim.createStructuredGrid(lengths=[length], nGridPoints=[nGridPoints])

    material = sim.createMaterial("LinearElastic", [youngsModulus, poissonsRatio])

    sim.assignStencils(
        DisplacementStencil,
        grid,
        material=material,
        stressState="uniaxial stress",
    )

    step = sim.createStep(stepLength=1.0, maxInc=1.0, minInc=1e-4, maxNumInc=100, maxIter=25)
    step.addDirichlet("clamped", grid.nodeSets["left"], "displacement", {0: 0.0})
    step.addNeumann("pulled", grid, "right", "displacement", [traction])

    sim.addNodeFieldOutput("displacement", "displacement", "U", nodeSet=grid.nodeSets["all"])
    sim.addNodeFieldOutput("reactionForce", "displacement", "P", nodeSet=grid.nodeSets["left"])
    sim.addStencilFieldOutput("stress", "stress")

    return sim.run()


def analyticalDisplacement(coordinates: np.ndarray) -> np.ndarray:
    """The closed form displacement of the bar.

    Parameters
    ----------
    coordinates
        The axial coordinates.

    Returns
    -------
    np.ndarray
        The axial displacement.
    """

    return traction * coordinates / youngsModulus


def main():
    model, fieldOutputs = run()

    x = np.array([node.coordinates[0] for node in model.nodeFields["displacement"].nodes])
    u = fieldOutputs.fieldOutputs["displacement"].getLastResult().flatten()

    uAnalytical = analyticalDisplacement(x)

    error = np.max(np.abs(u - uAnalytical))

    stress = fieldOutputs.fieldOutputs["stress"].getLastResult()
    reactionForce = fieldOutputs.fieldOutputs["reactionForce"].getLastResult().sum()

    print()
    print("axial stress          : {:.8f} (expected {:.8f})".format(stress[0, 0], traction))
    print("reaction force        : {:.8f} (expected {:.8f})".format(reactionForce, -traction))
    print("tip displacement      : {:.8f} (expected {:.8f})".format(u[-1], uAnalytical[-1]))
    print("max error vs analytic : {:.3e}".format(error))

    return error


if __name__ == "__main__":
    main()
