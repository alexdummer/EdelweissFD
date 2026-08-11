#!/usr/bin/env python3
"""The linear elastic tension bar, the finite difference counterpart of the EdelweissFE
regression tests ``testfiles/edelweiss-only/TensionBarQuad4`` and
``testfiles/marmot/LinearElasticIsotropic``.

A linear displacement field is in the span of the difference operators, so the discrete
solution has to reproduce the closed form solution to round off, on any grid. That makes
this a much sharper check than a convergence rate.
"""

import numpy as np
import pytest

from edelweissfd.drivers.pythonscriptedsimulation import FDSimulation
from edelweissfd.stencils.displacementstencil import DisplacementStencil

youngsModulus = 20000.0
poissonsRatio = 0.2

length = 100.0
traction = 10.0


def solveBar1D(nGridPoints: int, properties) -> tuple:
    """Solve the one dimensional bar under uniaxial stress.

    Parameters
    ----------
    nGridPoints
        The number of grid points along the bar.
    properties
        The material properties.

    Returns
    -------
    tuple
        The axial coordinates and the axial displacements.
    """

    sim = FDSimulation(domainSize=1, name="bar1D", verbose=False)

    grid = sim.createStructuredGrid(lengths=[length], nGridPoints=[nGridPoints])
    material = sim.createMaterial("LinearElastic", properties)

    sim.assignStencils(DisplacementStencil, grid, material=material, stressState="uniaxial stress")

    step = sim.createStep(stepLength=1.0)
    step.addDirichlet("clamped", grid.nodeSets["left"], "displacement", {0: 0.0})
    step.addNeumann("pulled", grid, "right", "displacement", [traction])

    sim.addNodeFieldOutput("U", "displacement", "U", nodeSet=grid.nodeSets["all"])
    sim.addStencilFieldOutput("stress", "stress")

    model, fieldOutputs = sim.run()

    coordinates = np.array([node.coordinates[0] for node in model.nodeFields["displacement"].nodes])
    displacements = fieldOutputs.fieldOutputs["U"].getLastResult().flatten()

    return coordinates, displacements, fieldOutputs.fieldOutputs["stress"].getLastResult()


@pytest.mark.marmot
@pytest.mark.parametrize("nGridPoints", [2, 5, 21, 101])
def test_bar1DReproducesClosedFormSolution(nGridPoints, linearElasticProperties):
    """u(x) = t x / E, exactly, independent of the grid."""

    coordinates, displacements, stress = solveBar1D(nGridPoints, linearElasticProperties)

    expected = traction * coordinates / youngsModulus

    assert np.allclose(displacements, expected, atol=1e-12)
    assert np.allclose(stress[:, 0], traction, atol=1e-10)


@pytest.mark.marmot
def test_bar2DPlaneStrainReproducesClosedFormSolution(linearElasticProperties):
    """A plane strain bar free to contract laterally has the reduced modulus E/(1-nu^2)
    and the reduced Poisson's ratio nu/(1-nu)."""

    height = 20.0

    sim = FDSimulation(domainSize=2, name="bar2D", verbose=False)

    grid = sim.createStructuredGrid(lengths=[length, height], nGridPoints=[11, 5])
    material = sim.createMaterial("LinearElastic", linearElasticProperties)

    sim.assignStencils(DisplacementStencil, grid, material=material, stressState="plane strain")

    step = sim.createStep(stepLength=1.0)
    step.addDirichlet("clamped", grid.nodeSets["left"], "displacement", {0: 0.0})
    step.addDirichlet("supported", grid.nodeSets["leftBottom"], "displacement", {1: 0.0})
    step.addNeumann("pulled", grid, "right", "displacement", [traction, 0.0])

    sim.addNodeFieldOutput("U", "displacement", "U", nodeSet=grid.nodeSets["all"])

    model, fieldOutputs = sim.run()

    coordinates = np.array([node.coordinates for node in model.nodeFields["displacement"].nodes])
    displacements = fieldOutputs.fieldOutputs["U"].getLastResult()

    reducedModulus = youngsModulus / (1.0 - poissonsRatio**2)
    reducedPoissonsRatio = poissonsRatio / (1.0 - poissonsRatio)

    assert np.allclose(displacements[:, 0], traction * coordinates[:, 0] / reducedModulus, atol=1e-12)
    assert np.allclose(
        displacements[:, 1], -reducedPoissonsRatio * traction / reducedModulus * coordinates[:, 1], atol=1e-12
    )


@pytest.mark.marmot
def test_bar2DHasNoHourglassMode(linearElasticProperties):
    """A single cell must not admit a zero energy checkerboard displacement pattern; if it
    did, the solution of the bar would oscillate from grid point to grid point."""

    sim = FDSimulation(domainSize=2, name="hourglass", verbose=False)

    grid = sim.createStructuredGrid(lengths=[10.0, 10.0], nGridPoints=[3, 3])
    material = sim.createMaterial("LinearElastic", linearElasticProperties)

    stencils = sim.assignStencils(DisplacementStencil, grid, material=material, stressState="plane strain")

    stencil = stencils[0]

    hourglass = np.zeros(stencil.nDof)
    for corner, offset in enumerate(stencil.cornerOffsets):
        hourglass[corner * 2 + 0] = (-1.0) ** offset.sum()

    K = np.zeros((stencil.nDof, stencil.nDof))
    P = np.zeros(stencil.nDof)

    sim.model.prepareYourself(sim.journal)
    stencil.computeKernels(K, P, hourglass, hourglass, 1.0, 1.0)

    energy = hourglass @ K @ hourglass

    assert energy > 1.0
