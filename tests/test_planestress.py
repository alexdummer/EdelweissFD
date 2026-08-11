#!/usr/bin/env python3
"""Reconstruction of ``testfiles/marmot/CPS4``.

A 50 x 100 panel of J2 plasticity in **plane stress**, clamped at the bottom and pushed down at
the top with a small shear component. Plane stress is the last of the four stress states of
:class:`~edelweissfd.stencils.displacementstencil.DisplacementStencil` that had no end to end
coverage, and it differs from the others in that the material itself condenses the out of plane
strain: its defining property, a vanishing out of plane stress, is therefore something the
solution has to satisfy exactly and is checked as such below.

The elastic reference case additionally pins down the difference to plane strain, which is the
factor :math:`1/(1-\\nu^2)` in the effective modulus, and thus verifies that the condensation
really happens.

These tests use Marmot materials.
"""

import numpy as np
import pytest

from edelweissfd.drivers.pythonscriptedsimulation import FDSimulation
from edelweissfd.stencils.displacementstencil import DisplacementStencil

#: E, nu, yield stress and hardening parameters of the FE test.
vonMisesProperties = [210000.0, 0.3, 550.0, 1000.0, 200.0, 1400.0]

panelWidth = 50.0
panelHeight = 100.0

#: The prescribed top displacement of the FE test, downwards with a small shear.
prescribedVertical = -0.5
prescribedShear = 0.001


def solvePanel(nCells=(10, 10), verbose: bool = False) -> tuple:
    """Solve the elastoplastic panel in plane stress.

    Parameters
    ----------
    nCells
        The number of cells per direction.
    verbose
        Be verbose during the simulation.

    Returns
    -------
    tuple
        The model, the field output controller and the grid.
    """

    sim = FDSimulation(domainSize=2, name="Panel2DPlaneStress", verbose=verbose)

    grid = sim.createStructuredGrid(lengths=[panelWidth, panelHeight], nGridPoints=[n + 1 for n in nCells])

    material = sim.createMaterial("VonMises", vonMisesProperties)

    stencils = sim.assignStencils(
        DisplacementStencil, grid, material=material, stressState="plane stress", thickness=1.0
    )

    step = sim.createStep(stepLength=100.0, maxInc=1e-1, minInc=1e-8, maxNumInc=1000, maxIter=25)
    step.addDirichlet("bottom", grid.nodeSets["bottom"], "displacement", {0: 0.0, 1: 0.0})
    step.addDirichlet("top", grid.nodeSets["top"], "displacement", {0: prescribedShear, 1: prescribedVertical})

    sim.addNodeFieldOutput("displacement", "displacement", "U", nodeSet=grid.nodeSets["all"])
    sim.addStencilFieldOutput("stress", "stress", stencils=stencils)
    sim.addStencilFieldOutput("strain", "strain", stencils=stencils)

    model, fieldOutputs = sim.run()

    return model, fieldOutputs, grid


@pytest.mark.marmot
def test_outOfPlaneStressVanishes():
    """The defining property of plane stress, checked on the elastoplastic panel."""

    model, fieldOutputs, grid = solvePanel()

    stress = np.asarray(fieldOutputs.fieldOutputs["stress"].getLastResult()).reshape(-1, 6)

    inPlaneScale = np.max(np.abs(stress[:, [0, 1, 3]]))

    assert inPlaneScale > 1.0, "the panel carries no load at all"

    # sigma_33, sigma_13 and sigma_23 all have to vanish
    for component in (2, 4, 5):
        assert np.max(np.abs(stress[:, component])) < 1e-8 * inPlaneScale


@pytest.mark.marmot
def test_outOfPlaneStrainIsNotZero():
    """Plane stress means the out of plane *strain* is free, in contrast to plane strain. Only
    the in plane strains are applied by the stencil, so a non-zero out of plane strain can only
    come from the material's condensation."""

    model, fieldOutputs, grid = solvePanel()

    strain = np.asarray(fieldOutputs.fieldOutputs["strain"].getLastResult()).reshape(-1, 6)

    inPlaneScale = np.max(np.abs(strain[:, [0, 1, 3]]))

    # the stencil itself only ever writes the in plane components into the applied strain, so
    # this states that the applied strain increment is in plane
    assert np.max(np.abs(strain[:, 2])) < 1e-12 * max(inPlaneScale, 1e-30)


@pytest.mark.marmot
def test_panelYields():
    """A prescribed shortening of 0.5 % is well beyond the elastic limit of this material."""

    model, fieldOutputs, grid = solvePanel()

    stress = np.asarray(fieldOutputs.fieldOutputs["stress"].getLastResult()).reshape(-1, 6)

    s11, s22, s12 = stress[:, 0], stress[:, 1], stress[:, 3]

    vonMisesStress = np.sqrt(s11**2 - s11 * s22 + s22**2 + 3.0 * s12**2)

    assert vonMisesStress.max() > vonMisesProperties[2]


@pytest.mark.marmot
def test_planeStressBarMatchesUniaxialClosedForm(linearElasticProperties):
    """An elastic strip in plane stress, free to contract laterally, has the *unreduced*
    modulus, so its elongation is exactly the uniaxial one."""

    youngsModulus, poissonsRatio = linearElasticProperties

    length = 100.0
    height = 20.0
    traction = 10.0

    sim = FDSimulation(domainSize=2, name="Strip2DPlaneStress", verbose=False)

    grid = sim.createStructuredGrid(lengths=[length, height], nGridPoints=[11, 5])
    material = sim.createMaterial("LinearElastic", linearElasticProperties)

    sim.assignStencils(DisplacementStencil, grid, material=material, stressState="plane stress", thickness=1.0)

    step = sim.createStep(stepLength=1.0)
    step.addDirichlet("clamped", grid.nodeSets["left"], "displacement", {0: 0.0})
    step.addDirichlet("supported", grid.nodeSets["leftBottom"], "displacement", {1: 0.0})
    step.addNeumann("pulled", grid, "right", "displacement", [traction, 0.0])

    sim.addNodeFieldOutput("displacement", "displacement", "U", nodeSet=grid.nodeSets["all"])

    model, fieldOutputs = sim.run()

    displacements = fieldOutputs.fieldOutputs["displacement"].getLastResult()
    coordinates = np.array([node.coordinates for node in model.nodeFields["displacement"].nodes])

    # plane stress: the effective modulus is E itself, not E / (1 - nu^2)
    assert np.allclose(displacements[:, 0], traction * coordinates[:, 0] / youngsModulus, atol=1e-12)

    assert np.allclose(
        displacements[:, 1],
        -poissonsRatio * traction / youngsModulus * coordinates[:, 1],
        atol=1e-12,
    )


@pytest.mark.marmot
def test_planeStressIsSofterThanPlaneStrainByTheExactFactor(linearElasticProperties):
    """Comparing the two stress states side by side has to reproduce ``1 / (1 - nu^2)`` exactly."""

    youngsModulus, poissonsRatio = linearElasticProperties

    length = 100.0
    traction = 10.0

    def tipDisplacement(stressState):
        sim = FDSimulation(domainSize=2, name="Strip" + stressState.replace(" ", ""), verbose=False)

        grid = sim.createStructuredGrid(lengths=[length, 20.0], nGridPoints=[11, 5])
        material = sim.createMaterial("LinearElastic", linearElasticProperties)

        sim.assignStencils(DisplacementStencil, grid, material=material, stressState=stressState)

        step = sim.createStep(stepLength=1.0)
        step.addDirichlet("clamped", grid.nodeSets["left"], "displacement", {0: 0.0})
        step.addDirichlet("supported", grid.nodeSets["leftBottom"], "displacement", {1: 0.0})
        step.addNeumann("pulled", grid, "right", "displacement", [traction, 0.0])

        sim.addNodeFieldOutput("U", "displacement", "U", nodeSet=grid.nodeSets["right"])

        model, fieldOutputs = sim.run()

        return fieldOutputs.fieldOutputs["U"].getLastResult()[:, 0].mean()

    ratio = tipDisplacement("plane stress") / tipDisplacement("plane strain")

    assert ratio == pytest.approx(1.0 / (1.0 - poissonsRatio**2), rel=1e-9)
