#!/usr/bin/env python3
"""Reconstruction of ``testfiles/edelweiss-only/TensionBarHexa8`` and ``WallShearHexa8``.

Both use the ``boxGen`` generator of EdelweissFE, i.e. a structured hexahedral grid, and are
the first cases here solved in **three dimensions**.

``TensionBarHexa8`` is a slender 5 x 5 x 500 bar, clamped on one end face and pulled on the
other by nodal forces. It has a closed form answer,

    sigma = F / A,     delta = sigma L / E

``WallShearHexa8`` is a 1000 x 1000 x 5 wall, clamped along its base, sheared along its top
edge and additionally loaded by its own weight. It is the only case here with a **body force**,
which makes it the test of that load type.

Both materials are the *native* EdelweissFE ``linearelastic``, so these tests need no Marmot.
"""

import numpy as np
import pytest

from edelweissfd.drivers.pythonscriptedsimulation import FDSimulation
from edelweissfd.stencils.displacementstencil import DisplacementStencil

# -- the tension bar -------------------------------------------------------------------

barYoungsModulus = 1.8e4
barPoissonsRatio = 0.22

barCrossSection = 5.0
barLength = 500.0

#: The axial force applied to every grid point of the loaded end face, as in the FE test.
barNodalForce = -4.0


def solveTensionBar(nCells=(2, 2, 20), verbose: bool = False) -> tuple:
    """Solve the 3D tension bar.

    Parameters
    ----------
    nCells
        The number of cells across the two lateral directions and along the axis.
    verbose
        Be verbose during the simulation.

    Returns
    -------
    tuple
        The model, the field output controller and the grid.
    """

    sim = FDSimulation(domainSize=3, name="TensionBar3D", verbose=verbose)

    grid = sim.createStructuredGrid(
        lengths=[barCrossSection, barCrossSection, barLength],
        nGridPoints=[n + 1 for n in nCells],
    )

    material = sim.createMaterial("linearelastic", [barYoungsModulus, barPoissonsRatio], provider="edelweiss")

    sim.assignStencils(DisplacementStencil, grid, material=material, stressState="3d")

    step = sim.createStep(stepLength=1.0, maxInc=1.0, minInc=1e-9, maxNumInc=1000, maxIter=25)

    # boxGen puts 'front' at the upper end of the third direction and 'back' at its lower one
    step.addDirichlet("clamped", grid.nodeSets["front"], "displacement", {0: 0.0, 1: 0.0, 2: 0.0})
    step.addNodeForces("pulled", grid.nodeSets["back"], "displacement", {0: 0.0, 1: 0.0, 2: barNodalForce})

    sim.addNodeFieldOutput("displacement", "displacement", "U", nodeSet=grid.nodeSets["all"])
    sim.addNodeFieldOutput("reaction", "displacement", "P", nodeSet=grid.nodeSets["front"])
    sim.addStencilFieldOutput("stress", "stress")

    model, fieldOutputs = sim.run()

    return model, fieldOutputs, grid


def test_tensionBarMatchesTheClosedFormSolution():
    """Axial stress and elongation have to match the closed form values.

    The bar is clamped at its far end and the load points away from it, so the bar is in
    *tension* and the loaded end moves in the negative axial direction.
    """

    model, fieldOutputs, grid = solveTensionBar()

    totalForce = len(grid.nodeSets["back"]) * barNodalForce

    area = barCrossSection**2
    tensileStress = abs(totalForce) / area
    expectedDisplacement = -tensileStress * barLength / barYoungsModulus

    displacements = fieldOutputs.fieldOutputs["displacement"].getLastResult()
    coordinates = np.array([node.coordinates for node in model.nodeFields["displacement"].nodes])

    atLoadedEnd = np.isclose(coordinates[:, 2], 0.0)

    axialDisplacement = displacements[atLoadedEnd, 2].mean()

    # the clamp suppresses the lateral contraction locally, which is a Saint-Venant boundary
    # layer of the order of the cross section and thus negligible for this slender bar
    assert axialDisplacement == pytest.approx(expectedDisplacement, rel=0.02)

    stress = np.asarray(fieldOutputs.fieldOutputs["stress"].getLastResult()).reshape(-1, 6)

    # away from the clamp the state has to be uniaxial tension
    assert np.median(stress[:, 2]) == pytest.approx(tensileStress, rel=0.02)

    # and free of lateral and shear stress
    for component in (0, 1, 3, 4, 5):
        assert abs(np.median(stress[:, component])) < 0.02 * tensileStress


def test_tensionBarReactionsBalanceTheLoad():
    """The clamped face has to carry the whole applied force."""

    model, fieldOutputs, grid = solveTensionBar(nCells=(2, 2, 10))

    totalForce = len(grid.nodeSets["back"]) * barNodalForce

    reaction = fieldOutputs.fieldOutputs["reaction"].getLastResult()

    assert reaction[:, 2].sum() == pytest.approx(-totalForce, rel=1e-9)
    assert reaction[:, 0].sum() == pytest.approx(0.0, abs=1e-9 * abs(totalForce))
    assert reaction[:, 1].sum() == pytest.approx(0.0, abs=1e-9 * abs(totalForce))


def test_tensionBarContractsLaterally():
    """A bar in tension has to contract laterally with Poisson's ratio.

    The lateral strain is measured as a *gradient* across the section, not as a displacement
    divided by the width: the clamp fixes the lateral displacement over its whole face, so the
    section contracts about its own centre line and the displacement at one face is only half
    of the total contraction.
    """

    model, fieldOutputs, grid = solveTensionBar()

    displacements = fieldOutputs.fieldOutputs["displacement"].getLastResult()
    coordinates = np.array([node.coordinates for node in model.nodeFields["displacement"].nodes])

    # at the free end, away from the clamp
    atLoadedEnd = np.isclose(coordinates[:, 2], 0.0)
    atLowSide = np.isclose(coordinates[:, 0], 0.0) & atLoadedEnd
    atHighSide = np.isclose(coordinates[:, 0], barCrossSection) & atLoadedEnd

    lateralStrain = (displacements[atHighSide, 0].mean() - displacements[atLowSide, 0].mean()) / barCrossSection

    # the bar is in tension, so the axial strain is positive while the displacement is negative
    axialStrain = -displacements[atLoadedEnd, 2].mean() / barLength

    assert lateralStrain / axialStrain == pytest.approx(-barPoissonsRatio, rel=0.05)


# -- the sheared wall with self weight -------------------------------------------------

wallYoungsModulus = 2.1e4
wallPoissonsRatio = 0.22

wallWidth = 1000.0
wallHeight = 1000.0
wallThickness = 5.0

#: The shear force applied to every grid point of the top edge, as in the FE test.
wallNodalForce = 1.0

#: The self weight as a force per unit volume, as in the FE test.
wallSpecificWeight = -0.000077


def solveShearedWall(nCells=(10, 10, 2), withBodyForce: bool = True, verbose: bool = False) -> tuple:
    """Solve the 3D sheared wall.

    Parameters
    ----------
    nCells
        The number of cells along the width, the height and the thickness.
    withBodyForce
        Whether the self weight is applied.
    verbose
        Be verbose during the simulation.

    Returns
    -------
    tuple
        The model, the field output controller and the grid.
    """

    sim = FDSimulation(domainSize=3, name="ShearedWall3D", verbose=verbose)

    grid = sim.createStructuredGrid(lengths=[wallWidth, wallHeight, wallThickness], nGridPoints=[n + 1 for n in nCells])

    material = sim.createMaterial("linearelastic", [wallYoungsModulus, wallPoissonsRatio], provider="edelweiss")

    stencils = sim.assignStencils(DisplacementStencil, grid, material=material, stressState="3d")

    step = sim.createStep(stepLength=1.0, maxInc=1.0, minInc=1e-9, maxNumInc=1000, maxIter=25)

    step.addDirichlet("base", grid.nodeSets["bottom"], "displacement", {0: 0.0, 1: 0.0, 2: 0.0})
    step.addNodeForces("shear", grid.nodeSets["top"], "displacement", {0: wallNodalForce, 1: 0.0, 2: 0.0})

    if withBodyForce:
        step.addBodyForce("weight", stencils, [0.0, wallSpecificWeight, 0.0])

    sim.addNodeFieldOutput("displacement", "displacement", "U", nodeSet=grid.nodeSets["all"])
    sim.addNodeFieldOutput("reaction", "displacement", "P", nodeSet=grid.nodeSets["bottom"])
    sim.addStencilFieldOutput("stress", "stress", stencils=stencils)

    model, fieldOutputs = sim.run()

    return model, fieldOutputs, grid


def totalWeightOfTheWall() -> float:
    """The total self weight of the wall."""

    return wallSpecificWeight * wallWidth * wallHeight * wallThickness


def bodyForceAppliedAtTheBase(grid) -> float:
    """The share of the self weight which is applied directly at the clamped base grid points.

    A body force is distributed onto the grid points in proportion to their tributary volume,
    and the tributary volume of a boundary plane is half a cell layer. The share matters for the
    equilibrium check, because ``P`` reported at a constrained grid point is the *internal* flux
    there, i.e. the reaction plus the external load applied at that very grid point.

    Parameters
    ----------
    grid
        The grid of the wall.

    Returns
    -------
    float
        The vertical load applied at the base grid points.
    """

    tributaryVolume = 0.5 * grid.spacings[1] * wallWidth * wallThickness

    return wallSpecificWeight * tributaryVolume


def test_shearedWallIsInEquilibriumIncludingSelfWeight():
    """The base has to carry the shear force *and* the total self weight, which is the check that
    the body force is assembled with the right magnitude."""

    model, fieldOutputs, grid = solveShearedWall()

    totalShear = len(grid.nodeSets["top"]) * wallNodalForce
    totalWeight = totalWeightOfTheWall()

    reaction = fieldOutputs.fieldOutputs["reaction"].getLastResult()

    assert reaction[:, 0].sum() == pytest.approx(-totalShear, rel=1e-8)

    verticalReaction = reaction[:, 1].sum() - bodyForceAppliedAtTheBase(grid)

    assert verticalReaction == pytest.approx(-totalWeight, rel=1e-8)


def test_bodyForceChangesTheSolution():
    """Switching the self weight off has to remove exactly the weight from the response."""

    _, withWeight, grid = solveShearedWall(withBodyForce=True)
    _, withoutWeight, _ = solveShearedWall(withBodyForce=False)

    totalWeight = totalWeightOfTheWall()

    verticalWith = withWeight.fieldOutputs["reaction"].getLastResult()[:, 1].sum()
    verticalWithout = withoutWeight.fieldOutputs["reaction"].getLastResult()[:, 1].sum()

    assert verticalWithout == pytest.approx(0.0, abs=1e-8 * abs(totalWeight))

    difference = verticalWith - verticalWithout - bodyForceAppliedAtTheBase(grid)

    assert difference == pytest.approx(-totalWeight, rel=1e-8)


def test_shearedWallDevelopsShearStress():
    """A wall sheared along its top edge has to carry shear stress in the plane of the wall."""

    model, fieldOutputs, grid = solveShearedWall(withBodyForce=False)

    stress = np.asarray(fieldOutputs.fieldOutputs["stress"].getLastResult()).reshape(-1, 6)

    # Voigt component 3 is sigma_12 in Marmot's convention
    inPlaneShear = np.abs(stress[:, 3])

    assert inPlaneShear.max() > 0.0
    assert inPlaneShear.max() > 0.1 * np.abs(stress).max()


def test_shearedWallDeflectsInTheDirectionOfTheShear():
    """The top of the wall has to move in the direction of the applied shear, and the base has
    to stay put."""

    model, fieldOutputs, grid = solveShearedWall(withBodyForce=False)

    displacements = fieldOutputs.fieldOutputs["displacement"].getLastResult()
    coordinates = np.array([node.coordinates for node in model.nodeFields["displacement"].nodes])

    atBase = np.isclose(coordinates[:, 1], 0.0)
    atTop = np.isclose(coordinates[:, 1], wallHeight)

    assert np.allclose(displacements[atBase], 0.0, atol=1e-12)
    assert displacements[atTop, 0].mean() > 0.0
