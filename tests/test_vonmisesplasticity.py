#!/usr/bin/env python3
"""Reconstruction of ``testfiles/edelweiss-only/VonMises``.

A 10 x 10 block of J2 plasticity with isotropic hardening, clamped on its left edge and
dragged along its right edge, in two steps. This is the first case here with a
**path dependent** material, and therefore the first one that exercises the state variables:
every material point owns its own hardening history, that history has to survive Newton
iterations and increment cutbacks, and it must not leak between material points.

That last point matters, because a single material instance is shared by all stencils -- the
material only holds a *pointer* to the state variables it operates on, and the stencil hands
it the right block before every evaluation. If that handover were missing, every material
point would silently compute with the history of whichever one was initialized last.

The material is the *native* EdelweissFE ``VonMises``, so this test needs no Marmot.
"""

import numpy as np
import pytest

from edelweissfd.drivers.pythonscriptedsimulation import FDSimulation
from edelweissfd.stencils.displacementstencil import DisplacementStencil

#: E, nu, yield stress, and the hardening parameters of the FE test.
vonMisesProperties = [210000.0, 0.3, 550.0, 1000.0, 200.0, 1400.0]

youngsModulus = vonMisesProperties[0]
yieldStress = vonMisesProperties[2]

sideLength = 10.0

#: The displacement the right edge is dragged to along y, as in the FE test.
prescribedDisplacement = 0.2


def createSimulation(nCells=(2, 2), verbose: bool = False) -> tuple:
    """Set up the block with its material and stencils.

    Parameters
    ----------
    nCells
        The number of cells per direction.
    verbose
        Be verbose during the simulation.

    Returns
    -------
    tuple
        The simulation, the grid and the stencils.
    """

    sim = FDSimulation(domainSize=2, name="VonMises2D", verbose=verbose)

    grid = sim.createStructuredGrid(lengths=[sideLength, sideLength], nGridPoints=[nCells[0] + 1, nCells[1] + 1])

    material = sim.createMaterial("VonMises", vonMisesProperties, provider="edelweiss")

    stencils = sim.assignStencils(
        DisplacementStencil, grid, material=material, stressState="plane strain", thickness=1.0
    )

    return sim, grid, stencils


def solve(nCells=(2, 2), unload: bool = False, verbose: bool = False) -> tuple:
    """Solve the block, optionally unloading it again in a third step.

    Parameters
    ----------
    nCells
        The number of cells per direction.
    unload
        Append a step which drives the prescribed displacement back to zero.
    verbose
        Be verbose during the simulation.

    Returns
    -------
    tuple
        The model, the field output controller, the grid and the stencils.
    """

    sim, grid, stencils = createSimulation(nCells, verbose)

    # step one only installs the supports, exactly as the FE test does
    firstStep = sim.createStep(stepLength=1.0, maxInc=1.0, minInc=1e-2, maxNumInc=100, maxIter=25)
    firstStep.addDirichlet("left", grid.nodeSets["left"], "displacement", {0: 0.0, 1: 0.0})
    firstStep.addDirichlet("support", grid.nodeSets["leftBottom"], "displacement", {1: 0.0})

    secondStep = sim.createStep(stepLength=1.0, maxInc=5e-3, minInc=1e-8, maxNumInc=10000, maxIter=25)
    secondStep.addDirichlet("right", grid.nodeSets["right"], "displacement", {1: prescribedDisplacement})

    if unload:
        # Dirichlet values are increments over the step, so unloading takes the negative of
        # what was applied before
        thirdStep = sim.createStep(stepLength=1.0, maxInc=5e-3, minInc=1e-8, maxNumInc=10000, maxIter=25)
        thirdStep.addDirichlet("right", grid.nodeSets["right"], "displacement", {1: -prescribedDisplacement})

    sim.addNodeFieldOutput("displacement", "displacement", "U", nodeSet=grid.nodeSets["all"])
    sim.addNodeFieldOutput("reaction", "displacement", "P", nodeSet=grid.nodeSets["left"])
    sim.addStencilFieldOutput("stress", "stress", stencils=stencils)
    sim.addStencilFieldOutput("kappa", "kappa", stencils=stencils)

    model, fieldOutputs = sim.run()

    return model, fieldOutputs, grid, stencils


def test_theBlockYields():
    """The prescribed displacement is far beyond the elastic limit, so the equivalent plastic
    strain has to be positive and the stress has to be bounded by the hardening law."""

    model, fieldOutputs, grid, stencils = solve()

    kappa = fieldOutputs.fieldOutputs["kappa"].getLastResult()

    assert kappa.max() > 1e-4, "no plastic strain accumulated at all"

    stress = fieldOutputs.fieldOutputs["stress"].getLastResult()

    vonMisesStress = _vonMisesStress(stress)

    # isotropic hardening from 550 with the given parameters cannot reach the ultimate 1400
    assert vonMisesStress.max() < 1.5 * vonMisesProperties[5]


def test_materialPointsDoNotShareHistory():
    """Each material point has to accumulate its own plastic strain.

    Under this loading the block is far from homogeneous, so the plastic strain has to differ
    between material points. A single shared history would make them all identical.
    """

    model, fieldOutputs, grid, stencils = solve(nCells=(4, 4))

    kappa = fieldOutputs.fieldOutputs["kappa"].getLastResult().flatten()

    assert kappa.size == len(stencils) * stencils[0].nMaterialPoints

    assert np.ptp(kappa) > 1e-6, "all material points show the same plastic strain"


def test_unloadingLeavesPlasticStrainAndResidualStress():
    """Plasticity is irreversible: after driving the displacement back to zero the plastic
    strain has to remain and residual stresses have to be left behind."""

    model, fieldOutputs, grid, stencils = solve(unload=True)

    displacements = fieldOutputs.fieldOutputs["displacement"].getLastResult()
    coordinates = np.array([node.coordinates for node in model.nodeFields["displacement"].nodes])

    atRight = np.isclose(coordinates[:, 0], sideLength)

    # the boundary is back where it started
    assert np.allclose(displacements[atRight, 1], 0.0, atol=1e-10)

    kappa = fieldOutputs.fieldOutputs["kappa"].getLastResult()
    stress = fieldOutputs.fieldOutputs["stress"].getLastResult()

    assert kappa.max() > 1e-4, "the plastic strain did not survive unloading"
    assert np.max(np.abs(stress)) > 1e-3 * yieldStress, "no residual stress after unloading"


def test_homogeneousDeformationMatchesTheMaterialDrivenDirectly():
    """A sharp check of the whole chain.

    Prescribing a linear displacement field on *every* boundary grid point makes the discrete
    solution exactly that linear field, so every material point sees the same, known strain
    history. The stress the stencils report then has to equal the stress obtained by driving
    the very same material with the same strain increments, computed here independently.
    """

    nIncrements = 4
    axialStrain = 0.01

    sim, grid, stencils = createSimulation(nCells=(2, 2))

    # A state of uniaxial strain: the lateral displacement is suppressed on the top and bottom
    # edges and the axial displacement is prescribed on the left and right ones. The exact
    # solution of that boundary value problem is the homogeneous field u = (eps * x, 0), so
    # every material point follows the very same strain path.
    step = sim.createStep(
        stepLength=1.0,
        maxInc=1.0 / nIncrements,
        minInc=1.0 / nIncrements,
        maxNumInc=nIncrements,
        maxIter=25,
    )

    step.addDirichlet("left", grid.nodeSets["left"], "displacement", {0: 0.0, 1: 0.0})
    step.addDirichlet("right", grid.nodeSets["right"], "displacement", {0: axialStrain * sideLength, 1: 0.0})
    step.addDirichlet("bottom", grid.nodeSets["bottom"], "displacement", {1: 0.0})
    step.addDirichlet("top", grid.nodeSets["top"], "displacement", {1: 0.0})

    sim.addStencilFieldOutput("stress", "stress", stencils=stencils)
    sim.addStencilFieldOutput("kappa", "kappa", stencils=stencils)

    model, fieldOutputs = sim.run()

    stress = np.asarray(fieldOutputs.fieldOutputs["stress"].getLastResult()).reshape(-1, 6)
    kappa = fieldOutputs.fieldOutputs["kappa"].getLastResult().flatten()

    # every material point sees the same strain, so all results have to coincide
    assert np.allclose(stress, stress[0], atol=1e-8 * max(1.0, np.max(np.abs(stress))))
    assert np.allclose(kappa, kappa[0], atol=1e-10)

    expectedStress, expectedKappa = _driveMaterialDirectly(axialStrain, nIncrements)

    assert np.allclose(stress[0], expectedStress, rtol=1e-9, atol=1e-9)
    assert kappa[0] == pytest.approx(expectedKappa, rel=1e-9, abs=1e-12)


def _driveMaterialDirectly(axialStrain: float, nIncrements: int) -> tuple:
    """Integrate the native VonMises material through a uniaxial strain path.

    Parameters
    ----------
    axialStrain
        The total axial strain.
    nIncrements
        The number of equal increments.

    Returns
    -------
    tuple
        The final six component Voigt stress and the final equivalent plastic strain.
    """

    from edelweissfe.config.materiallibrary import getMaterialClass

    materialClass = getMaterialClass("VonMises", "edelweiss")
    material = materialClass(np.array(vonMisesProperties))

    stateVars = np.zeros(material.getNumberOfRequiredStateVars())
    material.assignCurrentStateVars(stateVars)

    stress = np.zeros(6)
    tangent = np.zeros((6, 6))

    for increment in range(nIncrements):
        dStrain = np.zeros(6)
        dStrain[0] = axialStrain / nIncrements

        material.computeStress(stress, tangent, dStrain, float(increment + 1) / nIncrements, 1.0 / nIncrements)

    return stress.copy(), float(np.asarray(material.getResult("kappa")).flatten()[0])


def _vonMisesStress(stress: np.ndarray) -> np.ndarray:
    """The von Mises equivalent stress of Voigt stress vectors.

    Parameters
    ----------
    stress
        The stresses in Marmot's Voigt convention. A field output over several material points
        per stencil is shaped ``(nStencils, nMaterialPoints, 6)``, which is flattened here.

    Returns
    -------
    np.ndarray
        The equivalent stress per material point.
    """

    stress = np.asarray(stress).reshape(-1, 6)

    s11, s22, s33, s12, s13, s23 = (stress[:, i] for i in range(6))

    return np.sqrt(0.5 * ((s11 - s22) ** 2 + (s22 - s33) ** 2 + (s33 - s11) ** 2) + 3.0 * (s12**2 + s13**2 + s23**2))
