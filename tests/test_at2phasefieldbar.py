#!/usr/bin/env python3
"""The AT2 phase-field bar, the multifield finite difference counterpart of the EdelweissFE
regression test ``testfiles/marmot/AT2PhaseField``.

Two kinds of checks are made. The physical ones state what the solution has to look like:
the crack has to localize in the weakened zone, its width has to scale with the length
scale, and the load has to peak and then soften. The regression check compares the final
solution vector against a stored reference the same way the EdelweissFE test runner does,
i.e. the concatenation of all nodal values of all node fields with an absolute tolerance of
1e-6.
"""

import pathlib

import numpy as np
import pytest

from edelweissfd.drivers.pythonscriptedsimulation import FDSimulation
from edelweissfd.stencils.gradientenhanceddisplacementstencil import (
    GradientEnhancedDisplacementStencil,
)

barLength = 50.0
weakenedZoneHalfWidth = 2.5

#: Where the stored reference solutions live.
referenceDirectory = pathlib.Path(__file__).parent / "references"

#: The tolerance of the EdelweissFE regression runner.
regressionTolerance = 1e-6


def buildBar(sim: FDSimulation, nGridPoints: int, properties, lengthScale: float):
    """Create the grid and the stencils of the bar, with a weakened central zone.

    Parameters
    ----------
    sim
        The simulation.
    nGridPoints
        The number of grid points along the bar.
    properties
        The material properties ``[E, nu, Gc, l]``; the length scale entry is overwritten.
    lengthScale
        The phase field length scale.

    Returns
    -------
    The grid.
    """

    grid = sim.createStructuredGrid(lengths=[barLength], nGridPoints=[nGridPoints])

    strong = list(properties)
    strong[3] = lengthScale

    weak = list(strong)
    weak[2] = 0.9 * strong[2]

    strongMaterial = sim.createMaterial("AT2PhaseField", strong, baseClass="gradientEnhancedHypoElastic")
    weakMaterial = sim.createMaterial("AT2PhaseField", weak, baseClass="gradientEnhancedHypoElastic")

    def materialAt(coordinates):
        isWeakened = abs(coordinates[0] - 0.5 * barLength) <= weakenedZoneHalfWidth

        return weakMaterial if isWeakened else strongMaterial

    sim.assignStencils(GradientEnhancedDisplacementStencil, grid, material=materialAt, stressState="3d")

    return grid


def solveBarUnderCrackOpeningControl(
    nGridPoints: int,
    properties,
    lengthScale: float = 5.0,
    finalCrackOpening: float = 0.05,
    controlPointOffset: float = 2.0,
) -> tuple:
    """Solve the bar under indirect control of the opening across the localizing zone.

    Returns
    -------
    tuple
        The model, the field output controller and the grid.
    """

    sim = FDSimulation(domainSize=1, name="at2Bar", verbose=False)

    grid = buildBar(sim, nGridPoints, properties, lengthScale)

    sim.createSolver("NISTPArcLength")

    def closest(x):
        return min(grid.nodes.values(), key=lambda node: abs(node.coordinates[0] - x))

    step = sim.createStep(stepLength=1.0, maxInc=1e-2, minInc=1e-8, maxNumInc=2000, maxIter=15)
    step.addDirichlet("left", grid.nodeSets["left"], "displacement", {0: 0.0})
    step.addDirichlet("right", grid.nodeSets["right"], "displacement", {0: 0.5})
    step.addIndirectControl(
        closest(0.5 * barLength - controlPointOffset),
        [-1.0],
        closest(0.5 * barLength + controlPointOffset),
        [1.0],
        finalCrackOpening,
    )

    sim.addNodeFieldOutput("displacement", "displacement", "U", nodeSet=grid.nodeSets["all"])
    sim.addNodeFieldOutput("phaseField", "nonlocal damage", "U", nodeSet=grid.nodeSets["all"])
    sim.addNodeFieldOutput(
        "reactionForce",
        "displacement",
        "P",
        nodeSet=grid.nodeSets["right"],
        saveHistory=True,
        f_x=lambda x: np.sum(x[:, 0]),
    )

    model, fieldOutputs = sim.run()

    return model, fieldOutputs, grid


def solutionVector(model) -> np.ndarray:
    """The concatenation of all nodal values of all node fields, plus the scalar variables.

    This is exactly the quantity the EdelweissFE test runner compares against ``U.ref``.
    """

    nodalValues = [nodeField["U"].flatten() for nodeField in model.nodeFields.values()]
    scalarValues = [variable.value for variable in model.scalarVariables.values()]

    return np.hstack(nodalValues + scalarValues).flatten()


@pytest.mark.marmot
def test_crackLocalizesInTheWeakenedZone(at2PhaseFieldProperties):
    """The phase field has to concentrate in the weakened central zone.

    Note that the AT2 model has no elastic threshold, so the phase field is non-zero
    everywhere in a bar under a nearly uniform strain; what identifies the crack is the
    pronounced maximum in the weakened zone and the monotonic decay away from it, not a
    vanishing phase field at the ends.
    """

    model, fieldOutputs, grid = solveBarUnderCrackOpeningControl(51, at2PhaseFieldProperties)

    phaseField = fieldOutputs.fieldOutputs["phaseField"].getLastResult().flatten()
    coordinates = np.array([node.coordinates[0] for node in model.nodeFields["nonlocal damage"].nodes])

    order = np.argsort(coordinates)
    coordinates = coordinates[order]
    phaseField = phaseField[order]

    assert phaseField.max() > 0.9, "the bar should be fully cracked at the end of the step"

    centreIndex = int(np.argmax(phaseField))

    assert abs(coordinates[centreIndex] - 0.5 * barLength) <= weakenedZoneHalfWidth

    # a pronounced maximum: the ends carry only the background damage
    assert phaseField[0] < 0.4 * phaseField.max()
    assert phaseField[-1] < 0.4 * phaseField.max()

    # monotonic decay away from the crack, in both directions
    assert np.all(np.diff(phaseField[: centreIndex + 1]) > -1e-9)
    assert np.all(np.diff(phaseField[centreIndex:]) < 1e-9)

    # and symmetric, since the bar and its weakened zone are
    assert np.allclose(phaseField, phaseField[::-1], atol=1e-8)


@pytest.mark.marmot
def test_loadPeaksAndThenSoftens(at2PhaseFieldProperties):
    """The response has to rise to a peak and then decay towards zero."""

    model, fieldOutputs, grid = solveBarUnderCrackOpeningControl(51, at2PhaseFieldProperties)

    force = np.abs(np.array(fieldOutputs.fieldOutputs["reactionForce"].getResultHistory()).flatten())

    peak = int(np.argmax(force))

    assert 0 < peak < len(force) - 1, "the peak must be reached inside the step"
    assert force[-1] < 0.05 * force[peak], "the bar must be essentially unloaded at the end"

    # monotonic rise up to the peak
    assert np.all(np.diff(force[: peak + 1]) > -1e-9)


@pytest.mark.marmot
@pytest.mark.parametrize("lengthScale", [4.0, 8.0])
def test_localizationWidthScalesWithLengthScale(lengthScale, at2PhaseFieldProperties):
    """The width of the localized zone is set by the length scale of the material, which is
    the whole point of the gradient enhancement: the result must not be set by the grid."""

    model, fieldOutputs, grid = solveBarUnderCrackOpeningControl(51, at2PhaseFieldProperties, lengthScale=lengthScale)

    phaseField = fieldOutputs.fieldOutputs["phaseField"].getLastResult().flatten()
    coordinates = np.array([node.coordinates[0] for node in model.nodeFields["nonlocal damage"].nodes])

    localized = coordinates[phaseField > 0.5 * phaseField.max()]
    width = localized.max() - localized.min()

    # the AT2 profile has a support of a few length scales; well away from both the grid
    # spacing and the length of the bar
    assert 0.5 * lengthScale < width < 5.0 * lengthScale


@pytest.mark.marmot
def test_solutionMatchesStoredReference(at2PhaseFieldProperties):
    """Regression check against a stored reference solution, in the format and with the
    tolerance of the EdelweissFE test runner.

    Regenerate the reference with::

        python -m tests.regenerate_references
    """

    model, fieldOutputs, grid = solveBarUnderCrackOpeningControl(51, at2PhaseFieldProperties)

    computed = solutionVector(model)

    referenceFile = referenceDirectory / "AT2PhaseFieldBar1D.ref"

    if not referenceFile.exists():
        pytest.skip("no stored reference; regenerate it with tests/regenerate_references.py")

    reference = np.loadtxt(referenceFile)

    assert computed.shape == reference.shape

    assert np.max(np.abs(computed - reference)) < regressionTolerance
