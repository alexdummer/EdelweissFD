#!/usr/bin/env python3
"""
AT2 phase-field fracture of a bar with a weakened zone, solved by finite differences.

This is the finite difference counterpart of the EdelweissFE regression test
``testfiles/marmot/AT2PhaseField``. There, a 50 x 10 bar is discretized by an 11 x 1 grid of
``GCPS8R`` elements, i.e. essentially one dimensionally, with the central section weakened
by a reduced fracture energy, and pulled apart.

It is a genuine **multifield** problem: every grid point carries a displacement *and* the
phase field, and the two are coupled in both directions. The momentum balance is solved
together with the screened Poisson equation of the phase field,

    -div sigma(eps, phi) = 0
    phi - div( l^2 grad phi ) - 2 l / Gc (1 - phi) H(eps) = 0

with the degraded stress sigma = (1 - phi)^2 C eps. Both equations, the constitutive
response and all coupling tangents come from Marmot's ``AT2PHASEFIELD`` material, reached
point-wise through :mod:`edelweissfe.materials.marmot.marmotgradientenhancedhypoelastic`.

Four variants are run:

* a one dimensional bar under displacement control, which stops just past the peak load,
* the same bar under indirect (crack opening) control, which follows the whole softening
  branch including the snap-back that displacement control cannot pass,
* a two dimensional bar in plane strain, which exercises the coupled stencil in 2D,
* a two dimensional bar in plane stress, which is directly comparable to the FE test.

**Cross-validation.** The plane stress variant reproduces the peak load of the EdelweissFE
test, which uses 8-node quadratic reduced-integration elements on an 11 x 1 mesh, to within a
few tenths of a percent:

===================================  ===============  ==================
variant                              peak load        displacement
===================================  ===============  ==================
EdelweissFE, GCPS8R, 11 x 1 mesh     64.57            0.02835
EdelweissFD, plane stress, 51 x 11   64.42            0.02800
===================================  ===============  ==================

To regenerate the reference figures of the FE test::

    cd <EdelweissFE>
    run_tests_edelweissfe ./testfiles/marmot/ --tests AT2PhaseField

which writes ``testfiles/marmot/AT2PhaseField/RF.csv`` and ``U.csv``.

Run this script with::

    python run.py
"""

import numpy as np

from edelweissfd.drivers.pythonscriptedsimulation import FDSimulation
from edelweissfd.stencils.gradientenhanceddisplacementstencil import (
    GradientEnhancedDisplacementStencil,
)

youngsModulus = 20000.0
poissonsRatio = 0.2

fractureEnergy = 0.1
fractureEnergyWeakened = 0.09

lengthScale = 5.0

barLength = 50.0
barHeight = 10.0

#: The half width of the weakened zone at the centre of the bar.
weakenedZoneHalfWidth = 2.5

#: The end displacement the bar is pulled to under displacement control. Chosen just past
#: the peak load; beyond it the equilibrium path snaps back and displacement control has no
#: solution any more, which is what the arc-length variant below is for.
endDisplacement = 0.03

#: The opening across the localizing zone the arc-length variant drives to.
finalCrackOpening = 0.05

#: The end displacement the arc-length parameter scales. Only an upper bound, see run1DArcLength.
referenceEndDisplacement = 0.5

#: The distance of the two indirect control points from the centre of the bar, mirroring
#: the control points at x = 23 and x = 27 of the EdelweissFE test.
controlPointOffset = 2.0


def createMaterialFactory(sim: FDSimulation):
    """A material factory weakening the central zone of the bar.

    Parameters
    ----------
    sim
        The simulation, used to create the materials.

    Returns
    -------
    callable
        A callable taking the coordinates of a material point and returning a material.
    """

    strongMaterial = sim.createMaterial(
        "AT2PhaseField",
        [youngsModulus, poissonsRatio, fractureEnergy, lengthScale],
        baseClass="gradientEnhancedHypoElastic",
    )
    weakMaterial = sim.createMaterial(
        "AT2PhaseField",
        [youngsModulus, poissonsRatio, fractureEnergyWeakened, lengthScale],
        baseClass="gradientEnhancedHypoElastic",
    )

    def materialAt(coordinates):
        isWeakened = abs(coordinates[0] - 0.5 * barLength) <= weakenedZoneHalfWidth

        return weakMaterial if isWeakened else strongMaterial

    return materialAt


def run1D(nGridPoints: int = 101, verbose: bool = True):
    """Solve the one dimensional bar.

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

    sim = FDSimulation(domainSize=1, name="AT2PhaseFieldBar1D", verbose=verbose)

    grid = sim.createStructuredGrid(lengths=[barLength], nGridPoints=[nGridPoints])

    sim.assignStencils(
        GradientEnhancedDisplacementStencil,
        grid,
        material=createMaterialFactory(sim),
        stressState="3d",
    )

    step = sim.createStep(stepLength=1.0, maxInc=2e-2, minInc=1e-8, maxNumInc=1000, maxIter=15)
    step.addDirichlet("left", grid.nodeSets["left"], "displacement", {0: 0.0})
    step.addDirichlet("right", grid.nodeSets["right"], "displacement", {0: endDisplacement})

    _addOutputs(sim, grid)

    return sim.run()


def run1DArcLength(nGridPoints: int = 101, verbose: bool = True):
    """Solve the one dimensional bar under indirect (crack opening) control.

    Displacement control cannot follow the softening branch of this bar, because the
    equilibrium path snaps back: past the peak the end displacement has to *decrease* while
    the crack keeps opening. Controlling the opening across the localizing zone instead, with
    the arc-length solver of EdelweissFE, follows the whole branch. This mirrors what
    ``testfiles/marmot/AT2PhaseField`` does with ``NISTPArcLength`` and ``indirectcontrol``.

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

    sim = FDSimulation(domainSize=1, name="AT2PhaseFieldBar1DArcLength", verbose=verbose)

    grid = sim.createStructuredGrid(lengths=[barLength], nGridPoints=[nGridPoints])

    sim.assignStencils(
        GradientEnhancedDisplacementStencil,
        grid,
        material=createMaterialFactory(sim),
        stressState="3d",
    )

    sim.createSolver("NISTPArcLength")

    leftControlPoint = _closestGridPoint(grid, 0.5 * barLength - controlPointOffset)
    rightControlPoint = _closestGridPoint(grid, 0.5 * barLength + controlPointOffset)

    step = sim.createStep(stepLength=1.0, maxInc=1e-2, minInc=1e-8, maxNumInc=2000, maxIter=15)
    step.addDirichlet("left", grid.nodeSets["left"], "displacement", {0: 0.0})

    # The arc-length parameter scales this prescribed end displacement, while the indirect
    # control decides how much of it is applied in every increment. The value is therefore
    # only an upper bound on the end displacement, not the value it actually reaches, which
    # is exactly how testfiles/marmot/AT2PhaseField is set up.
    step.addDirichlet("right", grid.nodeSets["right"], "displacement", {0: referenceEndDisplacement})

    step.addIndirectControl(
        leftControlPoint,
        [-1.0],
        rightControlPoint,
        [1.0],
        finalCrackOpening,
    )

    _addOutputs(sim, grid)

    return sim.run()


def _closestGridPoint(grid, x: float):
    """The grid point closest to an axial coordinate.

    Parameters
    ----------
    grid
        The grid.
    x
        The axial coordinate.

    Returns
    -------
    Node
        The closest grid point.
    """

    return min(grid.nodes.values(), key=lambda node: abs(node.coordinates[0] - x))


def run2D(nGridPoints=(51, 11), stressState: str = "plane strain", verbose: bool = True):
    """Solve the two dimensional bar under displacement control.

    Parameters
    ----------
    nGridPoints
        The number of grid points along the bar and across its height.
    stressState
        ``plane strain`` or ``plane stress``. The latter is the state of the EdelweissFE
        test and is therefore the one to compare peak loads with; note the caveat on the
        plane stress tangent in
        :mod:`edelweissfd.stencils.gradientenhanceddisplacementstencil`.
    verbose
        Be verbose during the simulation.

    Returns
    -------
    tuple
        The model tree and the field output controller.
    """

    sim = FDSimulation(domainSize=2, name="AT2PhaseFieldBar2D_" + stressState.replace(" ", ""), verbose=verbose)

    grid = sim.createStructuredGrid(lengths=[barLength, barHeight], nGridPoints=list(nGridPoints))

    sim.assignStencils(
        GradientEnhancedDisplacementStencil,
        grid,
        material=createMaterialFactory(sim),
        stressState=stressState,
    )

    step = sim.createStep(stepLength=1.0, maxInc=1e-2, minInc=1e-9, maxNumInc=3000, maxIter=25)
    step.addDirichlet("left", grid.nodeSets["left"], "displacement", {0: 0.0})
    step.addDirichlet("bottomLeft", grid.nodeSets["leftBottom"], "displacement", {1: 0.0})
    step.addDirichlet("right", grid.nodeSets["right"], "displacement", {0: endDisplacement})

    _addOutputs(sim, grid)

    return sim.run()


def _addOutputs(sim: FDSimulation, grid):
    """Request the load-displacement history and the field distributions."""

    sim.addNodeFieldOutput("displacement", "displacement", "U", nodeSet=grid.nodeSets["all"])
    sim.addNodeFieldOutput("phaseField", "nonlocal damage", "U", nodeSet=grid.nodeSets["all"])

    sim.addNodeFieldOutput(
        "reactionForce",
        "displacement",
        "P",
        nodeSet=grid.nodeSets["right"],
        saveHistory=True,
        f_x=lambda x: np.sum(x[:, 0]),
        export="reactionForce",
    )
    sim.addNodeFieldOutput(
        "endDisplacement",
        "displacement",
        "U",
        nodeSet=grid.nodeSets["right"],
        saveHistory=True,
        f_x=lambda x: np.mean(x[:, 0]),
        export="endDisplacement",
    )


def report(name: str, model, fieldOutputs):
    """Print the peak load and the width of the localized zone.

    Parameters
    ----------
    name
        The name of the variant.
    model
        The model tree.
    fieldOutputs
        The field output controller.

    Returns
    -------
    dict
        The peak load, the displacement at the peak and the localization width.
    """

    force = np.array(fieldOutputs.fieldOutputs["reactionForce"].getResultHistory()).flatten()
    displacement = np.array(fieldOutputs.fieldOutputs["endDisplacement"].getResultHistory()).flatten()

    phaseField = fieldOutputs.fieldOutputs["phaseField"].getLastResult().flatten()
    coordinates = np.array([node.coordinates[0] for node in model.nodeFields["nonlocal damage"].nodes])

    peak = int(np.argmax(np.abs(force)))

    #: A crack is only meaningfully localized once the phase field has grown substantially;
    #: below that the profile is still smooth and a width would be meaningless.
    localizationThreshold = 0.5

    isLocalized = phaseField.max() > localizationThreshold

    if isLocalized:
        localized = coordinates[phaseField > 0.5 * phaseField.max()]
        width = localized.max() - localized.min()
    else:
        width = float("nan")

    print()
    print("=== {:} ===".format(name))
    print("increments solved     : {:}".format(len(force)))
    print("peak load             : {:.6f}".format(np.abs(force[peak])))
    print("displacement at peak  : {:.6f}".format(displacement[peak]))
    print("final load            : {:.6f}".format(np.abs(force[-1])))
    print("max phase field       : {:.6f}".format(phaseField.max()))
    print(
        "phase field maximum at: {:.4f} (weakened zone centre {:.4f})".format(
            coordinates[np.argmax(phaseField)], 0.5 * barLength
        )
    )

    if isLocalized:
        print("localization width    : {:.4f} (length scale {:.4f})".format(width, lengthScale))
    else:
        print("localization width    : not localized yet, max phase field below {:}".format(localizationThreshold))

    return dict(peakLoad=np.abs(force[peak]), width=width, maxPhaseField=phaseField.max())


def main():
    model1D, fieldOutputs1D = run1D(verbose=False)
    report("1D bar, displacement control", model1D, fieldOutputs1D)

    modelArc, fieldOutputsArc = run1DArcLength(verbose=False)
    report("1D bar, indirect (crack opening) control", modelArc, fieldOutputsArc)

    model2D, fieldOutputs2D = run2D(stressState="plane strain", verbose=False)
    report("2D bar, plane strain", model2D, fieldOutputs2D)

    modelPlaneStress, fieldOutputsPlaneStress = run2D(stressState="plane stress", verbose=False)
    results = report("2D bar, plane stress", modelPlaneStress, fieldOutputsPlaneStress)

    print()
    print(
        "cross-check: EdelweissFE testfiles/marmot/AT2PhaseField peaks at 64.5679, "
        "EdelweissFD at {:.4f} ({:+.2f} %)".format(results["peakLoad"], 100.0 * (results["peakLoad"] / 64.5679 - 1.0))
    )


if __name__ == "__main__":
    main()
