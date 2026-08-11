#!/usr/bin/env python3
"""Reconstruction of ``testfiles/edelweiss-only/SimpleBeamQuad4``.

A simply supported beam, 2000 x 150, thickness 25, carrying a uniformly distributed load on
its top edge. This is the first case here whose deformation is **bending** rather than a
homogeneous stretch or shear, so it is the one that actually probes the accuracy of the
difference operators: a bending solution is not in their span and has to be reached by
refinement.

Euler-Bernoulli theory gives the mid span deflection quoted by the FE test,

    w = 5 q L^4 / (384 E I) = 22.792 mm,     I = t h^3 / 12

and the finite difference solution approaches it from below at about second order::

    20 x 2    w = 10.87   (-52 %)
    40 x 4    w = 17.79   (-22 %)
    80 x 8    w = 21.20   ( -7 %)
    160 x 16  w = 22.30   ( -2 %)
    320 x 32  w = 22.63   ( -1 %)

The stiffness on coarse meshes is the classical locking of linear kinematics in bending, the
same effect that makes the bilinear ``CPE4`` of the FE test too stiff on a coarse mesh. It is
a property of the discretization, not an error, and the convergence test below is what pins
it down.

The material is the *native* EdelweissFE ``linearelastic``, so this test needs no Marmot.
"""

import numpy as np
import pytest

from edelweissfd.drivers.pythonscriptedsimulation import FDSimulation
from edelweissfd.stencils.displacementstencil import DisplacementStencil

youngsModulus = 13000.0
poissonsRatio = 0.15

span = 2000.0
depth = 150.0
thickness = 25.0

#: The distributed load on the top edge, as force per unit length of the span.
distributedLoad = 10.0

#: The nodal force the FE test applies to every grid point of the top edge. It is
#: ``distributedLoad * span / 76``, i.e. the value that produces the distributed load above
#: on the 75 x 4 mesh of that test, and it is only meaningful on that mesh.
nodalForceOfTheFETest = -263.16

#: The mid span deflection of Euler-Bernoulli beam theory, quoted by the FE test.
analyticalDeflection = 22.792


def solve(nCells=(75, 4), loadKind: str = "traction", verbose: bool = False) -> tuple:
    """Solve the beam.

    Parameters
    ----------
    nCells
        The number of cells along the span and over the depth.
    loadKind
        ``traction`` applies the distributed load consistently, i.e. weighted by the
        tributary length of every grid point, which is mesh independent. ``nodalForces``
        reproduces the literal definition of the FE test, a fixed force on every top grid
        point, which is only equivalent on the 75 x 4 mesh.
    verbose
        Be verbose during the simulation.

    Returns
    -------
    tuple
        The model, the field output controller and the grid.
    """

    sim = FDSimulation(domainSize=2, name="SimpleBeam2D", verbose=verbose)

    grid = sim.createStructuredGrid(lengths=[span, depth], nGridPoints=[nCells[0] + 1, nCells[1] + 1])

    material = sim.createMaterial("linearelastic", [youngsModulus, poissonsRatio], provider="edelweiss")

    sim.assignStencils(DisplacementStencil, grid, material=material, stressState="plane strain", thickness=thickness)

    step = sim.createStep(stepLength=100.0, maxInc=1.0, minInc=1e-8, maxNumInc=1000, maxIter=25)

    # a pin at one end and a roller at the other, exactly as in the FE test
    step.addDirichlet("pin", grid.nodeSets["leftBottom"], "displacement", {0: 0.0, 1: 0.0})
    step.addDirichlet("roller", grid.nodeSets["rightBottom"], "displacement", {1: 0.0})

    if loadKind == "traction":
        step.addNeumann("load", grid, "top", "displacement", [0.0, -distributedLoad])
    elif loadKind == "nodalForces":
        step.addNodeForces("load", grid.nodeSets["top"], "displacement", {0: 0.0, 1: nodalForceOfTheFETest})
    else:
        raise ValueError("unknown load kind '{:}'".format(loadKind))

    sim.addNodeFieldOutput("displacement", "displacement", "U", nodeSet=grid.nodeSets["all"])
    sim.addNodeFieldOutput("reaction", "displacement", "P", nodeSet=grid.nodeSets["bottom"])

    model, fieldOutputs = sim.run()

    return model, fieldOutputs, grid


def midSpanDeflection(model, fieldOutputs) -> float:
    """The largest downward deflection on the grid line closest to mid span.

    Parameters
    ----------
    model
        The model tree.
    fieldOutputs
        The field output controller.

    Returns
    -------
    float
        The magnitude of the mid span deflection.
    """

    displacements = fieldOutputs.fieldOutputs["displacement"].getLastResult()
    coordinates = np.array([node.coordinates for node in model.nodeFields["displacement"].nodes])

    # an odd number of cells leaves no grid point exactly at mid span
    distanceToMidSpan = np.abs(coordinates[:, 0] - 0.5 * span)
    atMidSpan = distanceToMidSpan <= distanceToMidSpan.min() + 1e-9

    return float(np.max(np.abs(displacements[atMidSpan, 1])))


def test_deflectionConvergesToBeamTheory():
    """The error has to shrink monotonically under refinement and end up within one percent."""

    deflections = [midSpanDeflection(*solve(nCells=(n, n // 10))[0:2]) for n in (20, 40, 80, 160, 320)]

    errors = [abs(deflection - analyticalDeflection) for deflection in deflections]

    assert all(later < earlier for earlier, later in zip(errors, errors[1:])), errors

    # approached from below, i.e. the discretization is too stiff, never too soft
    assert all(deflection < analyticalDeflection for deflection in deflections)

    assert deflections[-1] == pytest.approx(analyticalDeflection, rel=0.01)


def test_convergenceIsAtLeastFirstOrder():
    """Halving the spacing has to reduce the error by clearly more than a constant factor."""

    coarse = midSpanDeflection(*solve(nCells=(80, 8))[0:2])
    fine = midSpanDeflection(*solve(nCells=(160, 16))[0:2])
    finest = midSpanDeflection(*solve(nCells=(320, 32))[0:2])

    errors = [abs(deflection - analyticalDeflection) for deflection in (coarse, fine, finest)]

    observedRates = [np.log2(errors[i] / errors[i + 1]) for i in range(2)]

    assert min(observedRates) > 1.0, observedRates


def test_totalLoadIsMeshIndependentAndCarriedByTheSupports():
    """A consistently applied traction has to sum to the exact total load on every mesh, and
    the supports have to carry it."""

    totalAppliedLoad = -distributedLoad * span

    for nCells in [(20, 2), (75, 4), (80, 8)]:
        model, fieldOutputs, grid = solve(nCells=nCells)

        reaction = fieldOutputs.fieldOutputs["reaction"].getLastResult()

        # P holds the internal flux in positive sense, which balances the applied load
        assert reaction[:, 1].sum() == pytest.approx(-totalAppliedLoad, rel=1e-9)


def test_literalNodalForceDefinitionOfTheFETestAgrees():
    """On the 75 x 4 mesh of the FE test its fixed nodal force must be equivalent to the
    consistent traction, up to the different weighting of the two end grid points."""

    withTraction = midSpanDeflection(*solve(nCells=(75, 4), loadKind="traction")[0:2])
    withNodalForces = midSpanDeflection(*solve(nCells=(75, 4), loadKind="nodalForces")[0:2])

    assert withNodalForces == pytest.approx(withTraction, rel=0.02)


def test_deflectionIsSymmetricAboutMidSpan():
    """A symmetric beam under a symmetric load deflects symmetrically."""

    model, fieldOutputs, grid = solve(nCells=(30, 2))

    displacements = fieldOutputs.fieldOutputs["displacement"].getLastResult()
    coordinates = np.array([node.coordinates for node in model.nodeFields["displacement"].nodes])

    scale = np.max(np.abs(displacements[:, 1]))

    for index, coordinate in enumerate(coordinates):
        mirrored = np.array([span - coordinate[0], coordinate[1]])
        match = np.argmin(np.linalg.norm(coordinates - mirrored, axis=1))

        assert displacements[index, 1] == pytest.approx(displacements[match, 1], abs=1e-8 * scale)
