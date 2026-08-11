#!/usr/bin/env python3
"""Reconstruction of ``testfiles/edelweiss-only/FixedDisplacementQuad4``.

A 25 x 150 block, clamped at the bottom and dragged at the top by a combined stretch and
shear, ``u = 3`` and ``v = 11.45``, with free sides. It is a purely displacement driven
problem, so it needs no load at all, and it is the first case here that produces a
substantial **shear** deformation rather than a uniaxial one.

The material is the *native* EdelweissFE ``linearelastic``, so this test needs no Marmot.

The block with free sides has no closed form solution, so the checks are the ones that must
hold for any correct solution: the prescribed values are met exactly, the deformation is
antisymmetric about the centre of the block, shear stress is actually present, and the
reaction forces are in equilibrium with each other.
"""

import numpy as np
import pytest

from edelweissfd.drivers.pythonscriptedsimulation import FDSimulation
from edelweissfd.stencils.displacementstencil import DisplacementStencil

youngsModulus = 30000.0
poissonsRatio = 0.15

width = 25.0
height = 150.0

prescribedShear = 3.0
prescribedStretch = 11.45


def solve(nGridPoints=(9, 21), verbose: bool = False) -> tuple:
    """Solve the block.

    Parameters
    ----------
    nGridPoints
        The number of grid points across the width and along the height. The default matches
        the 8 x 20 cells of the EdelweissFE test.
    verbose
        Be verbose during the simulation.

    Returns
    -------
    tuple
        The model, the field output controller and the grid.
    """

    sim = FDSimulation(domainSize=2, name="FixedDisplacement2D", verbose=verbose)

    grid = sim.createStructuredGrid(lengths=[width, height], nGridPoints=list(nGridPoints))

    material = sim.createMaterial("linearelastic", [youngsModulus, poissonsRatio], provider="edelweiss")

    stencils = sim.assignStencils(
        DisplacementStencil, grid, material=material, stressState="plane strain", thickness=1.0
    )

    step = sim.createStep(stepLength=100.0, maxInc=1.0, minInc=1e-8, maxNumInc=1000, maxIter=25)
    step.addDirichlet("bottom", grid.nodeSets["bottom"], "displacement", {0: 0.0, 1: 0.0})
    step.addDirichlet("top", grid.nodeSets["top"], "displacement", {0: prescribedShear, 1: prescribedStretch})

    sim.addNodeFieldOutput("displacement", "displacement", "U", nodeSet=grid.nodeSets["all"])
    sim.addNodeFieldOutput("reactionBottom", "displacement", "P", nodeSet=grid.nodeSets["bottom"])
    sim.addNodeFieldOutput("reactionTop", "displacement", "P", nodeSet=grid.nodeSets["top"])
    sim.addStencilFieldOutput("stress", "stress", stencils=stencils)

    model, fieldOutputs = sim.run()

    return model, fieldOutputs, grid


def test_prescribedDisplacementsAreMetExactly():
    """The Dirichlet boundary conditions have to be satisfied to round off."""

    model, fieldOutputs, grid = solve()

    displacements = fieldOutputs.fieldOutputs["displacement"].getLastResult()
    coordinates = np.array([node.coordinates for node in model.nodeFields["displacement"].nodes])

    atBottom = np.isclose(coordinates[:, 1], 0.0)
    atTop = np.isclose(coordinates[:, 1], height)

    assert np.allclose(displacements[atBottom], 0.0, atol=1e-12)
    assert np.allclose(displacements[atTop, 0], prescribedShear, atol=1e-12)
    assert np.allclose(displacements[atTop, 1], prescribedStretch, atol=1e-12)


def test_deformationIsPointSymmetricAboutTheCentre():
    """The geometry, the material and the boundary conditions are point symmetric about the
    centre of the block, so the displacement field has to be too."""

    model, fieldOutputs, grid = solve()

    displacements = fieldOutputs.fieldOutputs["displacement"].getLastResult()
    coordinates = np.array([node.coordinates for node in model.nodeFields["displacement"].nodes])

    centre = np.array([0.5 * width, 0.5 * height])
    prescribed = np.array([prescribedShear, prescribedStretch])

    for index, coordinate in enumerate(coordinates):
        mirrored = 2.0 * centre - coordinate
        match = np.argmin(np.linalg.norm(coordinates - mirrored, axis=1))

        assert np.allclose(displacements[index] + displacements[match], prescribed, atol=1e-9)


def test_customNodeSetFromAGeometricPredicate():
    """Reconstruction of the idea behind ``testfiles/marmot/PythonCodeModelGeneration``.

    That test uses the ``executePythonCode`` generator to build a node set from a geometric
    predicate, ``all nodes at 50 % of the height``. In EdelweissFD the model is Python already,
    so the same thing is a call to :meth:`~edelweissfd.drivers.pythonscriptedsimulation.FDSimulation.createNodeSet`.

    At mid height of this block the axial displacement has to be half of the prescribed one, by
    the point symmetry of the problem.
    """

    sim = FDSimulation(domainSize=2, name="FixedDisplacementCustomSet", verbose=False)

    grid = sim.createStructuredGrid(lengths=[width, height], nGridPoints=[9, 21])

    material = sim.createMaterial("linearelastic", [youngsModulus, poissonsRatio], provider="edelweiss")

    sim.assignStencils(DisplacementStencil, grid, material=material, stressState="plane strain")

    midHeight = sim.createNodeSet(
        "centerHorizontal",
        [node for node in grid.nodes.values() if abs(node.coordinates[1] - 0.5 * height) <= 1e-12],
    )

    assert len(midHeight) == 9, "the predicate has to select one full grid row"

    step = sim.createStep(stepLength=100.0)
    step.addDirichlet("bottom", grid.nodeSets["bottom"], "displacement", {0: 0.0, 1: 0.0})
    step.addDirichlet("top", grid.nodeSets["top"], "displacement", {0: prescribedShear, 1: prescribedStretch})

    sim.addNodeFieldOutput("midHeight", "displacement", "U", nodeSet=midHeight)

    model, fieldOutputs = sim.run()

    displacements = fieldOutputs.fieldOutputs["midHeight"].getLastResult()

    assert displacements.shape == (9, 2)

    assert displacements[:, 0].mean() == pytest.approx(0.5 * prescribedShear, rel=1e-9)
    assert displacements[:, 1].mean() == pytest.approx(0.5 * prescribedStretch, rel=1e-9)


def test_shearIsPresentAndReactionsAreInEquilibrium():
    """A shear driven problem must show shear stress, and with no external load the reactions
    at the top and at the bottom have to cancel."""

    model, fieldOutputs, grid = solve()

    # a field output over all material points of a stencil is shaped
    # (nStencils, nMaterialPoints, 6), so flatten to one row per material point first
    stress = np.asarray(fieldOutputs.fieldOutputs["stress"].getLastResult()).reshape(-1, 6)

    # Voigt component 3 is sigma_12 in Marmot's convention, which EdelweissFD follows
    assert np.max(np.abs(stress[:, 3])) > 0.01 * np.max(np.abs(stress[:, 0:2]))

    reactionBottom = fieldOutputs.fieldOutputs["reactionBottom"].getLastResult().sum(axis=0)
    reactionTop = fieldOutputs.fieldOutputs["reactionTop"].getLastResult().sum(axis=0)

    scale = max(np.max(np.abs(reactionBottom)), np.max(np.abs(reactionTop)))

    assert np.allclose(reactionBottom + reactionTop, 0.0, atol=1e-8 * scale)
