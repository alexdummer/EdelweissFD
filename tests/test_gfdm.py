#!/usr/bin/env python3
"""The Generalized Finite Difference (GFDM) neighbour-cloud operators and the nodal gradient
plasticity stencil built from them, see :mod:`edelweissfd.operators.gfdm`.
"""

import numpy as np
import pytest

from edelweissfd.drivers.pythonscriptedsimulation import FDSimulation
from edelweissfd.grids.structuredgrid import StructuredGrid
from edelweissfd.operators.gfdm import gatherCloud, gfdmWeights
from edelweissfd.stencils.gfdmgradientplasticitystencil import (
    GFDMGradientPlasticityStencil,
)


def gradientVonMisesProperties(
    youngsModulus=20000.0,
    poissonsRatio=0.3,
    yieldStrength=100.0,
    hardeningModulus=-200.0,
    internalLength=8.0,
    useFischerBurmeister=1.0,
):
    """The property vector of Marmot's ``GRADIENTVONMISES``, see
    ``tests/test_gradientplasticity.py`` for the same helper."""

    return [
        youngsModulus,
        poissonsRatio,
        yieldStrength,
        hardeningModulus,
        abs(hardeningModulus) * internalLength**2,
        useFischerBurmeister,
        2.4e-9,
        0.0,
    ]


def buildNodalGrid(nGridPoints, lengths, stressState, domainSize=None, **properties):
    """A small grid covered with GFDM gradient plasticity stencils, one per node."""

    domainSize = domainSize if domainSize is not None else len(nGridPoints)
    sim = FDSimulation(domainSize=domainSize, name="gfdm", verbose=False)

    grid = sim.createStructuredGrid(lengths=list(lengths), nGridPoints=list(nGridPoints))

    material = sim.createMaterial(
        "GradientVonMises", gradientVonMisesProperties(**properties), baseClass="gradientPlasticityHypoElastic"
    )

    stencils = sim.assignNodalStencils(GFDMGradientPlasticityStencil, grid, material=material, stressState=stressState)

    for stencil in stencils:
        stencil.initializeStencil()

    return sim, grid, stencils


def pickStencil(stencils, nodePosition):
    """The stencil with the smallest (a boundary node's one-sided cloud) or largest (an
    interior node's two-sided cloud) molecule."""

    key = min if nodePosition == "boundary" else max
    return key(stencils, key=lambda s: s.nNodes)


def randomIncrement(stencil, seed, displacementScale=4e-3, multiplierScale=5e-4):
    """A displacement increment plus a strictly non-negative plastic multiplier increment,
    since the multiplier is a monotonic loading history variable."""

    rng = np.random.default_rng(seed)

    dU = np.zeros(stencil.nDof)
    dU[stencil._displacementDofs] = rng.normal(scale=displacementScale, size=stencil._displacementDofs.size)
    dU[stencil._multiplierDofs] = np.abs(rng.normal(scale=multiplierScale, size=stencil._multiplierDofs.size))

    return dU


# -- gatherCloud ------------------------------------------------------------------------------


@pytest.mark.parametrize("nGridPoints,minPoints", [((11,), 4), ((11, 11), 8)])
def test_gatherCloudFindsEnoughNeighboursAtAnInteriorPoint(nGridPoints, minPoints):
    """An interior point must be able to gather its cloud without widening beyond what the
    immediate neighbourhood already offers."""

    grid = StructuredGrid("g", [1.0] * len(nGridPoints), list(nGridPoints))
    centre = grid.nodeAt(*[n // 2 for n in nGridPoints])

    neighbours = gatherCloud(grid, centre, minPoints=minPoints)

    assert len(neighbours) >= minPoints
    assert centre not in neighbours


def test_gatherCloudWidensAtABoundary():
    """A boundary point only has a one-sided neighbourhood, so it must reach further than an
    interior point needs to, to gather the same cloud size."""

    grid = StructuredGrid("g", [1.0, 1.0], [11, 11])

    cornerNeighbours = gatherCloud(grid, grid.nodeAt(0, 0), minPoints=8)
    interiorNeighbours = gatherCloud(grid, grid.nodeAt(5, 5), minPoints=8)

    cornerRadius = np.max(np.abs(np.array([grid.gridIndexOf(n) for n in cornerNeighbours])))
    interiorRadius = np.max(np.abs(np.array([grid.gridIndexOf(n) for n in interiorNeighbours]) - 5))

    assert cornerRadius > interiorRadius


def test_gatherCloudRaisesWhenTheGridIsTooSmall():
    """A grid smaller than the requested cloud must fail loudly rather than silently return
    an undersized, rank deficient cloud."""

    grid = StructuredGrid("g", [1.0, 1.0], [2, 2])

    with pytest.raises(ValueError):
        gatherCloud(grid, grid.nodeAt(0, 0), minPoints=8, maxRadius=2)


def test_oneDimensionalBoundaryCloudFitsWithinTheDefaultSearchRadius():
    """A 1D boundary point can only gather one-sided, so a cloud size that needs more points
    than :func:`gatherCloud`'s default ``maxRadius`` allows one-sided must fail -- this is
    exactly why :class:`~edelweissfd.stencils.gfdmgradientplasticitystencil.GFDMGradientPlasticityStencil`
    picks a smaller default cloud size in one dimension."""

    grid = StructuredGrid("g", [1.0], [20])

    gatherCloud(grid, grid.nodeAt(0), minPoints=4)  # the 1D default: does not raise

    with pytest.raises(ValueError):
        gatherCloud(grid, grid.nodeAt(0), minPoints=8)  # the 2D default: unreachable one-sided


# -- gfdmWeights ------------------------------------------------------------------------------


@pytest.mark.parametrize("nDim", [1, 2, 3])
def test_gfdmWeightsReproduceAQuadraticFieldExactly(nDim):
    """The design matrix spans the full quadratic Taylor basis, so the weighted least squares
    fit is not merely accurate but exact whenever the sampled field truly is quadratic --
    regardless of the distance weighting, since the true parameters then leave zero residual."""

    rng = np.random.default_rng(2)

    centre = rng.uniform(-1.0, 1.0, size=nDim)
    neighbours = centre + rng.uniform(-1.0, 1.0, size=(20, nDim))

    coeffLinear = rng.uniform(-1.0, 1.0, size=nDim)
    coeffQuadratic = rng.uniform(0.5, 1.5, size=nDim)  # pure square terms only

    def field(x):
        d = x - centre
        return d @ coeffLinear + 0.5 * np.sum(coeffQuadratic * d**2, axis=-1)

    gradientWeights, laplacianWeights = gfdmWeights(centre, neighbours)

    values = field(neighbours)  # field(centre) is exactly zero by construction, no subtraction needed

    assert np.allclose(gradientWeights @ values, coeffLinear, atol=1e-9)
    assert np.isclose(laplacianWeights @ values, coeffQuadratic.sum(), atol=1e-9)


def test_gfdmWeightsRaisesForARankDeficientCloud():
    """A cloud collinear along one axis cannot determine the transverse derivatives of a 2D
    quadratic model -- the corresponding design matrix columns are all zero -- and must raise
    rather than silently return an unstable, wildly wrong answer."""

    centre = np.array([0.0, 0.0])
    neighbours = np.stack([np.linspace(-3.0, 3.0, 10), np.zeros(10)], axis=1)
    neighbours = neighbours[np.abs(neighbours[:, 0]) > 1e-9]

    with pytest.raises(ValueError):
        gfdmWeights(centre, neighbours)


# -- GFDMGradientPlasticityStencil -------------------------------------------------------------


def test_onlyOneMaterialPointPerStencil():
    """Unlike the corner-sampled stencil, there is exactly one material point: the node
    itself."""

    sim, grid, stencils = buildNodalGrid((5, 5), (12.0, 12.0), "plane strain")

    assert all(stencil.nMaterialPoints == 1 for stencil in stencils)
    assert all(
        fieldsAtNode == ["displacement", "plastic multiplier"]
        for stencil in stencils
        for fieldsAtNode in stencil.fields
    )


def test_boundaryNodesHaveASmallerMoleculeThanInteriorNodes():
    """A boundary node's cloud is gathered one-sided, so it should not need to reach as wide a
    molecule as an interior node's two-sided cloud of the same minimum size."""

    sim, grid, stencils = buildNodalGrid((9, 9), (16.0, 16.0), "plane strain")

    boundary = min(stencils, key=lambda s: s.nNodes)
    interior = max(stencils, key=lambda s: s.nNodes)

    assert boundary.nNodes <= interior.nNodes


@pytest.mark.marmot
@pytest.mark.parametrize("nodePosition", ["boundary", "interior"])
@pytest.mark.parametrize("stressState", ["plane strain", "plane stress"])
def test_tangentIsConsistentInTwoDimensions(nodePosition, stressState):
    """All coupling blocks, including plane stress's condensation -- see
    :func:`edelweissfd.operators.differences.condensePlaneStressTangents`."""

    sim, grid, stencils = buildNodalGrid((7, 7), (12.0, 12.0), stressState)

    stencil = pickStencil(stencils, nodePosition)

    dU = randomIncrement(stencil, seed=7)

    stencil.assertTangentConsistent(dU, dU, 1.0, 1.0, perturbation=1e-9, relativeTolerance=1e-5)


@pytest.mark.marmot
@pytest.mark.parametrize("nodePosition", ["boundary", "interior"])
def test_tangentIsConsistentInOneDimension(nodePosition):
    """The one dimensional case needs its own check: it is what motivated the dimension-aware
    default cloud size in the first place, see
    ``test_oneDimensionalBoundaryCloudFitsWithinTheDefaultSearchRadius``."""

    sim, grid, stencils = buildNodalGrid((9,), (12.0,), "3d", domainSize=1)

    stencil = pickStencil(stencils, nodePosition)

    dU = randomIncrement(stencil, seed=13)

    stencil.assertTangentConsistent(dU, dU, 1.0, 1.0, perturbation=1e-9, relativeTolerance=1e-5)


def test_onlyOneAndTwoDimensionsAreImplemented():
    with pytest.raises(ValueError):
        GFDMGradientPlasticityStencil(1, [1.0, 1.0, 1.0], stressState="3d")


def test_oneDimensionalGridNeedsStressState3d():
    with pytest.raises(ValueError):
        GFDMGradientPlasticityStencil(1, [1.0], stressState="plane strain")


def test_unknownStressStateRaises():
    with pytest.raises(ValueError):
        GFDMGradientPlasticityStencil(1, [1.0, 1.0], stressState="bogus")
