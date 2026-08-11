#!/usr/bin/env python3
"""The B-bar treatment that keeps a cell from locking at nearly incompressible Poisson ratios.

Corner sampling makes the strain operators the bilinear ones evaluated at the cell vertices, so
they constrain the volumetric strain once per corner. As Poisson's ratio approaches one half that
over-constrains the cell: it can no longer deform at constant volume, which is exactly what a
shear band has to do. Averaging the volumetric part over the cell leaves one volumetric constraint
per cell and cures it, while the deviatoric part stays sampled at the corners so the hourglass
mode is still controlled.
"""

import numpy as np
import pytest

from edelweissfd.operators.differences import (
    cellCornerGradientOperators,
    cellStrainOperator,
    nVoigtComponents,
    volumetricallyAveragedStrainOperators,
)


def cellOperators(spacings=(2.0, 3.0)):
    """The corner sampled strain operators of one cell."""

    gradients = cellCornerGradientOperators(np.asarray(spacings, dtype=float))

    return [cellStrainOperator(gradient) for gradient in gradients]


def volumetricStrain(B, u):
    """The trace of the strain the operator produces."""

    return float(np.sum((B @ u)[:3]))


def deviatoricStrain(B, u):
    """The deviatoric part of the strain the operator produces."""

    strain = B @ u
    deviatoric = strain.copy()
    deviatoric[:3] -= np.sum(strain[:3]) / 3.0

    return deviatoric


@pytest.mark.parametrize("spacings", [(1.0,), (2.0, 3.0), (2.0, 3.0, 0.5)])
def test_theDeviatoricPartIsUntouched(spacings):
    """Only the volumetric part is averaged, which is what keeps the hourglass control intact."""

    operators = cellOperators(spacings)
    averaged = volumetricallyAveragedStrainOperators(operators)

    rng = np.random.default_rng(3)
    u = rng.normal(size=operators[0].shape[1])

    for B, Bbar in zip(operators, averaged):
        assert deviatoricStrain(Bbar, u) == pytest.approx(deviatoricStrain(B, u), abs=1e-12)


def test_everyMaterialPointSeesTheSameVolumetricStrain():
    """The point of the treatment: one volumetric constraint per cell instead of one per corner."""

    operators = cellOperators((2.0, 3.0))
    averaged = volumetricallyAveragedStrainOperators(operators)

    rng = np.random.default_rng(4)
    u = rng.normal(size=operators[0].shape[1])

    volumetric = [volumetricStrain(Bbar, u) for Bbar in averaged]

    assert volumetric == pytest.approx([volumetric[0]] * len(volumetric), abs=1e-12)

    # and the corner sampled operators really did disagree, i.e. the test above is not vacuous
    assert np.std([volumetricStrain(B, u) for B in operators]) > 1e-3


def test_theCellAverageOfTheVolumetricStrainIsPreserved():
    """Averaging must not change the mean, only redistribute it -- otherwise the cell would
    dilate or contract spuriously."""

    operators = cellOperators((2.0, 3.0))
    averaged = volumetricallyAveragedStrainOperators(operators)

    rng = np.random.default_rng(5)
    u = rng.normal(size=operators[0].shape[1])

    before = np.mean([volumetricStrain(B, u) for B in operators])
    after = np.mean([volumetricStrain(Bbar, u) for Bbar in averaged])

    assert after == pytest.approx(before, abs=1e-12)


def test_aHomogeneousStateIsUnaffected():
    """For a uniform strain all corner operators already agree, so averaging changes nothing. This
    is why the uniform benchmarks of this package are insensitive to the treatment."""

    spacings = np.array([2.0, 3.0])
    operators = cellOperators(spacings)
    averaged = volumetricallyAveragedStrainOperators(operators)

    # a linear displacement field over the cell corners gives a uniform strain
    from edelweissfd.operators.differences import cellCornerOffsets

    gradient = np.array([[0.01, -0.004], [-0.004, 0.02]])
    u = np.concatenate([gradient @ (offset * spacings) for offset in cellCornerOffsets(2)])

    for B, Bbar in zip(operators, averaged):
        assert Bbar @ u == pytest.approx(B @ u, abs=1e-12)


def test_oneMaterialPointIsAFixedPoint():
    """In one dimension a cell owns a single material point, so there is nothing to average."""

    operators = cellOperators((1.0,))

    assert len(operators) == 2, "a 1D cell still has two corner operators"

    averaged = volumetricallyAveragedStrainOperators([operators[0]])

    assert averaged[0] == pytest.approx(operators[0], abs=1e-12)


def test_weightsAreNormalised():
    """Passing the material point volumes rather than fractions must give the same result."""

    operators = cellOperators((2.0, 3.0))

    equal = volumetricallyAveragedStrainOperators(operators)
    weighted = volumetricallyAveragedStrainOperators(operators, weights=[3.0] * len(operators))

    for a, b in zip(equal, weighted):
        assert a == pytest.approx(b, abs=1e-12)


def test_theOperatorStaysTheRightShape():
    """Shapes have to survive, since the stencils index the Voigt rows afterwards."""

    for spacings in [(1.0,), (2.0, 3.0), (2.0, 3.0, 0.5)]:
        operators = cellOperators(spacings)

        for Bbar in volumetricallyAveragedStrainOperators(operators):
            assert Bbar.shape == (nVoigtComponents, operators[0].shape[1])


@pytest.mark.marmot
@pytest.mark.parametrize("spacings, stressState", [([1.0, 2.0], "plane strain"), ([1.0, 2.0, 0.5], "3d")])
def test_theTangentStaysConsistentWithAveragingOn(spacings, stressState):
    """B-bar enters the residual and the tangent through the same operator, so consistency has to
    survive. Checked because the averaging is applied once in the constructor and it would be easy
    to average one of the two only."""

    from edelweissfe.materials.marmot.marmothypoelastic import MarmotHypoElasticMaterial

    from tests.test_tangents import buildStencil
    from edelweissfd.stencils.displacementstencil import DisplacementStencil

    material = MarmotHypoElasticMaterial("LINEARELASTIC", np.array([20000.0, 0.49]))

    stencil = buildStencil(
        DisplacementStencil,
        spacings,
        material,
        stressState=stressState,
        volumetricAveraging=True,
    )

    rng = np.random.default_rng(11)
    dU = rng.normal(scale=1e-3, size=stencil.nDof)

    stencil.assertTangentConsistent(dU, dU, 1.0, 1.0, perturbation=1e-9, relativeTolerance=1e-5)


def volumetricConstraintMatrix(nCells, spacings, volumetricAveraging: bool) -> tuple:
    """The rows expressing "the volumetric strain vanishes at this material point", assembled over
    a patch of cells, together with the number of free displacement degrees of freedom.

    This is the mechanism of locking stated exactly, without any material or solver in the way. A
    deformation is volume preserving if and only if it lies in the null space of this matrix, so the
    dimension of that null space is the number of independent volume preserving deformations the
    discretisation admits at all. A shear band is such a deformation.
    """

    from edelweissfd.grids.structuredgrid import StructuredGrid
    from edelweissfd.stencils.displacementstencil import DisplacementStencil

    nDim = len(nCells)
    lengths = [n * h for n, h in zip(nCells, spacings)]

    grid = StructuredGrid("g", lengths, [n + 1 for n in nCells])

    nodeIndex = {id(node): i for i, node in enumerate(grid.nodes.values())}
    nDof = len(nodeIndex) * nDim

    trace = np.zeros(nVoigtComponents)
    trace[:3] = 1.0

    rows = []

    for cellIndex in grid.cellIndices():
        stencil = DisplacementStencil(1, spacings, stressState="plane strain", volumetricAveraging=volumetricAveraging)
        stencil.setCell(grid, cellIndex)

        columns = [nodeIndex[id(node)] * nDim + d for node in stencil.nodes for d in range(nDim)]

        for B in stencil._strainOperators:
            row = np.zeros(nDof)
            row[columns] = trace @ B
            rows.append(row)

    # clamp the bottom vertically and pin one grid point horizontally, as the panel example does
    constrained = {nodeIndex[id(node)] * nDim + 1 for node in grid.nodeSets["bottom"]}
    constrained.add(nodeIndex[id(grid.nodeSets["rightBottom"][0])] * nDim)

    free = [dof for dof in range(nDof) if dof not in constrained]

    return np.array(rows)[:, free], len(free)


def nullity(matrix, nFree) -> int:
    """The dimension of the space of volume preserving deformations the discretisation admits."""

    return int(nFree - np.linalg.matrix_rank(matrix, tol=1e-9))


def test_averagingLeavesExactlyOneIndependentConstraintPerCell():
    """With the volumetric part averaged the bookkeeping is exact: every cell contributes one
    independent constraint and nothing more, so the volume preserving deformations span the free
    degrees of freedom minus the cells."""

    for nCells in [(3, 3), (4, 4), (6, 6), (10, 20)]:
        relieved, nFree = volumetricConstraintMatrix(nCells, [1.0, 1.0], volumetricAveraging=True)

        assert nullity(relieved, nFree) == nFree - nCells[0] * nCells[1], nCells


def test_withoutAveragingTheVolumePreservingDeformationsAreCrowdedOut():
    """The mechanism of locking, stated exactly and without a material or a solver in the way.

    Corner sampling imposes one volumetric constraint per corner, four per cell in two dimensions,
    against roughly two degrees of freedom per grid point. The constraints outnumber the freedoms
    and almost every volume preserving deformation is squeezed out -- and a shear band is volume
    preserving, so it cannot form however the material behaves.

    What makes it locking rather than a fixed inaccuracy is how the two scale. Measured here:

        cells    free   nullity B   nullity B-bar
        3x3        27           4              18
        4x4        44           5              28
        6x6        90           7              54
        8x8       152           9              88
        10x20     450          21             250

    The relieved nullity grows with the degrees of freedom, the locked one only with the linear
    dimension of the grid. So the *fraction* of volume preserving deformations that survive tends
    to zero under refinement: refining the grid makes locking worse, not better, which is why a
    locked computation's localisation band does not settle down as the grid is refined.
    """

    fractions = {}

    for nCells in [(3, 3), (4, 4), (6, 6), (8, 8)]:
        locked, nFree = volumetricConstraintMatrix(nCells, [1.0, 1.0], volumetricAveraging=False)
        relieved, _ = volumetricConstraintMatrix(nCells, [1.0, 1.0], volumetricAveraging=True)

        lockedNullity = nullity(locked, nFree)
        relievedNullity = nullity(relieved, nFree)

        # ratios measured: 0.22, 0.18, 0.13, 0.10 -- already small on the coarsest patch and
        # falling, which is the trend asserted below
        assert lockedNullity < 0.3 * relievedNullity, (nCells, lockedNullity, relievedNullity)

        fractions[nCells] = (lockedNullity / nFree, relievedNullity / nFree)

    lockedFractions = [fraction for fraction, _ in fractions.values()]
    relievedFractions = [fraction for _, fraction in fractions.values()]

    # the locked fraction shrinks monotonically under refinement, the relieved one does not
    assert lockedFractions == sorted(lockedFractions, reverse=True), fractions
    assert min(relievedFractions) > 0.55, fractions
    assert lockedFractions[-1] < 0.5 * lockedFractions[0], fractions
