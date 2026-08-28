#!/usr/bin/env python3
"""The finite difference Laplacian of the grid, including its Neumann boundary treatment.

This operator is what lets EdelweissFD solve a gradient plasticity problem as a genuine two
field problem: the material asks to be told the Laplacian of the plastic multiplier, and a
difference operator can simply provide it. A C0 finite element cannot, which is why Marmot's
own ``C0GradientPlasticityFiniteElement`` carries a third field for the gradient of the plastic
multiplier and ties it down with a penalty.

At a boundary the homogeneous Neumann condition is built in through a one-sided, ghost-free
formula using the boundary point and its two real interior neighbours -- see
:meth:`~edelweissfd.grids.structuredgrid.StructuredGrid.laplacianAt` for the derivation. A
straightforward ghost-node mirror rule also satisfies the Neumann condition itself, but is only
first order accurate for the resulting second derivative, which turned out to cap a whole
gradient plasticity solve at first order in mesh size -- see
``test_secondOrderAccurateForAGenericBoundaryCurvature`` below for a test field that actually
exercises the difference (``cos`` does not: its odd derivatives vanish at the boundary by
symmetry, which is why it passed under the old, buggy formula too).
"""

import numpy as np
import pytest

from edelweissfd.grids.structuredgrid import StructuredGrid


def coefficientsByIndex(grid: StructuredGrid, node) -> dict:
    """The Laplacian coefficients keyed by grid index rather than by node."""

    return {tuple(int(i) for i in grid.gridIndexOf(n)): c for n, c in grid.laplacianAt(node).items()}


def test_interiorIsTheClassicalFivePointStencil():
    """In two dimensions the interior Laplacian is [1, 1, -4, 1, 1] over the compact cross."""

    grid = StructuredGrid("g", [2.0, 2.0], [3, 3])

    coefficients = coefficientsByIndex(grid, grid.nodeAt(1, 1))

    assert coefficients == pytest.approx({(1, 1): -4.0, (0, 1): 1.0, (2, 1): 1.0, (1, 0): 1.0, (1, 2): 1.0})


def test_spacingEntersAsItsInverseSquare():
    """Each direction contributes with one over its own spacing squared."""

    grid = StructuredGrid("g", [2.0, 8.0], [3, 3])  # spacings 1 and 4

    coefficients = coefficientsByIndex(grid, grid.nodeAt(1, 1))

    assert coefficients[(0, 1)] == pytest.approx(1.0)
    assert coefficients[(1, 0)] == pytest.approx(1.0 / 16.0)
    assert coefficients[(1, 1)] == pytest.approx(-2.0 - 2.0 / 16.0)


def test_neumannBoundaryIsAOneSidedThreePointFormula():
    """The boundary formula reaches the two real interior neighbours, ghost-free.

    Coefficients (-3.5, 4.0, -0.5) at (self, first interior neighbour, second interior
    neighbour): the one-sided, second order accurate fit of a cubic through those three points
    with the Neumann condition imposed exactly, see :meth:`StructuredGrid.laplacianAt`.
    """

    grid = StructuredGrid("g", [4.0], [5])  # spacing 1

    atBoundary = coefficientsByIndex(grid, grid.nodeAt(0))
    assert atBoundary == pytest.approx({(0,): -3.5, (1,): 4.0, (2,): -0.5})

    atOtherBoundary = coefficientsByIndex(grid, grid.nodeAt(4))
    assert atOtherBoundary == pytest.approx({(4,): -3.5, (3,): 4.0, (2,): -0.5})

    inside = coefficientsByIndex(grid, grid.nodeAt(2))
    assert inside == pytest.approx({(1,): 1.0, (2,): -2.0, (3,): 1.0})


def test_neumannBoundaryFallsBackToGhostMirrorWhenTooThin():
    """A grid with only two points in a direction has no second interior neighbour to reach,
    so the one-sided formula cannot be built there; the ghost mirror rule still applies."""

    grid = StructuredGrid("g", [1.0], [2])  # spacing 1, only two points

    atBoundary = coefficientsByIndex(grid, grid.nodeAt(0))
    assert atBoundary == pytest.approx({(0,): -2.0, (1,): 2.0})


def test_neumannBoundaryInBothDirectionsAtACorner():
    """At a corner both directions contribute their own one-sided formula; in a grid only three
    points wide the "second interior neighbour" the formula reaches for is itself the opposite
    boundary point, which is still a perfectly valid third value to fit the cubic through."""

    grid = StructuredGrid("g", [2.0, 2.0], [3, 3])

    coefficients = coefficientsByIndex(grid, grid.nodeAt(0, 0))

    assert coefficients == pytest.approx({(0, 0): -7.0, (1, 0): 4.0, (2, 0): -0.5, (0, 1): 4.0, (0, 2): -0.5})


@pytest.mark.parametrize("nDim", [1, 2, 3])
def test_coefficientsAlwaysSumToZero(nDim):
    """A constant field has a vanishing Laplacian, on the boundary as well as inside. This is
    what the mirroring has to preserve and the reason it is not a one-sided window."""

    lengths = [6.0, 4.0, 2.0][:nDim]
    nGridPoints = [4, 3, 3][:nDim]

    grid = StructuredGrid("g", lengths, nGridPoints)

    for node in grid.nodes.values():
        assert sum(grid.laplacianAt(node).values()) == pytest.approx(0.0, abs=1e-12)


def test_exactForQuadraticFieldsInTheInterior():
    """The three point quotient differentiates a quadratic exactly."""

    grid = StructuredGrid("g", [10.0, 10.0], [11, 11])

    for node in grid.nodes.values():
        index = grid.gridIndexOf(node)

        if not all(0 < index[d] < grid.shape[d] - 1 for d in range(2)):
            continue

        laplacian = grid.laplacianAt(node)

        # laplace(x^2 + 3 y^2) = 2 + 6
        value = sum(c * (n.coordinates[0] ** 2 + 3.0 * n.coordinates[1] ** 2) for n, c in laplacian.items())

        assert value == pytest.approx(8.0)


def test_secondOrderAccurateForAFieldSatisfyingTheNeumannCondition():
    """For a field whose normal derivative vanishes at the boundary -- which is the condition
    the ghost nodes impose -- the error has to drop by four when the spacing is halved, at the
    boundary just as in the interior."""

    length = 1.0

    def field(x):
        return np.cos(np.pi * x / length)

    def exactLaplacian(x):
        return -((np.pi / length) ** 2) * np.cos(np.pi * x / length)

    errorsAtBoundary = []
    errorsInside = []

    for nCells in (20, 40, 80):
        grid = StructuredGrid("g", [length], [nCells + 1])

        # a quarter of the way in, deliberately not at the mid point: there the cosine and its
        # second difference both vanish identically and the "error" would be pure round off
        for index, errors in ((0, errorsAtBoundary), (nCells // 4, errorsInside)):
            node = grid.nodeAt(index)
            laplacian = grid.laplacianAt(node)

            approximation = sum(c * field(n.coordinates[0]) for n, c in laplacian.items())

            errors.append(abs(approximation - exactLaplacian(node.coordinates[0])))

    for errors in (errorsAtBoundary, errorsInside):
        rates = [np.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]

        assert min(rates) > 1.8, rates


def test_secondOrderAccurateForAGenericBoundaryCurvature():
    """``cos`` has odd derivatives that vanish at the boundary by symmetry, which is exactly
    the special case that hides a first order boundary error -- see the module docstring. A
    field with a generic, nonzero third derivative there is the test that actually catches it:
    the ghost mirror rule gives order 1 for this field, the one-sided formula order 2.
    """

    length = 1.0

    def field(x):
        return x**2 + x**3 + x**4  # u'(0) = 0 automatically (every term has degree >= 2), so
        # this satisfies the Neumann condition without being built to trivially cancel the
        # boundary formula's higher order terms the way an even function like cos does; its
        # third derivative at 0 is 6 != 0, which is exactly the term the ghost mirror rule drops

    def exactLaplacianAtZero():
        return 2.0

    errors = []
    for nCells in (20, 40, 80, 160):
        grid = StructuredGrid("g", [length], [nCells + 1])
        node = grid.nodeAt(0)
        laplacian = grid.laplacianAt(node)
        approximation = sum(c * field(n.coordinates[0]) for n, c in laplacian.items())
        errors.append(abs(approximation - exactLaplacianAtZero()))

    rates = [np.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]
    assert min(rates) > 1.8, rates
