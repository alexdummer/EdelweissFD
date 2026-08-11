#!/usr/bin/env python3
"""The finite difference Laplacian of the grid, including its ghost node boundary treatment.

This operator is what lets EdelweissFD solve a gradient plasticity problem as a genuine two
field problem: the material asks to be told the Laplacian of the plastic multiplier, and a
difference operator can simply provide it. A C0 finite element cannot, which is why Marmot's
own ``C0GradientPlasticityFiniteElement`` carries a third field for the gradient of the plastic
multiplier and ties it down with a penalty.

At a boundary the missing neighbour is a ghost node, eliminated by the homogeneous Neumann
condition through mirroring across the boundary grid point.
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


def test_ghostNodeMirrorsAtABoundary():
    """The homogeneous Neumann condition doubles the coefficient of the inward neighbour."""

    grid = StructuredGrid("g", [4.0], [5])  # spacing 1

    atBoundary = coefficientsByIndex(grid, grid.nodeAt(0))
    assert atBoundary == pytest.approx({(0,): -2.0, (1,): 2.0})

    atOtherBoundary = coefficientsByIndex(grid, grid.nodeAt(4))
    assert atOtherBoundary == pytest.approx({(4,): -2.0, (3,): 2.0})

    inside = coefficientsByIndex(grid, grid.nodeAt(2))
    assert inside == pytest.approx({(1,): 1.0, (2,): -2.0, (3,): 1.0})


def test_ghostNodeMirrorsInBothDirectionsAtACorner():
    """At a corner both directions are mirrored."""

    grid = StructuredGrid("g", [2.0, 2.0], [3, 3])

    coefficients = coefficientsByIndex(grid, grid.nodeAt(0, 0))

    assert coefficients == pytest.approx({(0, 0): -4.0, (1, 0): 2.0, (0, 1): 2.0})


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
