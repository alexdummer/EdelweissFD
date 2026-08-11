#!/usr/bin/env python3
"""The finite difference operators must reproduce derivatives to their formal order."""

import numpy as np
import pytest

from edelweissfd.operators.differences import (
    cellAverageOperator,
    cellCornerGradientOperators,
    cellCornerOffsets,
    cellGradientOperator,
    cellStrainOperator,
    finiteDifferenceCoefficients,
    stencilOffsets,
)


def test_classicalCoefficients():
    """The well known central difference quotients must come out exactly."""

    assert np.allclose(finiteDifferenceCoefficients([-1, 0, 1], 1), [-0.5, 0.0, 0.5])
    assert np.allclose(finiteDifferenceCoefficients([-1, 0, 1], 2), [1.0, -2.0, 1.0])
    assert np.allclose(finiteDifferenceCoefficients([0, 1, 2], 1), [-1.5, 2.0, -0.5])
    assert np.allclose(finiteDifferenceCoefficients([-2, -1, 0, 1, 2], 1), [1 / 12, -2 / 3, 0.0, 2 / 3, -1 / 12])


def test_coefficientsScaleWithSpacing():
    """A derivative of order m scales with h to the power of -m."""

    spacing = 0.25

    for derivativeOrder in (1, 2, 3):
        reference = finiteDifferenceCoefficients([-2, -1, 0, 1, 2], derivativeOrder, 1.0)
        scaled = finiteDifferenceCoefficients([-2, -1, 0, 1, 2], derivativeOrder, spacing)

        assert np.allclose(scaled, reference / spacing**derivativeOrder)


@pytest.mark.parametrize("derivativeOrder", [1, 2, 3, 4])
@pytest.mark.parametrize("accuracyOrder", [2, 4])
def test_polynomialsAreDifferentiatedExactly(derivativeOrder, accuracyOrder):
    """A difference quotient of accuracy order a is exact for polynomials of degree
    m + a - 1, so differentiating x**m must give exactly m!."""

    offsets = stencilOffsets(derivativeOrder, accuracyOrder)
    spacing = 0.3

    coefficients = finiteDifferenceCoefficients(offsets, derivativeOrder, spacing)

    x0 = 1.7
    values = (x0 + offsets * spacing) ** derivativeOrder

    from math import factorial

    assert np.isclose(coefficients @ values, factorial(derivativeOrder), rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("derivativeOrder", [1, 2])
@pytest.mark.parametrize("accuracyOrder", [2, 4])
def test_formalOrderOfAccuracy(derivativeOrder, accuracyOrder):
    """Halving the spacing must reduce the error by 2**accuracyOrder."""

    offsets = stencilOffsets(derivativeOrder, accuracyOrder)

    def f(x):
        return np.sin(1.3 * x)

    def exactDerivative(x):
        return (1.3**derivativeOrder) * np.sin(1.3 * x + derivativeOrder * np.pi / 2)

    x0 = 0.7

    errors = []
    for spacing in (0.1, 0.05):
        coefficients = finiteDifferenceCoefficients(offsets, derivativeOrder, spacing)
        approximation = coefficients @ f(x0 + offsets * spacing)
        errors.append(abs(approximation - exactDerivative(x0)))

    observedOrder = np.log2(errors[0] / errors[1])

    assert observedOrder > accuracyOrder - 0.3


def test_stencilOffsetsShiftAtBoundaries():
    """Near a boundary the window is shifted into the domain, keeping enough points."""

    assert np.array_equal(stencilOffsets(1, 2), [-1, 0, 1])
    assert np.array_equal(stencilOffsets(1, 2, nAvailableLeft=0), [0, 1, 2])
    assert np.array_equal(stencilOffsets(1, 2, nAvailableRight=0), [-2, -1, 0])
    assert np.array_equal(stencilOffsets(2, 2, nAvailableLeft=0), [0, 1, 2, 3])

    with pytest.raises(ValueError):
        stencilOffsets(2, 2, nAvailableLeft=0, nAvailableRight=1)


def test_finiteDifferenceCoefficientsRejectBadInput():
    with pytest.raises(ValueError):
        finiteDifferenceCoefficients([0, 0, 1], 1)

    with pytest.raises(ValueError):
        finiteDifferenceCoefficients([0], 1)

    with pytest.raises(ValueError):
        finiteDifferenceCoefficients([], 0)


@pytest.mark.parametrize("nDim", [1, 2, 3])
def test_cellCornerOffsetsOrdering(nDim):
    """The last axis has to vary fastest, everything else relies on it."""

    corners = cellCornerOffsets(nDim)

    assert corners.shape == (2**nDim, nDim)
    assert np.array_equal(corners[0], np.zeros(nDim, dtype=int))
    assert np.array_equal(corners[-1], np.ones(nDim, dtype=int))

    if nDim == 2:
        assert np.array_equal(corners, [[0, 0], [0, 1], [1, 0], [1, 1]])


@pytest.mark.parametrize("nDim", [1, 2, 3])
def test_cellGradientIsExactForLinearFields(nDim):
    """Both the averaged and the one-sided cell gradients are exact for linear fields."""

    spacings = np.array([0.5, 0.25, 2.0])[:nDim]
    corners = cellCornerOffsets(nDim)

    slope = np.array([1.3, -0.7, 0.4])[:nDim]
    offset = 2.1

    cornerCoordinates = corners * spacings
    cornerValues = cornerCoordinates @ slope + offset

    averaged = cellGradientOperator(spacings)
    assert np.allclose(averaged @ cornerValues, slope)

    for oneSided in cellCornerGradientOperators(spacings):
        assert np.allclose(oneSided @ cornerValues, slope)


@pytest.mark.parametrize("nDim", [1, 2, 3])
def test_cellAverageIsExactForLinearFields(nDim):
    """The average of the corner values is the value at the cell centre."""

    spacings = np.array([0.5, 0.25, 2.0])[:nDim]
    corners = cellCornerOffsets(nDim)

    slope = np.array([1.3, -0.7, 0.4])[:nDim]
    offset = 2.1

    cornerValues = (corners * spacings) @ slope + offset

    centre = 0.5 * spacings

    assert np.isclose(cellAverageOperator(nDim) @ cornerValues, centre @ slope + offset)


def test_averagedGradientMissesHourglassMode():
    """The reason the stencils sample the cell corners: the averaged cell gradient
    annihilates the hourglass mode, the one-sided operators do not."""

    spacings = np.array([1.0, 1.0])
    corners = cellCornerOffsets(2)

    hourglass = np.array([(-1.0) ** (corner.sum()) for corner in corners])

    assert np.allclose(cellGradientOperator(spacings) @ hourglass, 0.0)

    oneSidedResponses = [np.max(np.abs(G @ hourglass)) for G in cellCornerGradientOperators(spacings)]

    assert min(oneSidedResponses) > 0.5


@pytest.mark.parametrize("nDim", [1, 2, 3])
def test_cellStrainOperatorReproducesSymmetricGradient(nDim):
    """The strain operator must return the symmetric gradient in Marmot's Voigt convention."""

    spacings = np.array([0.5, 0.25, 2.0])[:nDim]
    corners = cellCornerOffsets(nDim)

    gradient = cellGradientOperator(spacings)
    strainOperator = cellStrainOperator(gradient)

    rng = np.random.default_rng(7)
    displacementGradient = rng.normal(size=(nDim, nDim))

    cornerCoordinates = corners * spacings
    # a linear displacement field, so the discrete gradient is exact
    displacements = cornerCoordinates @ displacementGradient.T

    strain = strainOperator @ displacements.flatten()

    symmetricGradient = 0.5 * (displacementGradient + displacementGradient.T)

    expected = np.zeros(6)
    for v, (i, j) in enumerate(((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))):
        if i >= nDim or j >= nDim:
            continue
        expected[v] = symmetricGradient[i, j] * (1.0 if i == j else 2.0)

    assert np.allclose(strain, expected)
