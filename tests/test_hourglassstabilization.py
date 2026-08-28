#!/usr/bin/env python3
"""The Flanagan-Belytschko orthogonal hourglass vector must have the two properties its
stabilization stiffness relies on: it must not react at all to a constant or linear field,
and its reaction to a smooth field with curvature must vanish at second order as the cell
shrinks -- see :func:`edelweissfd.operators.differences.hourglassVector`."""

import numpy as np
import pytest

from edelweissfd.operators.differences import cellCornerOffsets, hourglassVector


def test_hourglassVectorIsOrthogonalToConstantAndLinearFields():
    """gamma . u must vanish exactly for any affine field, since gamma is the checkerboard
    pattern with its constant and linear parts explicitly projected out."""

    rng = np.random.default_rng(0)
    corners = cellCornerOffsets(2).astype(float)

    for _ in range(20):
        spacings = rng.uniform(0.1, 3.0, size=2)
        gamma = hourglassVector(spacings)
        coordinates = corners * spacings

        a0, a1, a2 = rng.uniform(-1.0, 1.0, size=3)
        u = a0 + a1 * coordinates[:, 0] + a2 * coordinates[:, 1]

        assert abs(gamma @ u) < 1e-12


def test_hourglassVectorRespondsAtSecondOrderToCurvature():
    """For a smooth field with curvature, gamma . u must shrink as h**2 as the cell shrinks,
    matching the second order accuracy of the centre-sampled gradient operator it stabilizes
    without spoiling."""

    corners = cellCornerOffsets(2).astype(float)
    x0, y0 = 0.37, -0.21

    def field(x, y):
        return np.sin(x) * np.cos(y) + 0.5 * x * y**2

    hs = [0.1, 0.05, 0.025, 0.0125]
    responses = []
    for h in hs:
        spacings = np.array([h, h])
        gamma = hourglassVector(spacings)
        coordinates = np.array([x0, y0]) + corners * spacings
        responses.append(gamma @ field(coordinates[:, 0], coordinates[:, 1]))

    for coarse, fine in zip(responses[:-1], responses[1:]):
        assert np.isclose(coarse / fine, 4.0, rtol=0.1)


def test_hourglassVectorOnlyImplementedInTwoDimensions():
    """Documented as 2D only -- a trilinear hexahedron has three independent hourglass modes
    per component, not one, which this single vector does not cover."""

    with pytest.raises(ValueError):
        hourglassVector([1.0])

    with pytest.raises(ValueError):
        hourglassVector([1.0, 1.0, 1.0])
