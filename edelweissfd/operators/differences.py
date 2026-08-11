#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  ---------------------------------------------------------------------
#
#  _____    _      _              _         _____ _____ ____
# | ____|__| | ___| |_      _____(_)___ ___|  ___|  ___|  _ \
# |  _| / _` |/ _ \ \ \ /\ / / _ \ / __/ __| |_  | |_  | | | |
# | |__| (_| |  __/ |\ V  V /  __/ \__ \__ \  _| |  _| | |_| |
# |_____\__,_|\___|_| \_/\_/ \___|_|___/___/_|   |_|   |____/
#
#
#  Unit of Strength of Materials and Structural Analysis
#  University of Innsbruck,
#  2017 - today
#
#  Alexander Dummer alexander.dummer@uibk.ac.at
#
#  This file is part of EdelweissFD.
#
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#
#  The full text of the license can be found in the file LICENSE.md at
#  the top level directory of EdelweissFD.
#  ---------------------------------------------------------------------

"""
Finite difference operators on a uniform structured grid.

Two groups of functions live here.

**General difference quotients.** :func:`finiteDifferenceCoefficients` returns the
coefficients of an arbitrary difference quotient for an arbitrary set of grid offsets, and
:func:`stencilOffsets` picks the offsets: a symmetric window whenever the grid allows it,
a shifted window near the boundaries. These are used for post-processing, for boundary
expressions and by the tests which verify the formal order of accuracy.

**Cell operators.** The stencils of EdelweissFD evaluate the material at the *centre* of a
grid cell and difference the nodal values over the cell's corners. The reason is the
classical one: the wide, node-centred quotient :math:`(u_{i+1} - u_{i-1}) / 2h` decouples
the even from the odd grid points and admits spurious checkerboard modes, whereas the
compact quotient :math:`(u_{i+1} - u_{i}) / h` over a cell does not. Evaluating the
material once per cell moreover gives every material point, and thus every set of history
variables, exactly one owner.

:func:`cellGradientOperator` builds the gradient at the cell centre,
:func:`cellAverageOperator` the value at the cell centre, and :func:`cellStrainOperator`
the small strain operator in Voigt notation.

.. note::
    The Voigt convention used throughout EdelweissFD is the one of Marmot,

    .. math::
        (\\;11,\\; 22,\\; 33,\\; 12,\\; 13,\\; 23\\;)

    with engineering shear strains. It deliberately does **not** go through
    :mod:`edelweissfe.utils.voigtnotation`, whose native helpers order the shear
    components as ``12, 23, 13``.
"""

from itertools import product
from math import factorial

import numpy as np

#: The tensor index pair belonging to each Voigt component, in Marmot's convention.
voigtIndexPairs = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))

#: The number of components of a symmetric second order tensor in Voigt notation.
nVoigtComponents = len(voigtIndexPairs)


def finiteDifferenceCoefficients(offsets, derivativeOrder: int, spacing: float = 1.0) -> np.ndarray:
    """The coefficients of a finite difference quotient.

    The returned coefficients :math:`c_i` approximate

    .. math::
        \\frac{d^m f}{dx^m}(x) \\approx \\sum_i c_i \\, f(x + o_i \\, h)

    for the requested derivative order :math:`m`, the given offsets :math:`o_i` and the
    grid spacing :math:`h`. They are obtained by requiring the quotient to be exact for
    all monomials up to degree ``len(offsets) - 1``, which is the classical Vandermonde
    construction.

    Parameters
    ----------
    offsets
        The grid offsets of the stencil, in multiples of the spacing. Must be distinct.
    derivativeOrder
        The order of the derivative to be approximated.
    spacing
        The grid spacing.

    Returns
    -------
    np.ndarray
        The coefficients, one per offset.
    """

    offsets = np.asarray(offsets, dtype=float)

    if offsets.ndim != 1 or offsets.size == 0:
        raise ValueError("The offsets must be a non-empty one dimensional sequence.")

    if len(np.unique(offsets)) != offsets.size:
        raise ValueError("The offsets must be distinct.")

    if derivativeOrder < 0:
        raise ValueError("The derivative order must not be negative.")

    if offsets.size <= derivativeOrder:
        raise ValueError(
            "A derivative of order {:} needs at least {:} offsets, {:} were given.".format(
                derivativeOrder, derivativeOrder + 1, offsets.size
            )
        )

    # taylorSystem[k, i] = offsets[i]**k, i.e. the condition that the quotient reproduces
    # the k-th monomial exactly
    taylorSystem = np.vander(offsets, offsets.size, increasing=True).T

    rightHandSide = np.zeros(offsets.size)
    rightHandSide[derivativeOrder] = factorial(derivativeOrder)

    return np.linalg.solve(taylorSystem, rightHandSide) / spacing**derivativeOrder


def stencilOffsets(
    derivativeOrder: int,
    accuracyOrder: int = 2,
    nAvailableLeft: int = None,
    nAvailableRight: int = None,
) -> np.ndarray:
    """The grid offsets of a difference quotient of the requested accuracy.

    A symmetric window is returned whenever enough neighbours are available on both
    sides, since symmetry buys one order of accuracy for free. Otherwise the window is
    shifted into the domain, which is the usual treatment at a boundary.

    Parameters
    ----------
    derivativeOrder
        The order of the derivative to be approximated.
    accuracyOrder
        The requested formal order of accuracy.
    nAvailableLeft
        The number of available neighbours towards smaller coordinates. ``None`` means
        unlimited, i.e. an interior point.
    nAvailableRight
        The number of available neighbours towards larger coordinates. ``None`` means
        unlimited.

    Returns
    -------
    np.ndarray
        The offsets, in ascending order.
    """

    if accuracyOrder < 1:
        raise ValueError("The accuracy order must be at least one.")

    if nAvailableLeft is None:
        nAvailableLeft = derivativeOrder + accuracyOrder
    if nAvailableRight is None:
        nAvailableRight = derivativeOrder + accuracyOrder

    # A symmetric window of half width p spans 2p+1 points and, thanks to symmetry,
    # reaches an even order of accuracy of 2*ceil((2p+1-m)/2). Requesting order a
    # therefore needs p = ceil((a+m-2)/2).
    halfWidth = -(-(accuracyOrder + derivativeOrder - 2) // 2)

    if halfWidth <= nAvailableLeft and halfWidth <= nAvailableRight:
        return np.arange(-halfWidth, halfWidth + 1)

    # A shifted window has no symmetry to exploit and needs m + a points.
    nPoints = derivativeOrder + accuracyOrder

    if nPoints > nAvailableLeft + nAvailableRight + 1:
        raise ValueError(
            "A derivative of order {:} with accuracy order {:} needs {:} grid points, "
            "but only {:} are available.".format(
                derivativeOrder, accuracyOrder, nPoints, nAvailableLeft + nAvailableRight + 1
            )
        )

    left = min((nPoints - 1) // 2, nAvailableLeft)
    right = nPoints - 1 - left

    if right > nAvailableRight:
        right = nAvailableRight
        left = nPoints - 1 - right

    return np.arange(-left, right + 1)


def cellCornerOffsets(nDim: int) -> np.ndarray:
    """The offsets of the corners of a grid cell, relative to its lower left corner.

    The corners are ordered such that the last axis varies fastest, i.e. in 2D
    ``(0,0), (0,1), (1,0), (1,1)``. Everything else in EdelweissFD relies on this order.

    Parameters
    ----------
    nDim
        The spatial dimension.

    Returns
    -------
    np.ndarray
        The corner offsets, shape ``(2**nDim, nDim)``.
    """

    return np.array(list(product((0, 1), repeat=nDim)), dtype=int)


def cellAverageOperator(nDim: int) -> np.ndarray:
    """The operator evaluating a field at the centre of a grid cell.

    The centre value is the plain average of the corner values, which is the compact,
    second order accurate choice.

    Parameters
    ----------
    nDim
        The spatial dimension.

    Returns
    -------
    np.ndarray
        The coefficients, shape ``(2**nDim,)``.
    """

    nCorners = 2**nDim

    return np.full(nCorners, 1.0 / nCorners)


def cellGradientOperator(spacings) -> np.ndarray:
    """The operator evaluating the gradient of a field at the centre of a grid cell.

    The derivative along a direction is the average of the compact difference quotients
    along all cell edges parallel to that direction,

    .. math::
        \\left. \\frac{\\partial f}{\\partial x_d} \\right|_{centre}
        = \\frac{1}{2^{n-1}} \\sum_{\\text{edges} \\parallel x_d}
          \\frac{f_{far} - f_{near}}{h_d}

    which is second order accurate at the cell centre.

    Parameters
    ----------
    spacings
        The grid spacing per direction, shape ``(nDim,)``.

    Returns
    -------
    np.ndarray
        The coefficients, shape ``(nDim, 2**nDim)``.
    """

    spacings = np.asarray(spacings, dtype=float)
    nDim = spacings.size

    if np.any(spacings <= 0.0):
        raise ValueError("All grid spacings must be positive.")

    corners = cellCornerOffsets(nDim)
    nEdgesPerDirection = 2 ** (nDim - 1)

    gradient = np.zeros((nDim, corners.shape[0]))

    for d in range(nDim):
        # a corner sits at the far end of its edge if its offset along d is one
        sign = 2.0 * corners[:, d] - 1.0
        gradient[d, :] = sign / (spacings[d] * nEdgesPerDirection)

    return gradient


def cellCornerGradientOperators(spacings) -> list:
    """The one-sided gradient operators at the corners of a grid cell.

    The derivative along a direction is the plain one-sided difference quotient along the
    single cell edge which starts at that corner and points along that direction,

    .. math::
        \\left. \\frac{\\partial f}{\\partial x_d} \\right|_{corner}
        = \\frac{f_{next} - f_{previous}}{h_d}

    Each operator therefore involves only :math:`n+1` of the :math:`2^n` corners.

    In contrast to :func:`cellGradientOperator`, which averages over all parallel edges,
    these operators do **not** annihilate the hourglass (checkerboard) mode. Sampling a
    cell at its corners instead of only at its centre is what makes the two and three
    dimensional schemes of EdelweissFD free of spurious zero energy modes; in one
    dimension both choices coincide.

    Parameters
    ----------
    spacings
        The grid spacing per direction, shape ``(nDim,)``.

    Returns
    -------
    list
        One operator of shape ``(nDim, 2**nDim)`` per corner, ordered like
        :func:`cellCornerOffsets`.
    """

    spacings = np.asarray(spacings, dtype=float)
    nDim = spacings.size

    if np.any(spacings <= 0.0):
        raise ValueError("All grid spacings must be positive.")

    corners = cellCornerOffsets(nDim)
    cornerIndices = {tuple(corner): i for i, corner in enumerate(corners)}

    operators = []

    for corner in corners:
        gradient = np.zeros((nDim, corners.shape[0]))

        for d in range(nDim):
            previous = corner.copy()
            previous[d] = 0

            following = corner.copy()
            following[d] = 1

            gradient[d, cornerIndices[tuple(following)]] += 1.0 / spacings[d]
            gradient[d, cornerIndices[tuple(previous)]] -= 1.0 / spacings[d]

        operators.append(gradient)

    return operators


def cellStrainOperator(gradient: np.ndarray) -> np.ndarray:
    """The small strain operator of a grid cell in Voigt notation.

    Maps the displacement degrees of freedom of the cell's corners onto the six component
    Voigt strain at the cell centre. The degrees of freedom are ordered corner-major and
    component-minor, i.e. ``[corner 0 - x, corner 0 - y, corner 1 - x, ...]``, which is
    the layout the solvers of EdelweissFE expect.

    Components which do not exist for the given spatial dimension stay zero, so a 2D
    operator describes a state of plane strain and a 1D operator a state of uniaxial
    strain. Whether the out-of-plane components are subsequently condensed out is decided
    by the stencil, by choosing the corresponding material routine.

    Parameters
    ----------
    gradient
        The gradient operator of the cell, shape ``(nDim, nCorners)``.

    Returns
    -------
    np.ndarray
        The strain operator, shape ``(6, nCorners * nDim)``.
    """

    nDim, nCorners = gradient.shape

    strainOperator = np.zeros((nVoigtComponents, nCorners * nDim))

    for v, (i, j) in enumerate(voigtIndexPairs):
        if i >= nDim or j >= nDim:
            continue

        for corner in range(nCorners):
            if i == j:
                strainOperator[v, corner * nDim + i] += gradient[i, corner]
            else:
                # engineering shear strain, 2 * eps_ij
                strainOperator[v, corner * nDim + i] += gradient[j, corner]
                strainOperator[v, corner * nDim + j] += gradient[i, corner]

    return strainOperator
