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

#: The Voigt index of the out-of-plane normal component (zz), in the order above.
outOfPlaneVoigtIndex = 2


def condensePlaneStressTangents(
    D, dLambda, dLaplacian, dFdStrain, dFdLambda, dFdLaplacian, zz: int = outOfPlaneVoigtIndex
):
    """Statically condense out the zz row/column Marmot's ``computePlaneStress`` leaves in.

    Marmot's plane stress routine correctly condenses the *stress* -- sigma_zz = 0 is enforced by
    solving for the corresponding strain internally, verified directly against the closed form
    plane stress modulus -- but the algorithmic tangent blocks it returns are the raw, uncondensed
    ones: at nu=0.49, the returned ``dStress_dStrain`` is the full 3D elastic tangent, about 13x
    stiffer than the correct plane stress modulus E/(1-nu^2) (confirmed by differencing the
    material response directly). Left uncorrected, Newton's search direction is systematically
    too small everywhere plane stress is used, which shows up as slow (linear, not quadratic)
    convergence rather than an outright wrong answer -- easy to mistake for a genuine structural
    instability, since it can exceed the iteration budget right where the physics gets interesting
    (e.g. at the onset of yielding).

    The fix is the standard implicit function theorem static condensation: since sigma_zz is
    identically zero for every combination of the (in-plane strain, plastic multiplier increment,
    its Laplacian) ``computePlaneStress`` is called with, differentiating that identity through
    gives the dependence of the implicitly eliminated eps_zz on each of them, which every other
    tangent block must include to stay consistent. Verified to reduce the deviation from a central
    difference of the material response from order one down to 1e-8..1e-13 across the yield
    transition, at several plastic multiplier increments each.

    Parameters
    ----------
    D
        ``dStress_dStrain``, shape ``(..., 6, 6)``.
    dLambda, dLaplacian
        ``dStress_dLambda`` and ``dStress_dLaplacian``, shape ``(..., 6)``.
    dFdStrain
        Shape ``(..., 6)``.
    dFdLambda, dFdLaplacian
        Shape ``(...,)``.
    zz
        The Voigt index to condense out.

    Returns
    -------
    tuple
        The six condensed arrays, in the same shapes as given.
    """

    Dzz = D[..., zz, zz]

    Dcond = D - np.einsum("...i,...j->...ij", D[..., :, zz], D[..., zz, :]) / Dzz[..., None, None]
    dLambdaCond = dLambda - D[..., :, zz] * dLambda[..., zz][..., None] / Dzz[..., None]
    dLapCond = dLaplacian - D[..., :, zz] * dLaplacian[..., zz][..., None] / Dzz[..., None]
    dFdEcond = dFdStrain - dFdStrain[..., zz][..., None] * D[..., zz, :] / Dzz[..., None]
    dFdLambdaCond = dFdLambda - dFdStrain[..., zz] * dLambda[..., zz] / Dzz
    dFdLapCond = dFdLaplacian - dFdStrain[..., zz] * dLaplacian[..., zz] / Dzz

    return Dcond, dLambdaCond, dLapCond, dFdEcond, dFdLambdaCond, dFdLapCond


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


def hourglassVector(spacings) -> np.ndarray:
    """The Flanagan-Belytschko orthogonal hourglass shape vector of a 2D grid cell.

    Single point quadrature at the cell centre -- :func:`cellGradientOperator` sampled once,
    rather than :func:`cellCornerGradientOperators` sampled at every corner -- is second order
    accurate and does not lock volumetrically, but the bilinear cell then has one zero energy
    mode left uncontrolled: the alternating corner pattern ``(1, -1, -1, 1)``, the "hourglass"
    mode, which the single sampling point cannot see at all.

    Flanagan and Belytschko's remedy [1]_ projects the constant and linear parts out of that
    pattern,

    .. math::
        \\boldsymbol \\gamma = \\boldsymbol h
            - \\left(\\boldsymbol h \\cdot \\boldsymbol x\\right) \\boldsymbol b_x
            - \\left(\\boldsymbol h \\cdot \\boldsymbol y\\right) \\boldsymbol b_y

    with :math:`\\boldsymbol h = (1, -1, -1, 1)` and :math:`\\boldsymbol b_x, \\boldsymbol b_y`
    the centre gradient's corner weights. The result is, by construction, orthogonal to any
    constant or linear field -- :math:`\\boldsymbol \\gamma \\cdot \\boldsymbol u` is exactly
    zero for such a field and :math:`O(h^2)` for a smooth one with curvature, so a stabilization
    stiffness built from it, :math:`c \\, \\boldsymbol \\gamma \\boldsymbol \\gamma^T`, resists
    the hourglass mode without spoiling the second order accuracy of the underlying operator --
    see ``tests/test_hourglassstabilization.py`` for both properties verified numerically.

    Parameters
    ----------
    spacings
        The grid spacing per direction, shape ``(2,)``. Only two dimensions are supported: a
        trilinear hexahedron has three independent hourglass modes per component, not one, which
        is a materially larger extension than this single mode.

    Returns
    -------
    np.ndarray
        The hourglass vector, shape ``(4,)``, ordered like :func:`cellCornerOffsets`.

    References
    ----------
    .. [1] D.P. Flanagan, T. Belytschko, "A uniform strain hexahedron and quadrilateral with
       orthogonal hourglass control", IJNME 17, 1981, 679-706.
    """

    spacings = np.asarray(spacings, dtype=float)

    if spacings.size != 2:
        raise ValueError("The orthogonal hourglass vector is only implemented in two dimensions.")

    pattern = np.array([1.0, -1.0, -1.0, 1.0])
    corners = cellCornerOffsets(2).astype(float)
    coordinates = corners * spacings

    gradient = cellGradientOperator(spacings)
    bx, by = gradient[0, :], gradient[1, :]

    return pattern - (pattern @ coordinates[:, 0]) * bx - (pattern @ coordinates[:, 1]) * by


def volumetricallyAveragedStrainOperators(strainOperators, weights=None) -> list:
    """Replace the volumetric part of each strain operator by the average over the cell.

    The corner sampled strain operators of a cell are the bilinear ones evaluated at the cell's
    vertices, so they constrain the volumetric strain at every corner separately: one constraint
    per corner against, in the limit, one displacement degree of freedom per grid point. As
    Poisson's ratio approaches one half that over-constrains the cell and it locks -- the cell
    can no longer deform at constant volume, which is exactly what a shear band has to do, so the
    band does not form and the response comes out too stiff instead.

    Averaging the volumetric part over the cell, the classical B-bar or mean dilatation treatment,
    leaves one volumetric constraint per cell and cures it. The deviatoric part stays sampled at
    the corners, so the hourglass mode is still controlled -- see
    :func:`cellCornerGradientOperators` for why that sampling is used in the first place.

    Writing ``m`` for the Voigt trace vector, the volumetric part of an operator is
    ``B_vol = m m^T B / 3`` and

    .. math::
        \\bar{\\boldsymbol B}_p = \\boldsymbol B_p - \\boldsymbol B_{vol,p}
                                 + \\sum_q w_q \\boldsymbol B_{vol,q}

    Note that the trace has to be taken over all three normal components even in two dimensions:
    under plane strain ``eps_33`` vanishes and the third row of the operator is zero, so it drops
    out of the sum by itself, and the volumetric strain is still ``eps_11 + eps_22``.

    Parameters
    ----------
    strainOperators
        The strain operators of the cell's material points, each of shape
        ``(6, nCorners * nDim)``.
    weights
        The weights to average with, one per material point. Defaults to equal weights, which is
        what equal material point volumes call for.

    Returns
    -------
    list
        The averaged strain operators, same shapes as the input.
    """

    strainOperators = [np.asarray(B, dtype=float) for B in strainOperators]

    if weights is None:
        weights = np.full(len(strainOperators), 1.0 / len(strainOperators))
    else:
        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum()

    # the Voigt trace vector: the first three components are the normal ones
    trace = np.zeros(nVoigtComponents)
    trace[:3] = 1.0

    def volumetricPart(B):
        return np.outer(trace, trace @ B) / 3.0

    averaged = sum(weight * volumetricPart(B) for weight, B in zip(weights, strainOperators))

    return [B - volumetricPart(B) + averaged for B in strainOperators]
