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
A Generalized Finite Difference (GFDM) nodal differentiation operator.

The operators here (:func:`gatherCloud`, :func:`gfdmWeights`) work in any number of
dimensions; the stencil built from them,
:class:`~edelweissfd.stencils.gfdmgradientplasticitystencil.GFDMGradientPlasticityStencil`,
currently only wires up one and two.

The corner-sampled strain operator used by
:class:`~edelweissfd.stencils.gradientplasticitystencil.GradientPlasticityStencil`
is only first order accurate at the corner itself, and the price of second order accuracy
there without hourglassing would be a fundamentally different (staggered) grid topology. GFDM
offers a third way: at every grid node directly, fit a local quadratic Taylor polynomial through
a cloud of neighbours by *weighted least squares* rather than reading a derivative off two or
three points exactly. Consistency (correct derivatives in the limit) does not depend on the
cloud being small, symmetric, or exactly matching a monomial count, which is what removes the
corner operator's need to choose between accuracy and hourglass control in the first place: a
single, uniform stencil at every node is both second order accurate (interior, verified down to
machine precision on a cubic test field) and free of the checkerboard zero energy mode (verified
by direct eigenvalue inspection of a small elastic patch under the panel's own boundary
conditions) -- see the session notes for both checks.

Two things a naive implementation gets wrong, both encountered and fixed while prototyping this:

* Weights must be applied to *differences* from the centre's own value, not to the neighbours'
  absolute values -- the constant term of the local Taylor model is the centre value itself, and
  skipping the subtraction silently reintroduces an error that looks like a consistency failure
  but is really just a wrong centre-of-expansion.
* The neighbour cloud can be **rank deficient** for a chosen polynomial degree on a regular
  lattice, invisibly -- a well counted but badly shaped cloud solves to a numerically-unstable,
  wildly wrong answer rather than raising an error. :func:`gfdmWeights` checks the rank
  explicitly and :func:`gatherCloud` widens the search until it is satisfied.
"""

import numpy as np

#: [ux, uy, uxx, uxy, uyy], in this order, is what :func:`gfdmWeights` returns per neighbour.
nDerivatives = 5


def gatherCloud(grid, node, minPoints: int = 8, maxRadius: int = 6) -> list:
    """The neighbours of a grid point, widening a compass search until enough are available.

    Parameters
    ----------
    grid
        The :class:`~edelweissfd.grids.structuredgrid.StructuredGrid` the node belongs to.
    node
        The grid point to gather neighbours around.
    minPoints
        The minimum cloud size to stop widening at. Five is the bare minimum a two dimensional
        quadratic model needs; eight (the full interior compass ring) is the default so that the
        least squares fit is over-determined rather than exact, which is what makes it robust to
        a single badly placed neighbour.
    maxRadius
        The largest compass radius to try before giving up.

    Returns
    -------
    list[Node]
        The neighbouring grid points, excluding the centre itself.

    Raises
    ------
    ValueError
        If ``maxRadius`` is reached without finding ``minPoints`` neighbours -- only possible on
        a grid with fewer than ``minPoints`` points in total.
    """

    centreIndex = np.asarray(grid.gridIndexOf(node))
    shape = grid.shape

    for radius in range(1, maxRadius + 1):
        neighbours = []
        for offset in np.ndindex(*([2 * radius + 1] * grid.nDim)):
            offset = np.asarray(offset) - radius
            if not np.any(offset):
                continue
            if np.max(np.abs(offset)) > radius:
                continue
            index = centreIndex + offset
            if np.all((index >= 0) & (index < shape)):
                neighbours.append(grid._nodeGrid[tuple(index)])
        if len(neighbours) >= minPoints:
            return neighbours

    raise ValueError(
        "Could not find {:} neighbours of node {:} within a compass radius of {:}.".format(
            minPoints, node.label, maxRadius
        )
    )


def _taylorBasisIndices(nDim: int):
    """The multi-indices of a degree-2 Taylor basis in ``nDim`` dimensions, excluding the
    constant term: linear terms, pure second-derivative terms, and cross terms, in that order.
    All three groups are needed for a *correct* fit even though only the first two are read back
    out -- leaving the cross terms out biases the pure second-derivative (Laplacian) estimate."""

    linear = [(i,) for i in range(nDim)]
    pureSquare = [(i, i) for i in range(nDim)]
    cross = [(i, j) for i in range(nDim) for j in range(i + 1, nDim)]
    return linear, pureSquare, cross


def gfdmWeights(centre, neighbours, weightPower: int = 2, conditionLimit: float = 1e8) -> tuple:
    """The weighted least squares gradient and Laplacian weights over a neighbour cloud.

    The returned weights act on the neighbours' *values minus the centre's own value* -- the
    coefficient the centre itself would carry, so that a constant field gives an exactly
    vanishing derivative, is ``-weights.sum(axis=-1)`` and is added explicitly by the caller
    once it decides how to lay the centre out among its degrees of freedom.

    Works in any number of dimensions: one, as needed for a 1D bar, two, or three.

    Parameters
    ----------
    centre
        The coordinates of the point the derivatives are taken at, shape ``(nDim,)``.
    neighbours
        The coordinates of the cloud, shape ``(n, nDim)``.
    weightPower
        The distance weighting exponent; larger values favour closer neighbours more strongly.
    conditionLimit
        Raise if the design matrix's condition number exceeds this -- the signature of a rank
        deficient or near degenerate cloud, which a silent solve would turn into a wildly wrong,
        not merely inaccurate, answer.

    Returns
    -------
    tuple
        ``(gradientWeights, laplacianWeights)``: shapes ``(nDim, n)`` and ``(n,)``.
    """

    centre = np.atleast_1d(np.asarray(centre, dtype=float))
    neighbours = np.atleast_2d(np.asarray(neighbours, dtype=float))
    nDim = centre.size

    d = neighbours - centre[None, :]

    linear, pureSquare, cross = _taylorBasisIndices(nDim)
    columns = [d[:, i] for (i,) in linear]
    columns += [0.5 * d[:, i] ** 2 for (i, _) in pureSquare]
    columns += [d[:, i] * d[:, j] for (i, j) in cross]
    A = np.stack(columns, axis=1)

    distance = np.linalg.norm(d, axis=1)
    W = np.diag(1.0 / distance**weightPower)

    AtWA = A.T @ W @ A

    if np.linalg.cond(AtWA) > conditionLimit:
        raise ValueError(
            "The GFDM neighbour cloud is too close to rank deficient (condition number {:.3e}); "
            "widen the cloud (gatherCloud's minPoints) or check for a degenerate lattice pattern.".format(
                np.linalg.cond(AtWA)
            )
        )

    pseudo = np.linalg.solve(AtWA, A.T @ W)  # (nTerms, n)

    gradientWeights = pseudo[:nDim, :]
    laplacianWeights = pseudo[nDim : 2 * nDim, :].sum(axis=0)

    return gradientWeights, laplacianWeights
