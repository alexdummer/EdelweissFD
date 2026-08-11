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
The abstract base class of all finite difference stencils.

A stencil is a :class:`~edelweissfe.nodecouplingentity.base.nodecouplingentity.BaseNodeCouplingEntity`,
which is the abstraction EdelweissFE offers for anything that couples nodes. From the
point of view of the solvers a stencil is therefore indistinguishable from a finite
element: the :class:`~edelweissfe.numerics.dofmanager.DofManager` derives the global
degrees of freedom and the sparsity pattern from the stencil's nodes and fields, and the
solvers evaluate it through :meth:`~BaseStencil.computeKernels`.

Implementing a stencil means implementing :meth:`~BaseStencil.computeKernels`, plus the
handful of descriptive properties below.

Conventions inherited from EdelweissFE, all of which matter:

* The degrees of freedom of the stencil are ordered node-major and field-minor, i.e. all
  fields of node 0, then all fields of node 1, and so on.
* ``P`` is the internal flux in *positive* sense; the solvers form the residual as
  :math:`R = -P + P_{ext}`. Consequently ``K`` is :math:`\\partial P / \\partial U`.
* ``K`` arrives as a two dimensional, column-major view of the stencil's block in the
  global sparse matrix, so writing ``K[i, j]`` addresses row ``i`` and column ``j`` of the
  local block. It is *not* zeroed between iterations by the stencil; the solver zeroes the
  whole value array instead.
"""

from abc import abstractmethod

import numpy as np
from edelweissfe.nodecouplingentity.base.nodecouplingentity import (
    BaseNodeCouplingEntity,
)
from edelweissfe.numerics.vijentitybase import VIJEntityBase
from edelweissfe.points.node import Node


class BaseStencil(BaseNodeCouplingEntity, VIJEntityBase):
    """The interface every finite difference stencil has to fulfill."""

    @property
    @abstractmethod
    def stencilNumber(self) -> int:
        """The unique number of this stencil."""

    @property
    @abstractmethod
    def hasMaterial(self) -> bool:
        """Whether a material has been assigned to this stencil."""

    @abstractmethod
    def setMaterial(self, material):
        """Assign the material this stencil evaluates.

        Parameters
        ----------
        material
            The material instance.
        """

    def setCell(self, grid, cellIndex):
        """Attach this stencil to a cell of a grid.

        The default implementation couples exactly the corner grid points of the cell, which
        is all a stencil built from first derivatives needs. A stencil which also needs second
        derivatives -- a Laplacian, say -- overrides this to widen its molecule beyond the
        cell, asking the grid for the additional grid points and coefficients.

        Parameters
        ----------
        grid
            The :class:`~edelweissfd.grids.structuredgrid.StructuredGrid` the cell belongs to.
        cellIndex
            The grid index of the cell's lower left corner.
        """

        self.setNodes(grid.cellCornerNodes(cellIndex, self.cornerOffsets))

    @property
    @abstractmethod
    def cornerOffsets(self) -> np.ndarray:
        """The corner offsets of the cell this stencil sits on."""

    @abstractmethod
    def initializeStencil(self):
        """Prepare the stencil for computing, once its nodes and material are known."""

    @abstractmethod
    def computeKernels(
        self,
        K: np.ndarray,
        P: np.ndarray,
        U: np.ndarray,
        dU: np.ndarray,
        time: float,
        dT: float,
    ):
        """Evaluate the internal flux and the tangent of this stencil.

        Parameters
        ----------
        K
            The tangent to be defined, shape ``(nDof, nDof)``.
        P
            The internal flux to be defined, shape ``(nDof,)``.
        U
            The current solution vector of this stencil's degrees of freedom.
        dU
            The current increment of this stencil's degrees of freedom.
        time
            The total time at the end of the increment.
        dT
            The time increment.
        """

    def computeBodyForce(
        self,
        P: np.ndarray,
        K: np.ndarray,
        load: np.ndarray,
        U: np.ndarray,
        time: float,
        dT: float,
    ):
        """Evaluate the external flux due to a body force, i.e. a force per unit volume.

        The signature is the one the solvers of EdelweissFE call, and the contribution is
        added to the *external* flux vector.

        Parameters
        ----------
        P
            The external flux to be augmented, shape ``(nDof,)``.
        K
            The tangent to be augmented; a body force which does not depend on the solution
            contributes nothing.
        load
            The force per unit volume, one entry per spatial direction.
        U
            The current solution vector of this stencil's degrees of freedom.
        time
            The total time at the end of the increment.
        dT
            The time increment.
        """

        raise NotImplementedError("{:} does not support body forces.".format(type(self).__name__))

    @abstractmethod
    def acceptLastState(self):
        """Accept the state computed in the last converged increment."""

    @abstractmethod
    def resetToLastValidState(self):
        """Discard the current state and fall back to the last converged one."""

    @abstractmethod
    def getResultArray(self, result: str, quadraturePoint: int = 0, getPersistentView: bool = True) -> np.ndarray:
        """Get a result of this stencil.

        The ``quadraturePoint`` argument exists only because
        :class:`~edelweissfe.utils.elementresultcollector.ElementResultCollector` passes it;
        a stencil owns exactly one material point, so the only valid value is zero.

        Parameters
        ----------
        result
            The name of the result.
        quadraturePoint
            Must be zero.
        getPersistentView
            Whether a view or a copy should be returned.

        Returns
        -------
        np.ndarray
            The result.
        """

    def getNumberOfQuadraturePoints(self) -> int:
        """A stencil owns exactly one material point."""

        return 1

    @abstractmethod
    def getCoordinatesAtCenter(self) -> np.ndarray:
        """The coordinates of the point this stencil is centred at."""

    # -- descriptive properties with a sensible default for cell based stencils ---------

    @property
    def ensightType(self) -> str:
        """The shape of the stencil in Ensight Gold notation. Since a stencil evaluates its
        material at a single point, it is visualized as that point."""

        return "point"

    @property
    def visualizationNodes(self) -> list[Node]:
        """The nodes used for visualization."""

        return self.nodes

    @property
    def dofIndicesPermutation(self) -> np.ndarray:
        """Stencils assemble node-major and field-minor already, so no permutation is
        needed."""

        return None
