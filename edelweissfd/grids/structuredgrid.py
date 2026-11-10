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
A uniform, structured, collocated grid in one, two or three dimensions.

The grid points are plain :class:`~edelweissfe.points.node.Node` instances, so all
machinery of EdelweissFE which operates on nodes and node sets, in particular the
:class:`~edelweissfe.numerics.dofmanager.DofManager` and every step action, applies
unchanged. All fields live on all grid points, i.e. the grid is collocated.

The node sets are named after the ones the ``planeRectQuad`` and ``boxGen`` generators of
EdelweissFE create, so boundary conditions read the same in both packages:

``all``, ``interior``, ``boundary``, ``left`` / ``right`` (:math:`x`),
``bottom`` / ``top`` (:math:`y`), ``back`` / ``front`` (:math:`z`), and all pairwise
intersections of the boundary planes, e.g. ``leftBottom`` and ``bottomLeft``.
"""

from itertools import product

import numpy as np
from edelweissfe.points.node import Node
from edelweissfe.sets.nodeset import NodeSet

#: The node set name of the lower and the upper boundary per direction.
#:
#: The naming follows the ``planeRectQuad`` and ``boxGen`` generators of EdelweissFE, which
#: means that ``front`` is at the *upper* end of the third direction and ``back`` at its
#: lower end, opposite to the first two directions.
boundarySetNames = (("left", "right"), ("bottom", "top"), ("back", "front"))


def _concatenate(firstName: str, secondName: str) -> str:
    """Join two boundary names in camel case, e.g. ``left`` and ``bottom`` to ``leftBottom``.

    Parameters
    ----------
    firstName
        The leading name.
    secondName
        The trailing name, whose first letter is capitalized.

    Returns
    -------
    str
        The joined name.
    """

    return firstName + secondName[0].upper() + secondName[1:]


class StructuredGrid:
    """A uniform structured grid of collocated grid points.

    Parameters
    ----------
    name
        The name of the grid, used to prefix the node sets registered in the model.
    lengths
        The extent of the grid per direction, shape ``(nDim,)``.
    nGridPoints
        The number of grid points per direction, shape ``(nDim,)``. Each entry must be at
        least two.
    origin
        The coordinates of the lower left grid point, shape ``(nDim,)``. Defaults to the
        origin.
    firstNodeLabel
        The label of the first grid point. Labels are assigned consecutively.
    """

    def __init__(
        self,
        name: str,
        lengths,
        nGridPoints,
        origin=None,
        firstNodeLabel: int = 1,
    ):
        self.name = name

        self.lengths = np.asarray(lengths, dtype=float)
        self.shape = tuple(int(n) for n in nGridPoints)
        self.nDim = len(self.shape)

        if self.nDim not in (1, 2, 3):
            raise ValueError("A structured grid must be one, two or three dimensional.")

        if self.lengths.size != self.nDim:
            raise ValueError("lengths and nGridPoints must have the same number of entries.")

        if min(self.shape) < 2:
            raise ValueError("At least two grid points per direction are required.")

        if np.any(self.lengths <= 0.0):
            raise ValueError("All lengths must be positive.")

        self.origin = np.zeros(self.nDim) if origin is None else np.asarray(origin, dtype=float)

        self._firstNodeLabel = firstNodeLabel

        #: The grid spacing per direction.
        self.spacings = self.lengths / (np.array(self.shape) - 1)

        #: The number of cells per direction.
        self.cellShape = tuple(n - 1 for n in self.shape)

        self._createNodes(firstNodeLabel)
        self._createNodeSets()

    def _createNodes(self, firstNodeLabel: int):
        """Create the grid points."""

        self._nodeGrid = np.empty(self.shape, dtype=object)

        #: The grid points, keyed by their label.
        self.nodes = dict()

        label = firstNodeLabel

        for index in np.ndindex(*self.shape):
            coordinates = self.origin + np.array(index, dtype=float) * self.spacings

            node = Node(label, coordinates)

            self._nodeGrid[index] = node
            self.nodes[label] = node

            label += 1

    def _createNodeSets(self):
        """Create the node sets of the grid."""

        #: The node sets of the grid, keyed by their unprefixed name.
        self.nodeSets = dict()

        allNodes = list(self.nodes.values())

        self.nodeSets["all"] = NodeSet(self._qualify("all"), allNodes)

        lowerBoundaries = dict()
        upperBoundaries = dict()

        for d in range(self.nDim):
            lowerName, upperName = boundarySetNames[d]

            lowerBoundaries[lowerName] = self._nodesOnBoundary(d, 0)
            upperBoundaries[upperName] = self._nodesOnBoundary(d, self.shape[d] - 1)

            self.nodeSets[lowerName] = NodeSet(self._qualify(lowerName), lowerBoundaries[lowerName])
            self.nodeSets[upperName] = NodeSet(self._qualify(upperName), upperBoundaries[upperName])

        boundaryNodes = [n for n in allNodes if self.isOnBoundary(n)]

        self.nodeSets["boundary"] = NodeSet(self._qualify("boundary"), boundaryNodes)
        self.nodeSets["interior"] = NodeSet(
            self._qualify("interior"), [n for n in allNodes if not self.isOnBoundary(n)]
        )

        self._createIntersectionNodeSets()

    def _createIntersectionNodeSets(self):
        """Create the intersections of the boundary planes, as the EdelweissFE generators do.

        ``planeRectQuad`` names them ``leftBottom``, ``rightTop`` and so on, while ``boxGen``
        puts the second direction first, i.e. ``bottomLeft`` and ``topFront``. Both spellings
        are registered, so boundary conditions transfer either way.
        """

        if self.nDim < 2:
            return

        firstNames = boundarySetNames[0]
        secondNames = boundarySetNames[1]
        thirdNames = boundarySetNames[2] if self.nDim >= 3 else ()

        pairs = list(product(firstNames, secondNames))

        if thirdNames:
            pairs += list(product(secondNames, thirdNames)) + list(product(firstNames, thirdNames))

        for firstName, secondName in pairs:
            firstNodes = set(self.nodeSets[firstName])
            intersection = [node for node in self.nodeSets[secondName] if node in firstNodes]

            for name in (_concatenate(firstName, secondName), _concatenate(secondName, firstName)):
                if name not in self.nodeSets:
                    self.nodeSets[name] = NodeSet(self._qualify(name), intersection)

    def _qualify(self, setName: str) -> str:
        """The name a node set is registered under in the model.

        Parameters
        ----------
        setName
            The unprefixed name of the node set.

        Returns
        -------
        str
            The prefixed name.
        """

        return "{:}_{:}".format(self.name, setName)

    def _nodesOnBoundary(self, direction: int, index: int) -> list[Node]:
        """The grid points on a boundary plane.

        Parameters
        ----------
        direction
            The direction normal to the boundary.
        index
            The grid index along that direction.

        Returns
        -------
        list[Node]
            The grid points on that plane.
        """

        selection = [slice(None)] * self.nDim
        # a slice rather than an int, so the result stays an array also in 1D
        selection[direction] = slice(index, index + 1)

        return list(self._nodeGrid[tuple(selection)].flatten())

    def nodeAt(self, *index) -> Node:
        """The grid point at the given grid index.

        Parameters
        ----------
        index
            The grid index, one entry per direction.

        Returns
        -------
        Node
            The grid point.
        """

        return self._nodeGrid[tuple(index)]

    def gridIndexOf(self, node: Node) -> tuple:
        """The grid index of a grid point.

        Parameters
        ----------
        node
            The grid point.

        Returns
        -------
        tuple
            The grid index.
        """

        return np.unravel_index(node.label - self._firstNodeLabel, self.shape)

    def isOnBoundary(self, node: Node) -> bool:
        """Whether a grid point lies on the boundary of the grid.

        Parameters
        ----------
        node
            The grid point.

        Returns
        -------
        bool
            True if the grid point lies on any boundary plane.
        """

        index = self.gridIndexOf(node)

        return any(i == 0 or i == self.shape[d] - 1 for d, i in enumerate(index))

    def cellIndices(self):
        """Iterate over the grid indices of the lower left corner of every cell.

        Yields
        ------
        tuple
            The grid index of a cell's lower left corner.
        """

        return np.ndindex(*self.cellShape)

    def cellCornerNodes(self, cellIndex, cornerOffsets: np.ndarray) -> list[Node]:
        """The corner grid points of a cell.

        Parameters
        ----------
        cellIndex
            The grid index of the cell's lower left corner.
        cornerOffsets
            The corner offsets, shape ``(nCorners, nDim)``, as returned by
            :func:`~edelweissfd.operators.differences.cellCornerOffsets`.

        Returns
        -------
        list[Node]
            The corner grid points, in the order of ``cornerOffsets``.
        """

        base = np.asarray(cellIndex, dtype=int)

        return [self._nodeGrid[tuple(base + offset)] for offset in cornerOffsets]

    @property
    def cellVolume(self) -> float:
        """The volume of a single grid cell."""

        return float(np.prod(self.spacings))

    def boundaryNodeAreas(self, direction: int, index: int) -> dict:
        """The share of the boundary surface belonging to each grid point of a boundary plane.

        The shares are the tributary areas of a vertex centred discretization, i.e. a grid
        point in the interior of the plane owns a full cell face, one on an edge of the
        plane owns one half of it and one in a corner one quarter. They sum up to the total
        area of the boundary plane, which makes them the consistent factors for turning a
        traction into equivalent nodal forces.

        For a one dimensional grid the boundary plane is a point and the area is one, i.e.
        a traction is then a force per unit cross section.

        Parameters
        ----------
        direction
            The direction normal to the boundary plane.
        index
            The grid index along that direction.

        Returns
        -------
        dict
            The tributary area, keyed by grid point.
        """

        inPlaneDirections = [d for d in range(self.nDim) if d != direction]

        areas = dict()

        for node in self._nodesOnBoundary(direction, index):
            nodeIndex = self.gridIndexOf(node)

            area = 1.0
            for d in inPlaneDirections:
                # a grid point at the rim of the plane owns only half of the spacing
                isAtRim = nodeIndex[d] == 0 or nodeIndex[d] == self.shape[d] - 1
                area *= self.spacings[d] * (0.5 if isAtRim else 1.0)

            areas[node] = area

        return areas

    def __repr__(self) -> str:
        return "StructuredGrid('{:}', shape={:}, spacings={:})".format(
            self.name, self.shape, np.array2string(self.spacings, precision=4)
        )
