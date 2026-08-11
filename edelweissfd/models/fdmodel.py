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
The model tree of a finite difference simulation.

:class:`FDModel` is an :class:`~edelweissfe.models.femodel.FEModel` which additionally
holds the grids and which initializes stencils instead of relying on sections. Everything
else, in particular the creation of the field variables and node fields, the bundling into
node fields and the acceptance of converged states, is inherited unchanged, so that the
solvers, the DofManager and the output managers of EdelweissFE see a familiar model.

The stencils are stored in ``elements``, since that is the attribute the solvers iterate
over. ``stencils`` is an alias for the very same dictionary and is the name to prefer in
user code.
"""

from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.sets.nodeset import NodeSet


class FDModel(FEModel):
    """A finite difference model tree.

    Parameters
    ----------
    dimension
        The spatial dimension of the model.
    """

    identification = "FDModel"

    def __init__(self, dimension: int):
        super().__init__(dimension)

        #: The structured grids in the model, keyed by their name.
        self.grids = dict()

    @property
    def stencils(self) -> dict:
        """The stencils in the model, keyed by their number. An alias for ``elements``."""

        return self.elements

    def addGrid(self, grid):
        """Register a grid, its grid points and its node sets in the model.

        Parameters
        ----------
        grid
            The :class:`~edelweissfd.grids.structuredgrid.StructuredGrid` to be registered.
        """

        if grid.name in self.grids:
            raise ValueError("A grid named '{:}' already exists in this model.".format(grid.name))

        if grid.nDim != self.domainSize:
            raise ValueError(
                "The grid '{:}' is {:}-dimensional, but the model is {:}-dimensional.".format(
                    grid.name, grid.nDim, self.domainSize
                )
            )

        self.grids[grid.name] = grid

        for label, node in grid.nodes.items():
            if label in self.nodes:
                raise ValueError("A node with label {:} already exists in this model.".format(label))

            self.nodes[label] = node

        for nodeSet in grid.nodeSets.values():
            self.nodeSets[nodeSet.name] = nodeSet

    def addStencil(self, stencil):
        """Register a stencil in the model.

        Parameters
        ----------
        stencil
            The stencil to be registered.
        """

        if stencil.stencilNumber in self.elements:
            raise ValueError("A stencil with number {:} already exists in this model.".format(stencil.stencilNumber))

        self.elements[stencil.stencilNumber] = stencil

    def _prepareStencils(self, journal: Journal):
        """Let all stencils create their state variables and initialize their materials.

        Parameters
        ----------
        journal
            The journal instance.
        """

        journal.message("Initializing {:} stencils".format(len(self.elements)), self.identification)

        for stencil in self.elements.values():
            stencil.initializeStencil()

    def prepareYourself(self, journal: Journal):
        """Prepare the model for a simulation, i.e. initialize the stencils, create the
        variables and bundle the fields.

        Parameters
        ----------
        journal
            The journal instance.
        """

        if not self.nodeSets:
            raise ValueError("The model holds no grid; nothing to prepare.")

        # The node set 'all' is what the field variable linking and the default field
        # outputs operate on, so it has to span every grid.
        self.nodeSets["all"] = self._createCombinedAllNodeSet()

        self._prepareStencils(journal)

        self._prepareVariablesAndFields(journal)

    def _createCombinedAllNodeSet(self):
        """The node set spanning all nodes of all grids in the model."""

        return NodeSet("all", list(self.nodes.values()))
