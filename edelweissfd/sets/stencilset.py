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
An ordered set of stencils.

It plays the role :class:`~edelweissfe.sets.elementset.ElementSet` plays for finite
elements. A separate class is needed only because ``ElementSet`` validates its members
against the finite element base classes, which a stencil deliberately is not.
"""

from edelweissfe.sets.orderedset import ImmutableOrderedSet

from edelweissfd.stencils.base.basestencil import BaseStencil


class StencilSet(ImmutableOrderedSet):
    """A set of stencils.

    Parameters
    ----------
    label
        The unique label for this stencil set.
    stencils
        The stencils.
    """

    def __init__(self, label: str, stencils):
        self.allowedObjectTypes = [BaseStencil]

        super().__init__(label, stencils)

        self.stencils = self.items

    @property
    def elements(self):
        """The stencils. An alias making the set interchangeable with an ElementSet in the
        field output machinery."""

        return self.items
