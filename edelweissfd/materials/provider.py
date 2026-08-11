#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  ---------------------------------------------------------------------
#
#  _____    _      _              _         _____ ____
# | ____|__| | ___| |_      _____(_)___ ___|  ___|  _ \
# |  _| / _` |/ _ \ \ \ /\ / / _ \ / __/ __| |_  | | | |
# | |__| (_| |  __/ |\ V  V /  __/ \__ \__ \  _| | |_| |
# |_____\__,_|\___|_| \_/\_/ \___|_|___/___/_|   |____/
#
#
#  Unit of Strength of Materials and Structural Analysis
#  University of Innsbruck,
#  2024 - today
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
"""A description of a material, which mints one instance per stencil.

Why a material cannot simply be shared
--------------------------------------

A point-wise material is not a pure function of its arguments. It carries the storage it operates
on as mutable state: ``assignCurrentStateVars`` points it at the state variables of one particular
material point, and the next evaluation reads and writes exactly those. A single instance can
therefore only serve one material point at a time.

That is fine within a stencil, which evaluates its material points one after another, and it is
*not* fine across stencils, because the solvers hand chunks of entities to different threads (see
``edelweissfe.solvers.base.parallelelementcomputation``). Two threads re-pointing the same material
at their own state variables corrupt it. On a free threaded interpreter the failure is immediate
and total -- the Cython memoryview holding the state variables has its acquisition count
decremented twice and aborts the process with ``Fatal Python error: Acquisition count is -1`` --
and with the global interpreter lock in place the same race merely becomes unlikely and silent,
which is worse: a material point can be evaluated against another one's state variables.

Handing out a fresh instance per stencil removes the sharing rather than guarding it, so the
invariant holds no matter how the solver schedules its threads.
"""


import numpy as np


class MaterialProvider:
    """A material class together with the arguments to construct it.

    Parameters
    ----------
    materialClass
        The material class to instantiate.
    arguments
        The positional arguments its constructor takes.
    """

    def __init__(self, materialClass: type, arguments: tuple):
        self.materialClass = materialClass
        self.arguments = tuple(arguments)

    def createMaterial(self):
        """Create a new material instance, owned by a single stencil.

        Array arguments are copied, so that no two instances end up holding a view of the same
        buffer. The Cython materials keep the property array as a memoryview attribute, and the
        point of this class is that instances share nothing at all.

        Returns
        -------
        The material instance.
        """

        arguments = tuple(
            argument.copy() if isinstance(argument, np.ndarray) else argument for argument in self.arguments
        )

        return self.materialClass(*arguments)

    def __repr__(self):
        return "MaterialProvider({:})".format(self.materialClass.__name__)


def materialInstanceFrom(materialOrProvider):
    """Resolve either a provider or a ready made material into a material instance.

    A :class:`MaterialProvider` yields a fresh instance on every call, so each caller gets one of
    its own. Anything else is assumed to be a material already and is passed through -- which is
    what the tangent tests do, where a single stencil is built and evaluated by hand.

    Parameters
    ----------
    materialOrProvider
        A :class:`MaterialProvider` or a material instance.

    Returns
    -------
    The material instance.
    """

    if isinstance(materialOrProvider, MaterialProvider):
        return materialOrProvider.createMaterial()

    return materialOrProvider
