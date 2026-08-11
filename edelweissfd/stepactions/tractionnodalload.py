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
A Neumann boundary condition for a structured grid.

In the finite difference scheme of EdelweissFD a prescribed traction enters the momentum
balance as equivalent nodal forces, obtained by multiplying the traction with the tributary
area of each grid point of the boundary plane. Since these areas differ between the
interior of a boundary plane and its rim, the loads are node-wise, which
:mod:`edelweissfe.stepactions.nodeforces` cannot express -- it applies the same load to
every node of a set.

The action is a plain
:class:`~edelweissfe.stepactions.base.nodalloadbase.NodalLoadBase`, so the solvers of
EdelweissFE consume it exactly like nodal forces. It is registered under the
``nodeforces`` module key of the step manager, which is the key the solvers read.
"""

import numpy as np
import sympy as sp
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.stepactions.base.nodalloadbase import NodalLoadBase
from edelweissfe.timesteppers.timestep import TimeStep

#: The step manager module key the solvers read nodal loads from.
stepActionModule = "nodeforces"


class TractionNodalLoad(NodalLoadBase):
    """A traction on a boundary plane of a structured grid, converted to consistent nodal
    forces.

    The load is ramped over the step progress, i.e. it grows linearly from zero at the
    beginning of the step to the full value at its end, unless an amplitude is given.

    Parameters
    ----------
    name
        The name of the step action.
    field
        The field the load acts on.
    nodeSet
        The node set of the boundary plane. Its order defines the order of the load array.
    nodalLoads
        The nodal loads, shape ``(len(nodeSet), fieldSize)``.
    amplitudeExpression
        An optional expression in the step progress ``t``, evaluated on ``[0...1]``.
    """

    def __init__(
        self,
        name: str,
        field: str,
        nodeSet: NodeSet,
        nodalLoads: np.ndarray,
        amplitudeExpression: str = None,
    ):
        self.name = name

        self._field = field
        self._nSet = nodeSet

        self._nodalLoadsStepStart = np.zeros_like(nodalLoads, dtype=float)
        self._nodalLoadsDelta = np.asarray(nodalLoads, dtype=float)

        if self._nodalLoadsDelta.shape[0] != len(nodeSet):
            raise ValueError(
                "The nodal loads have {:} rows, but the node set holds {:} nodes.".format(
                    self._nodalLoadsDelta.shape[0], len(nodeSet)
                )
            )

        self.amplitude = self._createAmplitude(amplitudeExpression)

        self.active = True

    @staticmethod
    def _createAmplitude(amplitudeExpression: str) -> callable:
        """Build the amplitude function of the step progress.

        Parameters
        ----------
        amplitudeExpression
            The expression in the step progress ``t``, or ``None`` for a linear ramp.

        Returns
        -------
        callable
            The amplitude function.
        """

        if amplitudeExpression is None:
            return lambda stepProgress: stepProgress

        t = sp.symbols("t")

        return sp.lambdify(t, sp.sympify(amplitudeExpression), "numpy")

    @property
    def field(self) -> str:
        return self._field

    @property
    def nodeSet(self) -> NodeSet:
        return self._nSet

    def getCurrentLoad(self, timeStep: TimeStep) -> np.ndarray:
        """The nodal loads at the end of the current increment.

        Parameters
        ----------
        timeStep
            The current time step.

        Returns
        -------
        np.ndarray
            The nodal loads, shape ``(len(nodeSet), fieldSize)``.
        """

        if not self.active:
            return self._nodalLoadsStepStart

        return self._nodalLoadsStepStart + self.amplitude(timeStep.stepProgress) * self._nodalLoadsDelta

    def updateStepAction(self, action, jobInfo, model, fieldOutputController, journal):
        """Continue with the load reached at the end of the previous step.

        The signature is the one the step manager uses; ``action`` may carry the keys
        ``nodalLoads`` and ``f(t)`` to change the load in a subsequent step.
        """

        self.active = True

        if action is not None and action.get("nodalLoads") is not None:
            self._nodalLoadsDelta = np.asarray(action["nodalLoads"], dtype=float)

        if action is not None and action.get("f(t)") is not None:
            self.amplitude = self._createAmplitude(action["f(t)"])

    def applyAtStepEnd(self, model, stepMagnitude=None):
        """Freeze the load reached at the end of the step, so it is held constant in
        subsequent steps unless updated."""

        if stepMagnitude is None:
            self._nodalLoadsStepStart = self._nodalLoadsStepStart + self._nodalLoadsDelta
        else:
            self._nodalLoadsStepStart = self._nodalLoadsStepStart + stepMagnitude * self._nodalLoadsDelta

        self._nodalLoadsDelta = np.zeros_like(self._nodalLoadsDelta)
