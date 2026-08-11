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
A numerical tangent for stencils.

Analytic tangents are what the stencils use in production, since they are exact and cheap.
A wrong analytic tangent, however, does not announce itself: the residual still converges,
just slowly or not at all, and the cause is hard to see from a Newton log. This mixin
therefore provides a second, independent opinion by differencing the internal flux with
respect to the stencil's own degrees of freedom.

It is used by the test suite and can be switched on temporarily while developing a new
stencil. It is not meant for production runs, as it costs ``2 * nDof`` material
evaluations per stencil.
"""

import numpy as np


class NumericalTangentMixin:
    """Adds a numerically differenced tangent to a stencil.

    The stencil must implement ``computeKernels``, ``acceptLastState`` and
    ``resetToLastValidState``, and its ``computeKernels`` must not depend on state left
    over from a previous call, i.e. it must restore its temporary state variables from the
    converged ones on entry. All stencils of EdelweissFD do so.
    """

    #: The relative perturbation used if no absolute one is given.
    defaultRelativePerturbation = 1e-7

    def computeNumericalTangent(
        self,
        U: np.ndarray,
        dU: np.ndarray,
        time: float,
        dT: float,
        perturbation: float = None,
    ) -> np.ndarray:
        """The tangent obtained by central differences of the internal flux.

        Parameters
        ----------
        U
            The current solution vector of this stencil's degrees of freedom.
        dU
            The current increment of this stencil's degrees of freedom.
        time
            The total time at the end of the increment.
        dT
            The time increment.
        perturbation
            The absolute perturbation of the degrees of freedom. Defaults to a scaled
            relative perturbation.

        Returns
        -------
        np.ndarray
            The numerical tangent, shape ``(nDof, nDof)``.
        """

        nDof = self.nDof

        U = np.asarray(U, dtype=float)
        dU = np.asarray(dU, dtype=float)

        if perturbation is None:
            scale = max(1.0, float(np.max(np.abs(U))))
            perturbation = self.defaultRelativePerturbation * scale

        tangent = np.zeros((nDof, nDof))

        KDummy = np.zeros((nDof, nDof))

        for j in range(nDof):
            fluxes = []

            for sign in (+1.0, -1.0):
                UPerturbed = U.copy()
                dUPerturbed = dU.copy()

                UPerturbed[j] += sign * perturbation
                dUPerturbed[j] += sign * perturbation

                PPerturbed = np.zeros(nDof)
                KDummy[:, :] = 0.0

                self.computeKernels(KDummy, PPerturbed, UPerturbed, dUPerturbed, time, dT)

                fluxes.append(PPerturbed)

            tangent[:, j] = (fluxes[0] - fluxes[1]) / (2.0 * perturbation)

        # leave the stencil in the unperturbed state
        PUnperturbed = np.zeros(nDof)
        KDummy[:, :] = 0.0
        self.computeKernels(KDummy, PUnperturbed, U, dU, time, dT)

        return tangent

    def checkTangent(
        self,
        U: np.ndarray,
        dU: np.ndarray,
        time: float,
        dT: float,
        perturbation: float = None,
    ) -> tuple:
        """Compare the analytic tangent against the numerical one.

        Parameters
        ----------
        U
            The current solution vector of this stencil's degrees of freedom.
        dU
            The current increment of this stencil's degrees of freedom.
        time
            The total time at the end of the increment.
        dT
            The time increment.
        perturbation
            The absolute perturbation of the degrees of freedom.

        Returns
        -------
        tuple
            The analytic tangent, the numerical tangent, and the deviation relative to the
            largest entry of the analytic tangent.
        """

        nDof = self.nDof

        analyticTangent = np.zeros((nDof, nDof))
        P = np.zeros(nDof)

        self.computeKernels(analyticTangent, P, U, dU, time, dT)

        numericalTangent = self.computeNumericalTangent(U, dU, time, dT, perturbation)

        scale = np.max(np.abs(analyticTangent))
        if scale == 0.0:
            scale = 1.0

        deviation = float(np.max(np.abs(analyticTangent - numericalTangent)) / scale)

        return analyticTangent, numericalTangent, deviation

    def assertTangentConsistent(
        self,
        U: np.ndarray,
        dU: np.ndarray,
        time: float,
        dT: float,
        perturbation: float = None,
        relativeTolerance: float = 1e-5,
    ):
        """Raise if the analytic tangent disagrees with the numerical one.

        Parameters
        ----------
        U
            The current solution vector of this stencil's degrees of freedom.
        dU
            The current increment of this stencil's degrees of freedom.
        time
            The total time at the end of the increment.
        dT
            The time increment.
        perturbation
            The absolute perturbation of the degrees of freedom.
        relativeTolerance
            The tolerated deviation, relative to the largest entry of the analytic tangent.
        """

        analyticTangent, numericalTangent, deviation = self.checkTangent(U, dU, time, dT, perturbation)

        if not deviation < relativeTolerance:
            raise AssertionError(
                "The analytic tangent of {:} deviates from the numerical one by {:.3e} "
                "(tolerated {:.3e}).\nanalytic:\n{:}\nnumerical:\n{:}".format(
                    type(self).__name__,
                    deviation,
                    relativeTolerance,
                    np.array2string(analyticTangent, precision=4, suppress_small=True),
                    np.array2string(numericalTangent, precision=4, suppress_small=True),
                )
            )
