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
The multifield finite difference stencil for a gradient-enhanced continuum.

Two fields live on every grid point: the ``displacement`` and one nonlocal scalar,
by default ``nonlocal damage``. The governing equations are the balance of linear momentum
together with one screened Poisson equation per nonlocal variable,

.. math::
    -\\nabla \\cdot \\boldsymbol \\sigma(\\boldsymbol \\varepsilon, \\bar\\kappa) = \\boldsymbol 0
    \\qquad
    \\bar\\kappa - \\nabla \\cdot \\left( c(\\bar\\kappa) \\nabla \\bar\\kappa \\right)
    - \\kappa(\\boldsymbol \\varepsilon, \\bar\\kappa) = 0

as implemented by Marmot's ``MarmotMaterialGeneralGradientEnhancedHypoElastic``. Phase field
models fit this framework as well, with the phase field taking the role of the nonlocal
variable; ``AT2PHASEFIELD`` is such a material.

Discretely, per material point of the cell and with :math:`\\boldsymbol N` averaging the
corner values to the material point and :math:`\\boldsymbol G` differencing them,

.. math::
    \\boldsymbol P_u = \\boldsymbol B^T \\boldsymbol \\sigma \\, V
    \\qquad
    \\boldsymbol P_{\\bar\\kappa} =
        \\left[ \\boldsymbol N^T (\\bar\\kappa - \\kappa)
              + c \\, \\boldsymbol G^T \\boldsymbol G \\bar{\\boldsymbol \\kappa} \\right] V

The gradient term appears in its adjoint form, which is what makes the assembled operator
the compact difference Laplacian and keeps a traction free, flux free boundary natural.
The tangent follows by differentiation and is verified against
:class:`~edelweissfd.stencils.numericaltangent.NumericalTangentMixin`.
"""

import numpy as np
from edelweissfe.materials.base.basegradientenhancedhypoelasticmaterial import (
    GradientEnhancedIncrement,
    GradientEnhancedResponse,
    GradientEnhancedTangents,
)
from edelweissfe.points.node import Node

from edelweissfd.operators.differences import (
    cellAverageOperator,
    cellCornerGradientOperators,
    cellCornerOffsets,
    cellGradientOperator,
    cellStrainOperator,
    nVoigtComponents,
)
from edelweissfd.stencils.base.basestencil import BaseStencil
from edelweissfd.stencils.numericaltangent import NumericalTangentMixin

#: The supported stress states, mapping to the material routine of the gradient enhanced
#: material.
#:
#: .. warning::
#:     ``plane stress`` yields a correct residual, but Marmot's
#:     ``MarmotMaterialGeneralGradientEnhancedHypoElastic::computePlaneStress`` returns the
#:     un-condensed three dimensional tangent, i.e. the tangent is inconsistent with the
#:     condensed response. Newton still converges, but not quadratically, and the numerical
#:     tangent check will flag the deviation. Prefer ``plane strain``.
stressStates = {
    "3d": "computeStress",
    "plane strain": "computeStress",
    "plane stress": "computePlaneStress",
}


class GradientEnhancedDisplacementStencil(BaseStencil, NumericalTangentMixin):
    """A gradient-enhanced momentum balance stencil on a single grid cell.

    Parameters
    ----------
    stencilNumber
        A unique number for this stencil.
    spacings
        The grid spacing per direction, shape ``(nDim,)``.
    stressState
        One of ``3d``, ``plane strain`` or ``plane stress``.
    nonlocalField
        The name of the nonlocal field, e.g. ``nonlocal damage``.
    thickness
        An out-of-plane thickness, only meaningful for the two dimensional stress states.
    """

    def __init__(
        self,
        stencilNumber: int,
        spacings,
        stressState: str = "plane strain",
        nonlocalField: str = "nonlocal damage",
        thickness: float = 1.0,
    ):
        if stressState not in stressStates:
            raise ValueError(
                "Unknown stress state '{:}'; available are {:}".format(stressState, ", ".join(stressStates))
            )

        self._stencilNumber = stencilNumber
        self._stressState = stressState
        self._nonlocalField = nonlocalField

        self._spacings = np.asarray(spacings, dtype=float)
        self._nDim = self._spacings.size

        if self._nDim == 1 and stressState in ("plane strain", "plane stress"):
            raise ValueError("A one dimensional grid needs the stress state '3d'.")

        self._cornerOffsets = cellCornerOffsets(self._nDim)
        self._nNodes = self._cornerOffsets.shape[0]

        cellVolume = float(np.prod(self._spacings))
        if self._nDim < 3:
            cellVolume *= thickness

        #: The volume of the whole cell this stencil sits on.
        self._totalVolume = cellVolume

        if self._nDim == 1:
            self._gradients = [cellGradientOperator(self._spacings)]
        else:
            self._gradients = cellCornerGradientOperators(self._spacings)

        self._nMaterialPoints = len(self._gradients)
        self._materialPointVolumes = [cellVolume / self._nMaterialPoints] * self._nMaterialPoints

        self._strainOperators = [cellStrainOperator(gradient) for gradient in self._gradients]
        self._averageOperator = cellAverageOperator(self._nDim)

        self._materialRoutineName = stressStates[stressState]

        # node-major, field-minor: the displacement components of a corner are followed by
        # its nonlocal variable, which is the order the DofManager derives from `fields`
        nDofPerNode = self._nDim + 1
        self._nDof = self._nNodes * nDofPerNode

        self._displacementDofs = np.array(
            [corner * nDofPerNode + i for corner in range(self._nNodes) for i in range(self._nDim)],
            dtype=int,
        )
        self._nonlocalDofs = np.array(
            [corner * nDofPerNode + self._nDim for corner in range(self._nNodes)],
            dtype=int,
        )

        self._displacementBlock = np.ix_(self._displacementDofs, self._displacementDofs)
        self._displacementNonlocalBlock = np.ix_(self._displacementDofs, self._nonlocalDofs)
        self._nonlocalDisplacementBlock = np.ix_(self._nonlocalDofs, self._displacementDofs)
        self._nonlocalBlock = np.ix_(self._nonlocalDofs, self._nonlocalDofs)

        self._fields = [["displacement", nonlocalField] for _ in range(self._nNodes)]

        self._response = GradientEnhancedResponse.createZero(1)
        self._tangents = GradientEnhancedTangents.createZero(1)
        self._increment = GradientEnhancedIncrement.createZero(1)

        self._material = None
        self._nodes = None

    # -- descriptive properties ---------------------------------------------------------

    @property
    def stencilNumber(self) -> int:
        return self._stencilNumber

    @property
    def nNodes(self) -> int:
        return self._nNodes

    @property
    def nodes(self) -> list[Node]:
        return self._nodes

    @property
    def nDof(self) -> int:
        return self._nDof

    @property
    def fields(self) -> list[list[str]]:
        return self._fields

    @property
    def hasMaterial(self) -> bool:
        return self._material is not None

    @property
    def cornerOffsets(self) -> np.ndarray:
        """The corner offsets of the cell this stencil sits on."""

        return self._cornerOffsets

    @property
    def nMaterialPoints(self) -> int:
        """The number of material points this stencil owns."""

        return self._nMaterialPoints

    @property
    def nonlocalField(self) -> str:
        """The name of the nonlocal field."""

        return self._nonlocalField

    # -- setup --------------------------------------------------------------------------

    def setNodes(self, nodes: list[Node]):
        """Assign the corner grid points of the cell.

        Parameters
        ----------
        nodes
            The corner grid points, ordered like :attr:`cornerOffsets`.
        """

        if len(nodes) != self._nNodes:
            raise ValueError("This stencil couples {:} grid points, got {:}.".format(self._nNodes, len(nodes)))

        self._nodes = list(nodes)

        self._centerCoordinates = np.mean([n.coordinates[: self._nDim] for n in self._nodes], axis=0)

    def setMaterial(self, material):
        """Assign the material this stencil evaluates.

        Parameters
        ----------
        material
            A material with a single nonlocal variable, honoring
            :mod:`edelweissfe.materials.base.basegradientenhancedhypoelasticmaterial`.
        """

        if material.nNonlocalVariables != 1:
            raise ValueError("This stencil supports materials with a single nonlocal variable only.")

        self._material = material
        self._materialRoutine = getattr(material, self._materialRoutineName)

    def initializeStencil(self):
        """Create the state variables and let the material initialize them."""

        if self._nodes is None:
            raise ValueError("The grid points of this stencil have not been assigned yet.")

        if not self.hasMaterial:
            raise ValueError("No material has been assigned to this stencil.")

        # stress, strain, the local driving variable and the nonlocal variable
        self._nStateVarsOverhead = 2 * nVoigtComponents + 2

        nStateVarsPerPoint = self._nStateVarsOverhead + self._material.getNumberOfRequiredStateVars()

        self._stateVars = np.zeros((self._nMaterialPoints, nStateVarsPerPoint))
        self._stateVarsTemp = np.zeros((self._nMaterialPoints, nStateVarsPerPoint))

        self._stresses = [self._stateVarsTemp[p, 0:nVoigtComponents] for p in range(self._nMaterialPoints)]
        self._strains = [
            self._stateVarsTemp[p, nVoigtComponents : 2 * nVoigtComponents] for p in range(self._nMaterialPoints)
        ]
        self._localDrivingVariables = [
            self._stateVarsTemp[p, 2 * nVoigtComponents : 2 * nVoigtComponents + 1]
            for p in range(self._nMaterialPoints)
        ]
        self._nonlocalVariables = [
            self._stateVarsTemp[p, 2 * nVoigtComponents + 1 : 2 * nVoigtComponents + 2]
            for p in range(self._nMaterialPoints)
        ]
        self._materialStateVars = [
            self._stateVarsTemp[p, self._nStateVarsOverhead :] for p in range(self._nMaterialPoints)
        ]

        for p in range(self._nMaterialPoints):
            self._material.assignCurrentStateVars(self._materialStateVars[p])

            if hasattr(self._material, "initializeYourself"):
                self._material.initializeYourself()

        self.acceptLastState()

    @property
    def characteristicLength(self) -> float:
        """The characteristic length of a material point, i.e. the geometric mean of the grid
        spacings it covers. The out-of-plane thickness deliberately does not enter."""

        subdivision = 1 if self._nDim == 1 else 2

        return float(np.prod(self._spacings / subdivision) ** (1.0 / self._nDim))

    # -- evaluation ---------------------------------------------------------------------

    def computeKernels(
        self,
        K: np.ndarray,
        P: np.ndarray,
        U: np.ndarray,
        dU: np.ndarray,
        time: float,
        dT: float,
    ):
        self._stateVarsTemp[:, :] = self._stateVars

        response = self._response
        tangents = self._tangents
        increment = self._increment

        N = self._averageOperator

        nonlocalCornerValues = U[self._nonlocalDofs]
        nonlocalCornerIncrements = dU[self._nonlocalDofs]

        displacementIncrements = dU[self._displacementDofs]

        PDisplacement = np.zeros(self._displacementDofs.size)
        PNonlocal = np.zeros(self._nonlocalDofs.size)

        KDisplacement = np.zeros((self._displacementDofs.size, self._displacementDofs.size))
        KDisplacementNonlocal = np.zeros((self._displacementDofs.size, self._nonlocalDofs.size))
        KNonlocalDisplacement = np.zeros((self._nonlocalDofs.size, self._displacementDofs.size))
        KNonlocal = np.zeros((self._nonlocalDofs.size, self._nonlocalDofs.size))

        for p in range(self._nMaterialPoints):
            B = self._strainOperators[p]
            G = self._gradients[p]
            volume = self._materialPointVolumes[p]

            increment.dStrain[:] = B @ displacementIncrements
            increment.K[0] = N @ nonlocalCornerValues
            increment.dK[0] = N @ nonlocalCornerIncrements

            # the stress enters the rate form material as the stress of the last increment
            response.stress[:] = self._stresses[p]
            response.KLocal[:] = self._localDrivingVariables[p]
            response.c[:] = 0.0
            tangents.zero()

            self._material.assignCurrentStateVars(self._materialStateVars[p])

            self._materialRoutine(response, tangents, increment, time, dT)

            self._strains[p] += increment.dStrain
            self._stresses[p][:] = response.stress
            self._localDrivingVariables[p][:] = response.KLocal
            self._nonlocalVariables[p][:] = increment.K

            nonlocalGradient = G @ nonlocalCornerValues
            GTG = G.T @ G

            c = response.c[0]
            KLocal = response.KLocal[0]
            nonlocalValue = increment.K[0]

            # residuals
            PDisplacement += B.T @ response.stress * volume
            PNonlocal += (N * (nonlocalValue - KLocal) + c * (G.T @ nonlocalGradient)) * volume

            # tangents
            KDisplacement += B.T @ tangents.dStress_dStrain @ B * volume
            KDisplacementNonlocal += np.outer(B.T @ tangents.dStress_dK[:, 0], N) * volume
            KNonlocalDisplacement += np.outer(N, -tangents.dKLocal_dStrain[0, :] @ B) * volume
            KNonlocal += (
                np.outer(N, N) * (1.0 - tangents.dKLocal_dK[0, 0])
                + c * GTG
                + np.outer(GTG @ nonlocalCornerValues, tangents.dc_dK[0, 0] * N)
            ) * volume

        P[self._displacementDofs] += PDisplacement
        P[self._nonlocalDofs] += PNonlocal

        K[self._displacementBlock] += KDisplacement
        K[self._displacementNonlocalBlock] += KDisplacementNonlocal
        K[self._nonlocalDisplacementBlock] += KNonlocalDisplacement
        K[self._nonlocalBlock] += KNonlocal

    def computeBodyForce(
        self,
        P: np.ndarray,
        K: np.ndarray,
        load: np.ndarray,
        U: np.ndarray,
        time: float,
        dT: float,
    ):
        """Distribute a force per unit volume onto the displacement degrees of freedom of the
        corner grid points. The nonlocal field is not loaded."""

        load = np.asarray(load, dtype=float)[: self._nDim]

        shareOfVolume = self._totalVolume / self._nNodes

        P[self._displacementDofs] += np.tile(load, self._nNodes) * shareOfVolume

    def acceptLastState(self):
        self._stateVars[:, :] = self._stateVarsTemp

    def resetToLastValidState(self):
        self._stateVarsTemp[:, :] = self._stateVars

    # -- results ------------------------------------------------------------------------

    def getResultArray(self, result: str, quadraturePoint: int = 0, getPersistentView: bool = True) -> np.ndarray:
        if result == "stress":
            return np.array(self._stresses[quadraturePoint], copy=not getPersistentView)
        if result == "strain":
            return np.array(self._strains[quadraturePoint], copy=not getPersistentView)
        if result == "local driving variable":
            return np.array(self._localDrivingVariables[quadraturePoint], copy=not getPersistentView)
        if result == "nonlocal variable":
            return np.array(self._nonlocalVariables[quadraturePoint], copy=not getPersistentView)

        self._material.assignCurrentStateVars(self._materialStateVars[quadraturePoint])

        return np.array(self._material.getResult(result), copy=not getPersistentView)

    def getNumberOfQuadraturePoints(self) -> int:
        return self._nMaterialPoints

    def getCoordinatesAtCenter(self) -> np.ndarray:
        return self._centerCoordinates
