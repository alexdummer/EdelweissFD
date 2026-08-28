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
A Generalized Finite Difference (GFDM) gradient plasticity stencil, one per grid node.

:class:`~edelweissfd.stencils.gradientplasticitystencil.GradientPlasticityStencil` owns one
cell and samples its material once per corner, which is what needs a one-sided quotient to
avoid the hourglass mode, and is why that quotient is only first order accurate. This stencil
instead owns exactly one material point -- the grid point itself -- and builds both the strain
operator and the plastic multiplier Laplacian from the *same* weighted-least-squares neighbour
cloud, see :mod:`edelweissfd.operators.gfdm`. There is no cell, no corner, and no hourglass mode
to control in the first place: verified by direct eigenvalue inspection of a small elastic patch
under the panel example's own boundary conditions (see the session notes), and second order
accurate in the interior, verified against a manufactured field down to machine precision.

The trade-off is a wider, denser molecule (each node couples to roughly eight neighbours rather
than the four corners of one cell) and a boundary treatment that is only as accurate as a
quadratic local model allows -- first order for the Laplacian specifically, the same limitation
:func:`~edelweissfd.operators.gfdm.gfdmWeights`'s quadratic model has anywhere the neighbour
cloud is one-sided. A cubic local model recovers second order there too (verified separately)
but needs a larger cloud than is implemented here; left for later.
"""

import numpy as np
from edelweissfe.materials.base.basegradientplasticityhypoelasticmaterial import (
    GradientPlasticityIncrement,
    GradientPlasticityResponse,
    GradientPlasticityTangents,
)

from edelweissfd.operators.differences import (
    cellStrainOperator,
    condensePlaneStressTangents,
    nVoigtComponents,
)
from edelweissfd.operators.gfdm import gatherCloud, gfdmWeights
from edelweissfd.stencils.base.basestencil import BaseStencil
from edelweissfd.stencils.numericaltangent import NumericalTangentMixin

stressStates = {
    "3d": "computeStress",
    "plane strain": "computeStress",
    "plane stress": "computePlaneStress",
}


class GFDMGradientPlasticityStencil(BaseStencil, NumericalTangentMixin):
    """A gradient plasticity stencil owning exactly one material point: the grid node itself.

    Parameters
    ----------
    stencilNumber
        A unique number for this stencil.
    spacings
        The grid spacing per direction, shape ``(1,)`` or ``(2,)``. Only one and two
        dimensions are implemented.
    stressState
        One of ``3d``, ``plane strain`` or ``plane stress``.
    thickness
        An out-of-plane thickness.
    minCloudPoints
        The minimum neighbour cloud size, see :func:`~edelweissfd.operators.gfdm.gatherCloud`.
        Defaults to a cloud roughly 1.6x the quadratic Taylor model's unknown count -- 8 for
        two dimensions (5 unknowns), 4 for one (2 unknowns: :math:`u_x`, :math:`u_{xx}`, no
        cross term). The one dimensional default matters concretely: at a domain boundary the
        cloud can only be gathered one-sided, and :func:`~edelweissfd.operators.gfdm.gatherCloud`
        caps how far it searches, so the two dimensional default of 8 is unreachable one-sided
        within that cap and would make every 1D boundary node fail to construct.
    """

    def __init__(
        self,
        stencilNumber: int,
        spacings,
        stressState: str = "plane strain",
        thickness: float = 1.0,
        minCloudPoints: int = None,
    ):
        if stressState not in stressStates:
            raise ValueError(
                "Unknown stress state '{:}'; available are {:}".format(stressState, ", ".join(stressStates))
            )

        self._stencilNumber = stencilNumber
        self._stressState = stressState
        self._materialRoutineName = stressStates[stressState]

        self._spacings = np.asarray(spacings, dtype=float)
        self._nDim = self._spacings.size

        if self._nDim not in (1, 2):
            raise ValueError("GFDMGradientPlasticityStencil is only implemented in one or two dimensions.")

        if self._nDim == 1 and stressState != "3d":
            raise ValueError("A one dimensional grid needs the stress state '3d'.")

        if minCloudPoints is None:
            minCloudPoints = 8 if self._nDim == 2 else 4

        self._minCloudPoints = minCloudPoints
        self._thickness = thickness

        # set properly in setNode(), once the node's position relative to the grid boundary is
        # known -- a boundary node's tributary volume is smaller than an interior one's
        self._totalVolume = None

        self._nMaterialPoints = 1

        self._response = GradientPlasticityResponse.createZero(1)
        self._tangents = GradientPlasticityTangents.createZero(1)
        self._increment = GradientPlasticityIncrement.createZero(1)

        self._material = None
        self._nodes = None

    # -- descriptive properties ---------------------------------------------------------

    @property
    def stencilNumber(self) -> int:
        return self._stencilNumber

    @property
    def nNodes(self) -> int:
        return len(self._nodes)

    @property
    def nodes(self) -> list:
        return self._nodes

    @property
    def nDof(self) -> int:
        return self._nDof

    @property
    def fields(self) -> list:
        return self._fields

    @property
    def hasMaterial(self) -> bool:
        return self._material is not None

    @property
    def nMaterialPoints(self) -> int:
        return self._nMaterialPoints

    @property
    def visualizationNodes(self) -> list:
        """Only the centre: that is where this stencil's material point sits."""

        return [self._nodes[0]]

    @property
    def characteristicLength(self) -> float:
        return float(np.prod(self._spacings) ** (1.0 / self._nDim))

    @property
    def cornerOffsets(self):
        raise NotImplementedError("GFDMGradientPlasticityStencil has no cell to speak of; use setNode.")

    # -- setup --------------------------------------------------------------------------

    def setNode(self, grid, node):
        """Attach the stencil to a grid node, gathering its GFDM neighbour cloud.

        Parameters
        ----------
        grid
            The structured grid the node belongs to.
        node
            The grid point this stencil's material point sits at.
        """

        neighbours = gatherCloud(grid, node, minPoints=self._minCloudPoints)
        self._nodes = [node] + list(neighbours)

        centreCoords = node.coordinates[: self._nDim]
        neighbourCoords = np.array([n.coordinates[: self._nDim] for n in neighbours])

        # tributary (box/midpoint rule) volume: half the spacing in any direction the node sits
        # on the grid's boundary in, full spacing otherwise. Skipping this and using the full
        # cell volume everywhere -- fine for a domain wide enough that boundary nodes are a small
        # minority -- overcounts the total volume substantially on a narrow domain, where most
        # nodes sit on a boundary in at least one direction (found via a 33% peak-load
        # discrepancy on a 3-cell-wide strip: the uncorrected total volume there was 38% too
        # large).
        gridIndex = grid.gridIndexOf(node)
        gridShape = grid.shape
        tributarySpacings = [
            spacing * (0.5 if gridIndex[d] in (0, gridShape[d] - 1) else 1.0)
            for d, spacing in enumerate(self._spacings)
        ]
        self._totalVolume = float(np.prod(tributarySpacings)) * self._thickness

        gradientWeights, laplacianWeights = gfdmWeights(centreCoords, neighbourCoords)

        # centre coefficient = -sum(neighbours), so a constant field has a vanishing derivative
        gradientOperator = np.stack(
            [np.concatenate([[-row.sum()], row]) for row in gradientWeights]
        )  # (nDim, nMolecule), centre first
        self._laplacianWeights = np.concatenate([[-laplacianWeights.sum()], laplacianWeights])

        self._strainOperator = cellStrainOperator(gradientOperator)  # (6, nMolecule*nDim)

        self._centerCoordinates = centreCoords
        self._createDofIndices()

    def _createDofIndices(self):
        """Every node of the molecule carries both fields: the strain at the centre depends on
        every neighbour's displacement, and the Laplacian at the centre depends on every
        neighbour's plastic multiplier -- unlike the corner-sampled stencil, there is no node in
        the molecule that carries only one of the two."""

        nMolecule = len(self._nodes)
        slotsPerNode = self._nDim + 1  # nDim displacement components + 1 multiplier

        self._fields = [["displacement", "plastic multiplier"] for _ in range(nMolecule)]

        self._displacementDofs = np.array(
            [slotsPerNode * i + c for i in range(nMolecule) for c in range(self._nDim)], dtype=int
        )
        self._multiplierDofs = np.array([slotsPerNode * i + self._nDim for i in range(nMolecule)], dtype=int)

        self._nDof = slotsPerNode * nMolecule

        self._displacementBlock = np.ix_(self._displacementDofs, self._displacementDofs)
        self._displacementMultiplierBlock = np.ix_(self._displacementDofs, self._multiplierDofs)
        self._multiplierDisplacementBlock = np.ix_(self._multiplierDofs, self._displacementDofs)
        self._multiplierBlock = np.ix_(self._multiplierDofs, self._multiplierDofs)

    def setMaterial(self, material):
        if material.nYieldSurfaces != 1:
            raise ValueError("This stencil supports materials with a single yield surface only.")

        self._material = material
        self._materialRoutine = getattr(material, self._materialRoutineName)

    def initializeStencil(self):
        if self._nodes is None:
            raise ValueError("The grid node of this stencil has not been assigned yet; use setNode().")

        if not self.hasMaterial:
            raise ValueError("No material has been assigned to this stencil.")

        self._nStateVarsOverhead = 2 * nVoigtComponents + 2
        nStateVarsTotal = self._nStateVarsOverhead + self._material.getNumberOfRequiredStateVars()

        self._stateVars = np.zeros(nStateVarsTotal)
        self._stateVarsTemp = np.zeros(nStateVarsTotal)

        self._stress = self._stateVarsTemp[0:nVoigtComponents]
        self._strain = self._stateVarsTemp[nVoigtComponents : 2 * nVoigtComponents]
        self._multiplierState = self._stateVarsTemp[2 * nVoigtComponents : 2 * nVoigtComponents + 1]
        self._yieldState = self._stateVarsTemp[2 * nVoigtComponents + 1 : 2 * nVoigtComponents + 2]
        self._materialStateVars = self._stateVarsTemp[self._nStateVarsOverhead :]

        self._material.assignCurrentStateVars(self._materialStateVars)
        if hasattr(self._material, "initializeYourself"):
            self._material.initializeYourself()

        self.acceptLastState()

        self._response = GradientPlasticityResponse(stress=self._stress, f=self._yieldState)
        self._tangents = GradientPlasticityTangents(
            dStress_dStrain=np.zeros((nVoigtComponents, nVoigtComponents)),
            dStress_dLambda=np.zeros((nVoigtComponents, 1)),
            dStress_dLaplacian=np.zeros((nVoigtComponents, 1)),
            dF_dStrain=np.zeros((1, nVoigtComponents)),
            dF_dLambda=np.zeros((1, 1)),
            dF_dLaplacian=np.zeros((1, 1)),
        )
        self._increment = GradientPlasticityIncrement(
            dStrain=np.zeros(nVoigtComponents), dLambda=np.zeros(1), laplaceDLambda=np.zeros(1)
        )

    # -- evaluation ---------------------------------------------------------------------

    def computeKernels(self, K: np.ndarray, P: np.ndarray, U: np.ndarray, dU: np.ndarray, time: float, dT: float):
        self._stateVarsTemp[:] = self._stateVars

        B = self._strainOperator
        L = self._laplacianWeights

        displacementIncrements = dU[self._displacementDofs]
        multiplierIncrements = dU[self._multiplierDofs]
        multipliers = U[self._multiplierDofs]

        self._increment.dStrain[:] = B @ displacementIncrements
        self._increment.dLambda[:] = multiplierIncrements[0]
        self._increment.laplaceDLambda[:] = L @ multiplierIncrements

        self._material.assignCurrentStateVars(self._materialStateVars)
        self._materialRoutine(self._response, self._tangents, self._increment, time, dT)

        if self._materialRoutineName == "computePlaneStress":
            # Marmot's computePlaneStress returns an uncondensed tangent -- see
            # edelweissfd.operators.differences.condensePlaneStressTangents.
            t = self._tangents
            Dcond, dLambdaCond, dLapCond, dFdEcond, dFdLambdaCond, dFdLapCond = condensePlaneStressTangents(
                t.dStress_dStrain,
                t.dStress_dLambda[:, 0],
                t.dStress_dLaplacian[:, 0],
                t.dF_dStrain[0, :],
                t.dF_dLambda[0, 0],
                t.dF_dLaplacian[0, 0],
            )
            t.dStress_dStrain[...] = Dcond
            t.dStress_dLambda[:, 0] = dLambdaCond
            t.dStress_dLaplacian[:, 0] = dLapCond
            t.dF_dStrain[0, :] = dFdEcond
            t.dF_dLambda[0, 0] = dFdLambdaCond
            t.dF_dLaplacian[0, 0] = dFdLapCond

        self._strain += self._increment.dStrain
        self._multiplierState[0] = multipliers[0]

        V = self._totalVolume

        localTangent = np.zeros((self._nDof, self._nDof))
        localFlux = np.zeros(self._nDof)

        localFlux[self._displacementDofs] = B.T @ self._stress * V
        localFlux[self._multiplierDofs[0]] += self._yieldState[0] * V

        dStress_dStrain = self._tangents.dStress_dStrain
        dStress_dLambda = self._tangents.dStress_dLambda
        dStress_dLaplacian = self._tangents.dStress_dLaplacian
        dF_dStrain = self._tangents.dF_dStrain
        dF_dLambda = self._tangents.dF_dLambda
        dF_dLaplacian = self._tangents.dF_dLaplacian

        localTangent[np.ix_(self._displacementDofs, self._displacementDofs)] = V * (B.T @ dStress_dStrain @ B)

        # the plastic multiplier enters the stress directly at the centre and through the
        # Laplacian over the whole molecule
        KUL = V * (B.T @ dStress_dLambda)  # (nDispDofs, 1), acts on the centre's multiplier only
        KULLaplace = V * np.outer(B.T @ dStress_dLaplacian, L)  # (nDispDofs, nMolecule)
        localTangent[np.ix_(self._displacementDofs, [self._multiplierDofs[0]])] += KUL
        localTangent[np.ix_(self._displacementDofs, self._multiplierDofs)] += KULLaplace

        localTangent[self._multiplierDofs[0], self._displacementDofs] += V * (dF_dStrain @ B)[0]
        localTangent[self._multiplierDofs[0], self._multiplierDofs[0]] += V * dF_dLambda[0, 0]
        localTangent[self._multiplierDofs[0], self._multiplierDofs] += V * dF_dLaplacian[0, 0] * L

        P[:] += localFlux
        K[:, :] += localTangent

    def acceptLastState(self):
        self._stateVars[:] = self._stateVarsTemp

    def resetToLastValidState(self):
        self._stateVarsTemp[:] = self._stateVars

    # -- results ------------------------------------------------------------------------

    def getResultArray(self, result: str, quadraturePoint: int = 0, getPersistentView: bool = True) -> np.ndarray:
        if result == "stress":
            return np.array(self._stress, copy=not getPersistentView)
        if result == "strain":
            return np.array(self._strain, copy=not getPersistentView)
        if result == "plastic multiplier":
            return np.array(self._multiplierState, copy=not getPersistentView)
        if result == "yield function":
            return np.array(self._yieldState, copy=not getPersistentView)

        self._material.assignCurrentStateVars(self._materialStateVars)
        return np.array(self._material.getResult(result), copy=not getPersistentView)

    def getNumberOfQuadraturePoints(self) -> int:
        return 1

    def getCoordinatesAtCenter(self) -> np.ndarray:
        return self._centerCoordinates
