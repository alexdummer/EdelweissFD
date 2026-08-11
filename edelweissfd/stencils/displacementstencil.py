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
The finite difference stencil for the balance of linear momentum.

One stencil sits on one grid cell and owns the material points of that cell. The strain at
a material point is a difference quotient of the corner displacements,

.. math::
    \\boldsymbol \\varepsilon = \\boldsymbol B \\, \\boldsymbol u_{corners}

and the internal flux is the discrete adjoint of the divergence,

.. math::
    \\boldsymbol P = \\sum_{p} \\boldsymbol B_p^T \\boldsymbol \\sigma_p \\; V_p,
    \\qquad
    \\boldsymbol K = \\sum_{p} \\boldsymbol B_p^T \\boldsymbol C_p \\, \\boldsymbol B_p \\; V_p

so that summing the contributions of all cells adjacent to a grid point yields the compact
difference approximation of :math:`-\\nabla \\cdot \\boldsymbol \\sigma` there. Traction
free boundaries follow without any further treatment, and a prescribed traction enters as
equivalent nodal forces.

**Where the material points sit.** In one dimension a single point per cell suffices and
the strain is the compact quotient :math:`(u_{i+1} - u_i)/h`. In two and three dimensions a
single point per cell would have to use a gradient averaged over all parallel cell edges,
and that averaged operator annihilates the hourglass (checkerboard) mode, i.e. the scheme
would admit spurious zero energy displacement patterns. EdelweissFD therefore samples the
cell at its :math:`2^n` corners with plain one-sided difference quotients, see
:func:`~edelweissfd.operators.differences.cellCornerGradientOperators`, each owning
:math:`2^{-n}` of the cell volume. In one dimension both corner operators reduce to the
same compact quotient, so the one dimensional scheme is unaffected.
"""

import numpy as np
from edelweissfe.points.node import Node

from edelweissfd.operators.differences import (
    cellCornerGradientOperators,
    cellCornerOffsets,
    cellGradientOperator,
    cellStrainOperator,
    nVoigtComponents,
)
from edelweissfd.stencils.base.basestencil import BaseStencil
from edelweissfd.stencils.numericaltangent import NumericalTangentMixin

#: The supported stress states, mapping to the material routine and the Voigt components
#: which carry the reduced tangent.
stressStates = {
    "3d": {"materialRoutine": "computeStress", "activeVoigtIndices": (0, 1, 2, 3, 4, 5)},
    "plane strain": {"materialRoutine": "computeStress", "activeVoigtIndices": (0, 1, 2, 3, 4, 5)},
    "plane stress": {"materialRoutine": "computePlaneStress", "activeVoigtIndices": (0, 1, 3)},
    "uniaxial stress": {"materialRoutine": "computeUniaxialStress", "activeVoigtIndices": (0,)},
}

#: Number of state variables the stencil keeps per material point in front of the
#: material's own ones, namely the stress and the strain.
nStateVarsOverheadPerMaterialPoint = 2 * nVoigtComponents


class DisplacementStencil(BaseStencil, NumericalTangentMixin):
    """A momentum balance stencil on a single grid cell.

    Parameters
    ----------
    stencilNumber
        A unique number for this stencil.
    spacings
        The grid spacing per direction, shape ``(nDim,)``.
    stressState
        One of ``3d``, ``plane strain``, ``plane stress`` or ``uniaxial stress``. It selects
        the material routine and hence which components are condensed out by the material.
    thickness
        An out-of-plane thickness, only meaningful for the two dimensional stress states.
    """

    def __init__(
        self,
        stencilNumber: int,
        spacings,
        stressState: str = "plane strain",
        thickness: float = 1.0,
    ):
        if stressState not in stressStates:
            raise ValueError(
                "Unknown stress state '{:}'; available are {:}".format(stressState, ", ".join(stressStates))
            )

        self._stencilNumber = stencilNumber
        self._stressState = stressState

        self._spacings = np.asarray(spacings, dtype=float)
        self._nDim = self._spacings.size

        if self._nDim == 1 and stressState in ("plane strain", "plane stress"):
            raise ValueError("A one dimensional grid needs the stress state 'uniaxial stress' or '3d'.")

        self._cornerOffsets = cellCornerOffsets(self._nDim)
        self._nNodes = self._cornerOffsets.shape[0]

        cellVolume = float(np.prod(self._spacings))
        if self._nDim < 3:
            cellVolume *= thickness

        #: The volume of the whole cell this stencil sits on.
        self._totalVolume = cellVolume

        self._gradients, self._materialPointVolumes = self._createMaterialPointOperators(cellVolume)
        self._nMaterialPoints = len(self._gradients)

        self._activeVoigtIndices = np.array(stressStates[stressState]["activeVoigtIndices"], dtype=int)
        self._materialRoutineName = stressStates[stressState]["materialRoutine"]

        self._strainOperators = [cellStrainOperator(gradient) for gradient in self._gradients]
        self._activeStrainOperators = [
            np.ascontiguousarray(B[self._activeVoigtIndices, :]) for B in self._strainOperators
        ]

        self._nDof = self._nNodes * self._nDim
        self._fields = [["displacement"] for _ in range(self._nNodes)]

        nActive = self._activeVoigtIndices.size
        self._tangent = np.zeros((nActive, nActive))

        self._material = None
        self._nodes = None

    def _createMaterialPointOperators(self, cellVolume: float) -> tuple:
        """The gradient operator and the volume of every material point of the cell.

        Parameters
        ----------
        cellVolume
            The volume of the whole cell.

        Returns
        -------
        tuple
            The list of gradient operators and the list of material point volumes.
        """

        if self._nDim == 1:
            # both corner operators coincide with the compact quotient, so one point suffices
            return [cellGradientOperator(self._spacings)], [cellVolume]

        gradients = cellCornerGradientOperators(self._spacings)
        volumes = [cellVolume / len(gradients)] * len(gradients)

        return gradients, volumes

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
    def strainOperators(self) -> list:
        """The full six component strain operator of every material point."""

        return self._strainOperators

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
            A material honoring
            :class:`~edelweissfe.materials.base.basehypoelasticmaterial.BaseHypoElasticMaterial`.
            The instance must belong to this stencil alone: it is re-pointed at the state
            variables of each material point during evaluation, so sharing it with another
            stencil is a data race as soon as the solver computes stencils on several threads.
            See :mod:`edelweissfd.materials.provider`.
        """

        self._material = material
        self._materialRoutine = getattr(material, self._materialRoutineName)

    def initializeStencil(self):
        """Create the state variables and let the material initialize them."""

        if self._nodes is None:
            raise ValueError("The grid points of this stencil have not been assigned yet.")

        if not self.hasMaterial:
            raise ValueError("No material has been assigned to this stencil.")

        nStateVarsPerPoint = nStateVarsOverheadPerMaterialPoint + self._material.getNumberOfRequiredStateVars()

        self._stateVars = np.zeros((self._nMaterialPoints, nStateVarsPerPoint))
        self._stateVarsTemp = np.zeros((self._nMaterialPoints, nStateVarsPerPoint))

        self._stresses = [self._stateVarsTemp[p, 0:nVoigtComponents] for p in range(self._nMaterialPoints)]
        self._strains = [
            self._stateVarsTemp[p, nVoigtComponents : 2 * nVoigtComponents] for p in range(self._nMaterialPoints)
        ]
        self._materialStateVars = [
            self._stateVarsTemp[p, nStateVarsOverheadPerMaterialPoint:] for p in range(self._nMaterialPoints)
        ]

        for p in range(self._nMaterialPoints):
            self._material.assignCurrentStateVars(self._materialStateVars[p])

            if hasattr(self._material, "setCharacteristicElementLength"):
                self._material.setCharacteristicElementLength(self.characteristicLength)

            if hasattr(self._material, "initializeYourself"):
                self._material.initializeYourself()

        self.acceptLastState()

    @property
    def characteristicLength(self) -> float:
        """The characteristic length of a material point, communicated to regularized
        materials.

        It is the geometric mean of the grid spacings a material point covers, i.e. the whole
        cell in one dimension and half a cell per direction where the cell is sampled at its
        corners. The out-of-plane thickness deliberately does not enter."""

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

        for p in range(self._nMaterialPoints):
            dStrain = self._strainOperators[p] @ dU

            stress = self._stresses[p]

            # The material only holds a pointer to its state variables, so a shared
            # material instance has to be pointed at this material point's block.
            self._material.assignCurrentStateVars(self._materialStateVars[p])

            self._tangent[:, :] = 0.0

            self._materialRoutine(stress, self._tangent, dStrain, time, dT)

            self._strains[p] += dStrain

            B = self._activeStrainOperators[p]
            volume = self._materialPointVolumes[p]

            P += B.T @ stress[self._activeVoigtIndices] * volume
            K += B.T @ self._tangent @ B * volume

    def computeBodyForce(
        self,
        P: np.ndarray,
        K: np.ndarray,
        load: np.ndarray,
        U: np.ndarray,
        time: float,
        dT: float,
    ):
        """Distribute a force per unit volume onto the corner grid points.

        Each corner owns an equal share of the cell volume, so the equivalent grid point
        forces are the load times that share. For a uniform load this is exactly what a
        bilinear finite element with a two by two quadrature yields.
        """

        load = np.asarray(load, dtype=float)[: self._nDim]

        shareOfVolume = self._totalVolume / self._nNodes

        P += np.tile(load, self._nNodes) * shareOfVolume

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

        self._material.assignCurrentStateVars(self._materialStateVars[quadraturePoint])

        return np.array(self._material.getResult(result), copy=not getPersistentView)

    def getNumberOfQuadraturePoints(self) -> int:
        return self._nMaterialPoints

    def getCoordinatesAtCenter(self) -> np.ndarray:
        return self._centerCoordinates
