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
The multifield finite difference stencil for gradient plasticity.

Two fields live on every grid point: the ``displacement`` and the ``plastic multiplier``
:math:`\\lambda`. The equations solved are the balance of linear momentum and the yield
condition, which for this material family depends on the Laplacian of the plastic multiplier,

.. math::
    -\\nabla \\cdot \\boldsymbol \\sigma = \\boldsymbol 0
    \\qquad
    f\\left( \\boldsymbol \\sigma,\\, \\lambda,\\, \\nabla^2 \\lambda \\right) = 0

as implemented by Marmot's ``MarmotMaterialGradientPlasticityHypoElastic``; ``GRADIENTVONMISES``
is such a material.

**No auxiliary field, no penalty.** Marmot's own ``C0GradientPlasticityFiniteElement`` carries
a third field for :math:`\\nabla \\lambda` and ties it to the plastic multiplier by a penalty,
because a C0 shape function cannot deliver a second derivative. A finite difference operator
has no such restriction: the Laplacian is simply the compact three point second difference per
direction, see :meth:`~edelweissfd.grids.structuredgrid.StructuredGrid.laplacianAt`. That
removes one field, the penalty parameter and the ill-conditioning it brings, and leaves a
genuine two field problem.

The price is that the molecule is wider than a single cell: the second difference at a corner
reaches one grid point beyond it. The stencil therefore couples the corners of its cell plus
the neighbours entering the Laplacian, and declares the ``displacement`` field only on the
corners, since the outer grid points contribute through the plastic multiplier alone.

At the boundary of the grid the missing neighbour is a ghost node, eliminated by the
homogeneous Neumann condition :math:`\\partial \\lambda / \\partial n = 0`, i.e. no plastic
flux through the boundary.

Discretely, per material point of the cell, with :math:`\\boldsymbol B` the strain operator of
the cell and :math:`\\boldsymbol L` the Laplacian coefficients over the molecule,

.. math::
    \\boldsymbol P_u = \\boldsymbol B^T \\boldsymbol \\sigma V,
    \\qquad
    P_{\\lambda, p} = f \\, V

where the yield condition is collocated at the material point, i.e. lumped onto the grid point
it sits on.
"""

import numpy as np
from edelweissfe.materials.base.basegradientplasticityhypoelasticmaterial import (
    GradientPlasticityIncrement,
    GradientPlasticityResponse,
    GradientPlasticityTangents,
)
from edelweissfe.points.node import Node

from edelweissfd.operators.differences import (
    cellCornerGradientOperators,
    cellCornerOffsets,
    cellGradientOperator,
    cellStrainOperator,
    nVoigtComponents,
    volumetricallyAveragedStrainOperators,
)
from edelweissfd.stencils.base.basestencil import BaseStencil
from edelweissfd.stencils.numericaltangent import NumericalTangentMixin

#: The supported stress states, mapping to the material routine.
stressStates = {
    "3d": "computeStress",
    "plane strain": "computeStress",
    "plane stress": "computePlaneStress",
}


class GradientPlasticityStencil(BaseStencil, NumericalTangentMixin):
    """A gradient plasticity stencil on a single grid cell.

    Parameters
    ----------
    stencilNumber
        A unique number for this stencil.
    spacings
        The grid spacing per direction, shape ``(nDim,)``.
    stressState
        One of ``3d``, ``plane strain`` or ``plane stress``.
    thickness
        An out-of-plane thickness, only meaningful for the two dimensional stress states.
    volumetricAveraging
        Average the volumetric part of the strain operator over the cell, the classical B-bar or
        mean dilatation treatment, see
        :func:`~edelweissfd.operators.differences.volumetricallyAveragedStrainOperators`. On by
        default: corner sampling constrains the volumetric strain once per corner, which
        over-constrains the cell as Poisson's ratio approaches one half, and a shear band deforms
        at constant volume, so without it the band simply does not form. Turn it off only to
        reproduce that.
    """

    def __init__(
        self,
        stencilNumber: int,
        spacings,
        stressState: str = "plane strain",
        thickness: float = 1.0,
        volumetricAveraging: bool = True,
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
            raise ValueError("A one dimensional grid needs the stress state '3d'.")

        self._cornerOffsets = cellCornerOffsets(self._nDim)
        self._nCorners = self._cornerOffsets.shape[0]

        cellVolume = float(np.prod(self._spacings))
        if self._nDim < 3:
            cellVolume *= thickness

        self._totalVolume = cellVolume

        if self._nDim == 1:
            self._gradients = [cellGradientOperator(self._spacings)]
        else:
            self._gradients = cellCornerGradientOperators(self._spacings)

        self._nMaterialPoints = len(self._gradients)
        self._materialPointVolumes = [cellVolume / self._nMaterialPoints] * self._nMaterialPoints

        self._strainOperators = [cellStrainOperator(gradient) for gradient in self._gradients]

        # Without this the cell locks volumetrically as Poisson's ratio approaches one half, and a
        # shear band -- which deforms at constant volume -- cannot form at all. On by default,
        # because a gradient plasticity computation is localisation at nearly incompressible
        # plastic flow, i.e. exactly the case that locks.
        self._volumetricAveraging = volumetricAveraging

        if volumetricAveraging:
            self._strainOperators = volumetricallyAveragedStrainOperators(
                self._strainOperators, self._materialPointVolumes
            )

        self._materialRoutineName = stressStates[stressState]

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
    def nCorners(self) -> int:
        """The number of corner grid points, i.e. those carrying the displacement field."""

        return self._nCorners

    @property
    def nMaterialPoints(self) -> int:
        """The number of material points this stencil owns."""

        return self._nMaterialPoints

    @property
    def visualizationNodes(self) -> list[Node]:
        """Only the corners of the cell, not the wider molecule of the Laplacian."""

        return self._nodes[: self._nCorners]

    # -- setup --------------------------------------------------------------------------

    def setCell(self, grid, cellIndex):
        """Attach the stencil to a cell, widening the molecule by the Laplacian's reach.

        Parameters
        ----------
        grid
            The structured grid the cell belongs to.
        cellIndex
            The grid index of the cell's lower left corner.
        """

        corners = grid.cellCornerNodes(cellIndex, self._cornerOffsets)

        # one Laplacian per material point, i.e. per corner in two and three dimensions
        laplacians = [grid.laplacianAt(corners[self._materialPointNode(p)]) for p in range(self._nMaterialPoints)]

        molecule = list(corners)
        for laplacian in laplacians:
            for node in laplacian:
                if node not in molecule:
                    molecule.append(node)

        self._laplacianCoefficients = np.zeros((self._nMaterialPoints, len(molecule)))
        for p, laplacian in enumerate(laplacians):
            for node, coefficient in laplacian.items():
                self._laplacianCoefficients[p, molecule.index(node)] = coefficient

        self.setNodes(molecule)

    def _materialPointNode(self, materialPoint: int) -> int:
        """The corner a material point sits on.

        In one dimension the single material point sits at the centre of the cell; the
        Laplacian is then centred at the lower corner, which keeps it a proper second
        difference.

        Parameters
        ----------
        materialPoint
            The index of the material point.

        Returns
        -------
        int
            The index of the corner.
        """

        return 0 if self._nDim == 1 else materialPoint

    def setNodes(self, nodes: list[Node]):
        """Assign the molecule of this stencil, the cell corners first.

        Parameters
        ----------
        nodes
            The grid points, the first :attr:`nCorners` of them being the cell corners in the
            order of :attr:`cornerOffsets`.
        """

        if len(nodes) < self._nCorners:
            raise ValueError("This stencil needs at least the {:} corners of its cell.".format(self._nCorners))

        self._nodes = list(nodes)

        if not hasattr(self, "_laplacianCoefficients"):
            raise ValueError("Use setCell() so that the Laplacian molecule is built as well.")

        self._centerCoordinates = np.mean([n.coordinates[: self._nDim] for n in self._nodes[: self._nCorners]], axis=0)

        self._createDofIndices()

    def _createDofIndices(self):
        """Create the local index arrays of the two fields.

        The corners carry the displacement and the plastic multiplier, the outer grid points of
        the molecule carry the plastic multiplier alone. Degrees of freedom are node-major and
        field-minor, in the order the fields are listed per node.
        """

        nDim = self._nDim

        self._fields = [
            ["displacement", "plastic multiplier"] if node < self._nCorners else ["plastic multiplier"]
            for node in range(len(self._nodes))
        ]

        displacementDofs = []
        multiplierDofs = []

        offset = 0
        for node, fieldsOnNode in enumerate(self._fields):
            if "displacement" in fieldsOnNode:
                displacementDofs.extend(offset + i for i in range(nDim))
                offset += nDim

            multiplierDofs.append(offset)
            offset += 1

        self._nDof = offset

        self._displacementDofs = np.array(displacementDofs, dtype=int)
        self._multiplierDofs = np.array(multiplierDofs, dtype=int)

        self._displacementBlock = np.ix_(self._displacementDofs, self._displacementDofs)
        self._displacementMultiplierBlock = np.ix_(self._displacementDofs, self._multiplierDofs)
        self._multiplierDisplacementBlock = np.ix_(self._multiplierDofs, self._displacementDofs)
        self._multiplierBlock = np.ix_(self._multiplierDofs, self._multiplierDofs)

    def setMaterial(self, material):
        """Assign the material this stencil evaluates.

        Parameters
        ----------
        material
            A material with a single yield surface, honoring
            :mod:`edelweissfe.materials.base.basegradientplasticityhypoelasticmaterial`.
            The instance must belong to this stencil alone, see
            :mod:`edelweissfd.materials.provider`.
        """

        if material.nYieldSurfaces != 1:
            raise ValueError("This stencil supports materials with a single yield surface only.")

        self._material = material
        self._materialRoutine = getattr(material, self._materialRoutineName)

    def initializeStencil(self):
        """Create the state variables and let the material initialize them."""

        if self._nodes is None:
            raise ValueError("The grid points of this stencil have not been assigned yet.")

        if not self.hasMaterial:
            raise ValueError("No material has been assigned to this stencil.")

        # stress, strain, the plastic multiplier and the yield function value
        self._nStateVarsOverhead = 2 * nVoigtComponents + 2

        nStateVarsPerPoint = self._nStateVarsOverhead + self._material.getNumberOfRequiredStateVars()

        self._stateVars = np.zeros((self._nMaterialPoints, nStateVarsPerPoint))
        self._stateVarsTemp = np.zeros((self._nMaterialPoints, nStateVarsPerPoint))

        self._stresses = [self._stateVarsTemp[p, 0:nVoigtComponents] for p in range(self._nMaterialPoints)]
        self._strains = [
            self._stateVarsTemp[p, nVoigtComponents : 2 * nVoigtComponents] for p in range(self._nMaterialPoints)
        ]
        self._multipliers = [
            self._stateVarsTemp[p, 2 * nVoigtComponents : 2 * nVoigtComponents + 1]
            for p in range(self._nMaterialPoints)
        ]
        self._yieldFunctions = [
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
        """The characteristic length of a material point."""

        subdivision = 1 if self._nDim == 1 else 2

        return float(np.prod(self._spacings / subdivision) ** (1.0 / self._nDim))

    @property
    def laplacianCoefficients(self) -> np.ndarray:
        """The Laplacian coefficients per material point over the molecule."""

        return self._laplacianCoefficients

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

        displacementIncrements = dU[self._displacementDofs]

        multipliers = U[self._multiplierDofs]
        multiplierIncrements = dU[self._multiplierDofs]

        nDisplacementDofs = self._displacementDofs.size
        nMultiplierDofs = self._multiplierDofs.size

        PDisplacement = np.zeros(nDisplacementDofs)
        PMultiplier = np.zeros(nMultiplierDofs)

        KUU = np.zeros((nDisplacementDofs, nDisplacementDofs))
        KUL = np.zeros((nDisplacementDofs, nMultiplierDofs))
        KLU = np.zeros((nMultiplierDofs, nDisplacementDofs))
        KLL = np.zeros((nMultiplierDofs, nMultiplierDofs))

        for p in range(self._nMaterialPoints):
            B = self._strainOperators[p]
            laplacian = self._laplacianCoefficients[p]
            volume = self._materialPointVolumes[p]

            # the material point is collocated on a grid point, so its plastic multiplier is
            # that grid point's value rather than an interpolation
            centre = self._materialPointNode(p)

            increment.dStrain[:] = B @ displacementIncrements
            increment.dLambda[0] = multiplierIncrements[centre]
            increment.laplaceDLambda[0] = laplacian @ multiplierIncrements

            # the stress enters the rate form material as the stress of the last increment
            response.stress[:] = self._stresses[p]
            response.f[:] = 0.0
            tangents.zero()

            self._material.assignCurrentStateVars(self._materialStateVars[p])

            self._materialRoutine(response, tangents, increment, time, dT)

            self._strains[p] += increment.dStrain
            self._stresses[p][:] = response.stress
            self._multipliers[p][0] = multipliers[centre]
            self._yieldFunctions[p][:] = response.f

            PDisplacement += B.T @ response.stress * volume
            PMultiplier[centre] += response.f[0] * volume

            # the plastic multiplier enters the stress both directly and through its Laplacian
            dStress_dMultipliers = np.zeros((nVoigtComponents, nMultiplierDofs))
            dStress_dMultipliers += np.outer(tangents.dStress_dLaplacian[:, 0], laplacian)
            dStress_dMultipliers[:, centre] += tangents.dStress_dLambda[:, 0]

            dYield_dMultipliers = tangents.dF_dLaplacian[0, 0] * laplacian
            dYield_dMultipliers[centre] += tangents.dF_dLambda[0, 0]

            KUU += B.T @ tangents.dStress_dStrain @ B * volume
            KUL += B.T @ dStress_dMultipliers * volume

            KLU[centre, :] += tangents.dF_dStrain[0, :] @ B * volume
            KLL[centre, :] += dYield_dMultipliers * volume

        P[self._displacementDofs] += PDisplacement
        P[self._multiplierDofs] += PMultiplier

        K[self._displacementBlock] += KUU
        K[self._displacementMultiplierBlock] += KUL
        K[self._multiplierDisplacementBlock] += KLU
        K[self._multiplierBlock] += KLL

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
        if result == "plastic multiplier":
            return np.array(self._multipliers[quadraturePoint], copy=not getPersistentView)
        if result == "yield function":
            return np.array(self._yieldFunctions[quadraturePoint], copy=not getPersistentView)

        self._material.assignCurrentStateVars(self._materialStateVars[quadraturePoint])

        return np.array(self._material.getResult(result), copy=not getPersistentView)

    def getNumberOfQuadraturePoints(self) -> int:
        return self._nMaterialPoints

    def getCoordinatesAtCenter(self) -> np.ndarray:
        return self._centerCoordinates
