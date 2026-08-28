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

import os
from time import perf_counter_ns

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
    condensePlaneStressTangents,
    hourglassVector,
    nVoigtComponents,
    volumetricallyAveragedStrainOperators,
)
from edelweissfd.stencils.base.basestencil import BaseStencil
from edelweissfd.stencils.numericaltangent import NumericalTangentMixin

try:
    from edelweissfe.kernels.gradientplasticitykernel import GradientPlasticityKernel
except ImportError:  # pragma: no cover - depends on whether EdelweissFE built its extensions
    GradientPlasticityKernel = None

#: Use the compiled kernel of EdelweissFE when it is available.
#:
#: The cell couples few degrees of freedom and its blocks are small -- a six by six tangent,
#: operators of six by twelve -- so at those sizes a numpy call costs more in dispatch than in
#: arithmetic. Measured on a real localising run, the Marmot material is only 18.4 percent of the
#: kernel time and the rest is that overhead, which is what the compiled kernel removes.
#:
#: Set to False, or set ``EDELWEISSFD_NO_COMPILED_KERNEL=1`` in the environment, to force the
#: Python implementation. The two are tested against each other in ``tests/test_compiledkernel.py``
#: and the whole suite is run both ways, so neither path is allowed to rot.
useCompiledKernel = os.environ.get("EDELWEISSFD_NO_COMPILED_KERNEL", "") not in ("1", "true", "True")

#: Accumulate how much of the kernel time is spent inside the material, per stencil.
#:
#: Off by default. The kernel is a few tens of microseconds and two clock reads cost about a
#: tenth of one, so the overhead is small but there is no reason to pay it in production. It is
#: worth having permanently available because the share decides whether compiling the kernel is
#: worth anything at all: the attainable speedup is bounded by one over the material share, and
#: that share is far higher in a real localising run than in a synthetic probe.
profileMaterialTime = False


def materialTimeReport(stencils) -> dict:
    """Sum the per stencil material and kernel timings collected under
    :data:`profileMaterialTime`.

    Accumulated per stencil rather than globally on purpose: each stencil is evaluated by one
    thread only, so per stencil counters need no synchronisation, while a shared accumulator would
    lose updates on a free threaded interpreter.

    Parameters
    ----------
    stencils
        The stencils to sum over.

    Returns
    -------
    dict
        The material and kernel times in seconds, the number of material evaluations, the
        microseconds per evaluation, the material share of the kernel time and the speedup the
        kernel could reach at most if all of the remaining time were removed.
    """

    materialSeconds = sum(s._materialTimeNs for s in stencils) * 1e-9
    kernelSeconds = sum(s._kernelTimeNs for s in stencils) * 1e-9
    calls = sum(s._materialCalls for s in stencils)

    share = materialSeconds / kernelSeconds if kernelSeconds > 0.0 else float("nan")

    return dict(
        materialSeconds=materialSeconds,
        kernelSeconds=kernelSeconds,
        calls=calls,
        microsecondsPerCall=1e6 * materialSeconds / calls if calls else float("nan"),
        materialShare=share,
        ceiling=1.0 / share if share > 0.0 else float("nan"),
    )


#: The supported stress states, mapping to the material routine.
stressStates = {
    "3d": "computeStress",
    "plane strain": "computeStress",
    "plane stress": "computePlaneStress",
}


def _condensePlaneStressTangents(stacks: dict):
    """Apply :func:`~edelweissfd.operators.differences.condensePlaneStressTangents` to every
    material point of the cell at once, in place -- see there for why this is needed at all."""

    Dcond, dLambdaCond, dLapCond, dFdEcond, dFdLambdaCond, dFdLapCond = condensePlaneStressTangents(
        stacks["dStress_dStrain"],
        stacks["dStress_dLambda"][:, :, 0],
        stacks["dStress_dLaplacian"][:, :, 0],
        stacks["dF_dStrain"][:, 0, :],
        stacks["dF_dLambda"][:, 0, 0],
        stacks["dF_dLaplacian"][:, 0, 0],
    )

    stacks["dStress_dStrain"][...] = Dcond
    stacks["dStress_dLambda"][:, :, 0] = dLambdaCond
    stacks["dStress_dLaplacian"][:, :, 0] = dLapCond
    stacks["dF_dStrain"][:, 0, :] = dFdEcond
    stacks["dF_dLambda"][:, 0, 0] = dFdLambdaCond
    stacks["dF_dLaplacian"][:, 0, 0] = dFdLapCond


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
        reproduce that. Ignored when ``hourglassControl`` is ``"stabilized"``, which does not lock
        in the first place.
    hourglassControl
        ``"corner"`` (default) samples the strain at every corner of the cell with a one-sided
        quotient, see :func:`~edelweissfd.operators.differences.cellCornerGradientOperators`.
        That resists the hourglass mode, but is only first order accurate at the corner itself --
        confirmed to be the accuracy bottleneck of the whole coupled solve, since the plastic
        multiplier's Laplacian is a compact, essentially exact second order stencil. ``"stabilized"``
        instead samples the strain once, at the cell centre, with :func:`cellGradientOperator`
        (second order, and does not lock volumetrically to begin with), and adds a Flanagan-Belytschko
        orthogonal hourglass stabilization stiffness, see
        :func:`~edelweissfd.operators.differences.hourglassVector`, to control the one zero energy
        mode single point sampling leaves open. Only implemented in two dimensions. Experimental:
        validated on small linear elastic patches (no near-zero eigenvalues under the panel's own,
        weakly constrained boundary conditions, and an :math:`O(h^2)` stabilization response on a
        smooth field), not yet exercised at the scale of the compiled kernel, which is bypassed
        whenever this is not ``"corner"``.
    hourglassStiffness
        A representative elastic modulus the stabilization stiffness is scaled from, required when
        ``hourglassControl`` is ``"stabilized"``. Fixed at construction, deliberately not the
        current (possibly heavily softened) material tangent -- scaling down with the tangent would
        withdraw stabilization exactly where localisation makes it most needed.
    hourglassCoefficient
        The dimensionless factor :math:`a_{hg}` scaling the stabilization stiffness,
        :math:`c = a_{hg} \\, G \\, V / L^2` with :math:`G` the ``hourglassStiffness``, :math:`V`
        the cell volume and :math:`L` the :attr:`characteristicLength`. The classical range for
        artificial stiffness hourglass control is 0.03 to 0.1.
    """

    def __init__(
        self,
        stencilNumber: int,
        spacings,
        stressState: str = "plane strain",
        thickness: float = 1.0,
        volumetricAveraging: bool = True,
        hourglassControl: str = "corner",
        hourglassStiffness: float = None,
        hourglassCoefficient: float = 0.05,
    ):
        if stressState not in stressStates:
            raise ValueError(
                "Unknown stress state '{:}'; available are {:}".format(stressState, ", ".join(stressStates))
            )

        if hourglassControl not in ("corner", "stabilized"):
            raise ValueError("Unknown hourglassControl '{:}'; available are 'corner', 'stabilized'.")

        self._stencilNumber = stencilNumber
        self._stressState = stressState

        self._spacings = np.asarray(spacings, dtype=float)
        self._nDim = self._spacings.size

        if self._nDim == 1 and stressState in ("plane strain", "plane stress"):
            raise ValueError("A one dimensional grid needs the stress state '3d'.")

        if hourglassControl == "stabilized" and self._nDim != 2:
            raise ValueError("hourglassControl='stabilized' is only implemented in two dimensions.")

        self._hourglassControl = hourglassControl

        self._cornerOffsets = cellCornerOffsets(self._nDim)
        self._nCorners = self._cornerOffsets.shape[0]

        cellVolume = float(np.prod(self._spacings))
        if self._nDim < 3:
            cellVolume *= thickness

        self._totalVolume = cellVolume

        if hourglassControl == "stabilized":
            # single point quadrature at the cell centre, repeated onto every corner so the
            # material points still sit one per corner -- the yield condition still needs a grid
            # point to be collocated on -- but all of them see the identical, second order
            # accurate strain, sampled only once.
            self._gradients = [cellGradientOperator(self._spacings)] * self._nCorners
        elif self._nDim == 1:
            self._gradients = [cellGradientOperator(self._spacings)]
        else:
            self._gradients = cellCornerGradientOperators(self._spacings)

        self._nMaterialPoints = len(self._gradients)
        self._materialPointVolumes = [cellVolume / self._nMaterialPoints] * self._nMaterialPoints

        self._strainOperators = [cellStrainOperator(gradient) for gradient in self._gradients]

        # Without this the cell locks volumetrically as Poisson's ratio approaches one half, and a
        # shear band -- which deforms at constant volume -- cannot form at all. On by default,
        # because a gradient plasticity computation is localisation at nearly incompressible
        # plastic flow, i.e. exactly the case that locks. Single point quadrature does not lock in
        # the first place, so there is nothing to avearge there.
        self._volumetricAveraging = volumetricAveraging and hourglassControl == "corner"

        if self._volumetricAveraging:
            self._strainOperators = volumetricallyAveragedStrainOperators(
                self._strainOperators, self._materialPointVolumes
            )

        self._hourglassStiffnessMatrix = None
        if hourglassControl == "stabilized":
            if hourglassStiffness is None:
                raise ValueError("hourglassControl='stabilized' needs hourglassStiffness.")

            gamma = hourglassVector(self._spacings)
            stiffnessScale = hourglassCoefficient * hourglassStiffness * cellVolume / self.characteristicLength**2

            nDisplacementDofs = self._nCorners * self._nDim
            Khg = np.zeros((nDisplacementDofs, nDisplacementDofs))
            for component in range(self._nDim):
                index = np.arange(component, nDisplacementDofs, self._nDim)
                Khg[np.ix_(index, index)] += stiffnessScale * np.outer(gamma, gamma)

            self._hourglassStiffnessMatrix = Khg

        self._materialRoutineName = stressStates[stressState]

        self._response = GradientPlasticityResponse.createZero(1)
        self._tangents = GradientPlasticityTangents.createZero(1)
        self._increment = GradientPlasticityIncrement.createZero(1)

        self._material = None
        self._nodes = None
        self._kernel = None

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

        # the same storage seen as one block over all material points, so that the bookkeeping
        # around the material evaluations is a handful of array operations rather than one per point
        self._previousStresses = self._stateVarsTemp[:, 0:nVoigtComponents]
        self._strainStates = self._stateVarsTemp[:, nVoigtComponents : 2 * nVoigtComponents]
        self._multiplierStates = self._stateVarsTemp[:, 2 * nVoigtComponents : 2 * nVoigtComponents + 1]
        self._yieldStates = self._stateVarsTemp[:, 2 * nVoigtComponents + 1 : 2 * nVoigtComponents + 2]

        for p in range(self._nMaterialPoints):
            self._material.assignCurrentStateVars(self._materialStateVars[p])

            if hasattr(self._material, "initializeYourself"):
                self._material.initializeYourself()

        self.acceptLastState()

        self._allocateKernelBuffers()

        self._kernel = self._createCompiledKernel()

    def _createCompiledKernel(self):
        """The compiled kernel for this cell, or ``None`` to fall back to Python.

        Returns ``None`` when EdelweissFE was installed without its Marmot extensions, when the
        compiled kernel has been switched off, when the material is not a Marmot one -- the
        kernel reaches Marmot's C++ shim directly, so it cannot drive a material that only exists
        in Python --, when hourglass stabilization is in use, which the compiled kernel does not
        yet know about, or when the stress state is plane stress: Marmot's ``computePlaneStress``
        returns an uncondensed tangent (verified directly against central differences of the
        material response -- see :func:`_condensePlaneStressTangents`), and the compiled kernel
        has no equivalent correction, so plane stress always falls back to the Python path, which
        does.
        """

        if GradientPlasticityKernel is None or not useCompiledKernel:
            return None

        if self._hourglassStiffnessMatrix is not None:
            return None

        if self._materialRoutineName == "computePlaneStress":
            return None

        if not hasattr(self._material, "materialName") or not hasattr(self._material, "materialProperties"):
            return None

        return GradientPlasticityKernel(
            self._material.materialName,
            np.ascontiguousarray(self._material.materialProperties, dtype=float),
            self._materialRoutineName == "computePlaneStress",
            self._strainOperatorStack,
            self._weightedTransposedStack,
            np.ascontiguousarray(self._laplacianCoefficients, dtype=float),
            np.ascontiguousarray(self._materialPointVolumeArray, dtype=float),
            np.ascontiguousarray(self._materialPointNodes, dtype=np.int64),
            np.ascontiguousarray(self._displacementDofs, dtype=np.int64),
            np.ascontiguousarray(self._multiplierDofs, dtype=np.int64),
            self._stateVars,
            self._stateVarsTemp,
            self._nStateVarsOverhead,
        )

    def _allocateKernelBuffers(self):
        """Allocate everything :meth:`computeKernels` needs, once, and stacked over material points.

        The blocks involved are tiny -- a six by six tangent, operators of six by twelve -- so a
        numpy call costs far more in dispatch than in arithmetic. Measured on a two dimensional
        cell, the Marmot material evaluation is 4.2 of the 37 microseconds spent per material point
        even when it is yielding; the remaining 89 percent is the overhead of some twenty small
        numpy operations per point. Making each operation faster cannot help much, so the algebra is
        instead done for all material points of the cell at once, in a handful of batched calls.

        For that to cost nothing in copying, the material writes its results *directly* into the
        stacked storage: the response and tangent objects handed to material point ``p`` have their
        entries as views into row ``p`` of arrays shaped ``(nMaterialPoints, ...)``. A row of a C
        contiguous array is itself C contiguous, which is what the Cython materials require.
        """

        nMaterialPoints = self._nMaterialPoints
        nDisplacementDofs = self._displacementDofs.size
        nMultiplierDofs = self._multiplierDofs.size

        self._PDisplacement = np.zeros(nDisplacementDofs)
        self._PMultiplier = np.zeros(nMultiplierDofs)

        self._KUU = np.zeros((nDisplacementDofs, nDisplacementDofs))
        self._KUL = np.zeros((nDisplacementDofs, nMultiplierDofs))
        self._KLU = np.zeros((nMultiplierDofs, nDisplacementDofs))
        self._KLL = np.zeros((nMultiplierDofs, nMultiplierDofs))

        # the corner each material point sits on, and its volume
        self._materialPointNodes = np.array([self._materialPointNode(p) for p in range(nMaterialPoints)], dtype=int)
        volumes = np.asarray(self._materialPointVolumes, dtype=float)
        self._materialPointVolumeArray = volumes

        # the strain operators stacked, and their transposes already scaled by the material point
        # volume, since every place the transpose appears it is multiplied by that volume
        self._strainOperatorStack = np.ascontiguousarray(np.stack(self._strainOperators))
        self._weightedTransposedStack = np.ascontiguousarray(
            np.stack([B.T * volume for B, volume in zip(self._strainOperators, volumes)])
        )
        self._weightedLaplacianStack = np.ascontiguousarray(self._laplacianCoefficients * volumes[:, None])

        # the stacked increment, response and tangent storage the materials write into
        self._dStrainStack = np.zeros((nMaterialPoints, nVoigtComponents))
        self._dLambdaStack = np.zeros((nMaterialPoints, 1))
        self._laplaceDLambdaStack = np.zeros((nMaterialPoints, 1))

        # The stress is both an input and an output of a rate form material, and the yield value is
        # an output, so the response points straight at the state variable slots that hold them.
        # That removes the copies in and out entirely; the reset at the top of computeKernels
        # restores the last converged stress, which is exactly what the material expects to read.
        self._stressStack = self._previousStresses
        self._yieldStack = self._yieldStates

        self._tangentStacks = dict(
            dStress_dStrain=np.zeros((nMaterialPoints, nVoigtComponents, nVoigtComponents)),
            dStress_dLambda=np.zeros((nMaterialPoints, nVoigtComponents, 1)),
            dStress_dLaplacian=np.zeros((nMaterialPoints, nVoigtComponents, 1)),
            dF_dStrain=np.zeros((nMaterialPoints, 1, nVoigtComponents)),
            dF_dLambda=np.zeros((nMaterialPoints, 1, 1)),
            dF_dLaplacian=np.zeros((nMaterialPoints, 1, 1)),
        )

        self._increments = [
            GradientPlasticityIncrement(
                dStrain=self._dStrainStack[p],
                dLambda=self._dLambdaStack[p],
                laplaceDLambda=self._laplaceDLambdaStack[p],
            )
            for p in range(nMaterialPoints)
        ]
        self._responses = [
            GradientPlasticityResponse(stress=self._stressStack[p], f=self._yieldStack[p])
            for p in range(nMaterialPoints)
        ]
        self._tangentsPerPoint = [
            GradientPlasticityTangents(**{name: stack[p] for name, stack in self._tangentStacks.items()})
            for p in range(nMaterialPoints)
        ]

        # scratch for the batched products
        self._weightedTimesTangent = np.zeros((nMaterialPoints, nDisplacementDofs, nVoigtComponents))
        self._displacementBlocks = np.zeros((nMaterialPoints, nDisplacementDofs, nDisplacementDofs))
        self._stressCouplings = np.zeros((nMaterialPoints, nDisplacementDofs, 1))
        self._laplacianCouplings = np.zeros((nMaterialPoints, nDisplacementDofs, 1))
        self._yieldRows = np.zeros((nMaterialPoints, 1, nDisplacementDofs))

        # The four blocks live as views in the quadrants of one local matrix, in field grouped
        # order, so that the whole cell contribution reaches the global matrix in a single indexed
        # assignment. Four separate ones cost 12 microseconds against 4.3 for this one, because
        # indexed assignment on the global array is expensive per call and nearly free per entry.
        nDof = self._nDof
        nD = nDisplacementDofs

        self._localTangent = np.zeros((nDof, nDof))
        self._localFlux = np.zeros(nDof)

        self._localKUU = self._localTangent[:nD, :nD]
        self._localKUL = self._localTangent[:nD, nD:]
        self._localKLU = self._localTangent[nD:, :nD]
        self._localKLL = self._localTangent[nD:, nD:]

        self._localPDisplacement = self._localFlux[:nD]
        self._localPMultiplier = self._localFlux[nD:]

        self._localOrder = np.concatenate([self._displacementDofs, self._multiplierDofs])
        self._localBlock = np.ix_(self._localOrder, self._localOrder)

        self._materialTimeNs = 0
        self._kernelTimeNs = 0
        self._materialCalls = 0

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
        profiling = profileMaterialTime

        if profiling:
            kernelStart = perf_counter_ns()

        if self._kernel is not None:
            self._kernel.computeKernels(K, P, U, dU, time, dT)

            if profiling:
                self._kernelTimeNs += perf_counter_ns() - kernelStart

            return

        self._stateVarsTemp[:, :] = self._stateVars

        displacementIncrements = dU[self._displacementDofs]

        multipliers = U[self._multiplierDofs]
        multiplierIncrements = dU[self._multiplierDofs]

        matmul = np.matmul
        einsum = np.einsum

        centres = self._materialPointNodes
        volumes = self._materialPointVolumeArray

        B = self._strainOperatorStack
        BTv = self._weightedTransposedStack
        laplacians = self._laplacianCoefficients
        weightedLaplacians = self._weightedLaplacianStack

        dStrains = self._dStrainStack
        stresses = self._stressStack
        yieldValues = self._yieldStack

        stacks = self._tangentStacks

        # -- the increment, for every material point at once ------------------------------
        matmul(B, displacementIncrements, out=dStrains)
        self._dLambdaStack[:, 0] = multiplierIncrements[centres]
        matmul(laplacians, multiplierIncrements, out=self._laplaceDLambdaStack[:, 0])

        # the stress the material reads is the one of the last increment, already in place
        for stack in stacks.values():
            stack.fill(0.0)

        # -- the material evaluations, which are sequential because each owns its state ----
        assignCurrentStateVars = self._material.assignCurrentStateVars
        materialRoutine = self._materialRoutine
        materialStateVars = self._materialStateVars
        responses = self._responses
        tangentsPerPoint = self._tangentsPerPoint
        increments = self._increments

        if profiling:
            materialStart = perf_counter_ns()

        for p in range(self._nMaterialPoints):
            assignCurrentStateVars(materialStateVars[p])
            materialRoutine(responses[p], tangentsPerPoint[p], increments[p], time, dT)

        if profiling:
            self._materialTimeNs += perf_counter_ns() - materialStart
            self._materialCalls += self._nMaterialPoints

        if self._materialRoutineName == "computePlaneStress":
            _condensePlaneStressTangents(stacks)

        self._strainStates += dStrains
        self._multiplierStates[:, 0] = multipliers[centres]

        # -- the assembly, again for every material point at once --------------------------
        self._localTangent.fill(0.0)
        self._localFlux.fill(0.0)

        einsum("pij,pj->i", BTv, stresses, out=self._localPDisplacement)
        self._localPMultiplier[centres] += yieldValues[:, 0] * volumes

        matmul(BTv, stacks["dStress_dStrain"], out=self._weightedTimesTangent)
        matmul(self._weightedTimesTangent, B, out=self._displacementBlocks)
        self._displacementBlocks.sum(axis=0, out=self._localKUU)

        if self._hourglassStiffnessMatrix is not None:
            # a fixed, linear elastic-like spring resisting the hourglass mode, referenced to the
            # total displacement -- tangent-consistent by construction, since it does not depend
            # on the material's (possibly heavily softened) state at all
            Khg = self._hourglassStiffnessMatrix
            self._localKUU += Khg
            self._localPDisplacement += Khg @ U[self._displacementDofs]

        # The plastic multiplier enters the stress through its Laplacian over the whole molecule,
        # and directly at the grid point the material point sits on. Summing the outer products of
        # the two stacked factors is one plain matrix product, which beats a three operand einsum
        # by better than three times.
        matmul(BTv, stacks["dStress_dLaplacian"], out=self._laplacianCouplings)
        matmul(self._laplacianCouplings[:, :, 0].T, laplacians, out=self._localKUL)

        matmul(BTv, stacks["dStress_dLambda"], out=self._stressCouplings)
        self._localKUL[:, centres] += self._stressCouplings[:, :, 0].T

        matmul(stacks["dF_dStrain"], B, out=self._yieldRows)
        self._localKLU[centres, :] += self._yieldRows[:, 0, :] * volumes[:, None]

        self._localKLL[centres, :] += stacks["dF_dLaplacian"][:, 0, 0][:, None] * weightedLaplacians
        self._localKLL[centres, centres] += stacks["dF_dLambda"][:, 0, 0] * volumes

        P[self._localOrder] += self._localFlux
        K[self._localBlock] += self._localTangent

        if profiling:
            self._kernelTimeNs += perf_counter_ns() - kernelStart

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
