#!/usr/bin/env python3
"""The analytic tangents of the stencils must agree with a numerical differentiation.

This is the test that keeps the hand derived tangents honest. A wrong tangent does not
show up as a wrong result -- the residual still governs the solution -- but only as slow or
absent Newton convergence, which is hard to attribute. Comparing against a central
difference of the internal flux catches it immediately.
"""

import numpy as np
import pytest
from edelweissfe.points.node import Node

from edelweissfd.operators.differences import cellCornerOffsets
from edelweissfd.stencils.displacementstencil import DisplacementStencil
from edelweissfd.stencils.gradientenhanceddisplacementstencil import (
    GradientEnhancedDisplacementStencil,
)


def createCellNodes(spacings) -> list:
    """The corner grid points of a single cell.

    Parameters
    ----------
    spacings
        The grid spacing per direction.

    Returns
    -------
    list
        The corner grid points.
    """

    spacings = np.asarray(spacings, dtype=float)
    corners = cellCornerOffsets(spacings.size)

    return [Node(i + 1, corner * spacings) for i, corner in enumerate(corners)]


def buildStencil(stencilClass, spacings, material, **options):
    """Create a single stencil on one cell, ready for evaluation."""

    stencil = stencilClass(1, spacings, **options)
    stencil.setNodes(createCellNodes(spacings))
    stencil.setMaterial(material)
    stencil.initializeStencil()

    return stencil


@pytest.mark.marmot
@pytest.mark.parametrize(
    "spacings, stressState",
    [
        ([1.0], "uniaxial stress"),
        ([1.0], "3d"),
        ([1.0, 2.0], "plane strain"),
        ([1.0, 2.0], "plane stress"),
        ([1.0, 2.0, 0.5], "3d"),
    ],
)
def test_displacementStencilTangentIsConsistent(spacings, stressState, linearElasticProperties):
    """Linear elasticity: the tangent must match to round off."""

    from edelweissfe.materials.marmot.marmothypoelastic import MarmotHypoElasticMaterial

    material = MarmotHypoElasticMaterial("LINEARELASTIC", linearElasticProperties)

    stencil = buildStencil(DisplacementStencil, spacings, material, stressState=stressState)

    rng = np.random.default_rng(3)
    dU = rng.normal(scale=1e-3, size=stencil.nDof)

    stencil.assertTangentConsistent(dU, dU, 1.0, 1.0, relativeTolerance=1e-6)


@pytest.mark.marmot
@pytest.mark.parametrize(
    "spacings, stressState",
    [
        ([1.0], "3d"),
        ([1.0, 2.0], "plane strain"),
        ([1.0, 2.0, 0.5], "3d"),
    ],
)
@pytest.mark.parametrize("nonlocalMagnitude", [0.0, 0.15])
def test_gradientEnhancedStencilTangentIsConsistent(spacings, stressState, nonlocalMagnitude, at2PhaseFieldProperties):
    """The AT2 phase field couples both fields in every tangent block, so this exercises
    dStress_dStrain, dStress_dK, dKLocal_dStrain, dKLocal_dK and the gradient term."""

    from edelweissfe.materials.marmot.marmotgradientenhancedhypoelastic import (
        MarmotGradientEnhancedHypoElasticMaterial,
    )

    material = MarmotGradientEnhancedHypoElasticMaterial("AT2PHASEFIELD", at2PhaseFieldProperties)

    stencil = buildStencil(GradientEnhancedDisplacementStencil, spacings, material, stressState=stressState)

    rng = np.random.default_rng(11)

    dU = np.zeros(stencil.nDof)
    dU[stencil._displacementDofs] = rng.normal(scale=2e-3, size=stencil._displacementDofs.size)
    dU[stencil._nonlocalDofs] = nonlocalMagnitude + rng.normal(scale=1e-2, size=stencil._nonlocalDofs.size)

    stencil.assertTangentConsistent(dU, dU, 1.0, 1.0, perturbation=1e-8, relativeTolerance=1e-5)


@pytest.mark.marmot
def test_gradientEnhancedStencilTangentAfterDamageHistory(at2PhaseFieldProperties):
    """After a loading step the crack driving force history variable is active, which
    switches on the dKLocal_dStrain block; the tangent must still be consistent."""

    from edelweissfe.materials.marmot.marmotgradientenhancedhypoelastic import (
        MarmotGradientEnhancedHypoElasticMaterial,
    )

    material = MarmotGradientEnhancedHypoElasticMaterial("AT2PHASEFIELD", at2PhaseFieldProperties)

    spacings = [1.0, 1.0]
    stencil = buildStencil(GradientEnhancedDisplacementStencil, spacings, material, stressState="plane strain")

    # load once and accept, so that the history variable is non-zero
    U = np.zeros(stencil.nDof)
    U[stencil._displacementDofs] = 5e-3
    U[stencil._nonlocalDofs] = 0.05

    K = np.zeros((stencil.nDof, stencil.nDof))
    P = np.zeros(stencil.nDof)
    stencil.computeKernels(K, P, U, U, 1.0, 1.0)
    stencil.acceptLastState()

    dU = np.full(stencil.nDof, 1e-4)

    stencil.assertTangentConsistent(U + dU, dU, 2.0, 1.0, perturbation=1e-8, relativeTolerance=1e-5)


@pytest.mark.marmot
def test_numericalTangentLeavesStencilUnperturbed(linearElasticProperties):
    """Computing the numerical tangent must not leave the stencil in a perturbed state."""

    from edelweissfe.materials.marmot.marmothypoelastic import MarmotHypoElasticMaterial

    material = MarmotHypoElasticMaterial("LINEARELASTIC", linearElasticProperties)

    stencil = buildStencil(DisplacementStencil, [1.0, 1.0], material, stressState="plane strain")

    dU = np.full(stencil.nDof, 1e-3)

    K = np.zeros((stencil.nDof, stencil.nDof))
    P = np.zeros(stencil.nDof)
    stencil.computeKernels(K, P, dU, dU, 1.0, 1.0)

    stressBefore = stencil.getResultArray("stress", 0, getPersistentView=False)

    stencil.computeNumericalTangent(dU, dU, 1.0, 1.0)

    stressAfter = stencil.getResultArray("stress", 0, getPersistentView=False)

    assert np.allclose(stressBefore, stressAfter)
