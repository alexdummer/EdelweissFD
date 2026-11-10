#!/usr/bin/env python3
"""The compiled cell kernels must agree with the Python ones exactly.

A stencil delegates its inner loop to a compiled kernel when that is available and
falls back to the Python implementation otherwise, so both paths are live and both have to give
the same answer. The bound is round-off rather than an accuracy tolerance: the two evaluate the
same expressions over the same material and differ only in the order the sums are accumulated --
explicit C loops against numpy's BLAS calls -- so they agree to about 1e-13 relative and anything
looser than that would be a real discrepancy.
"""

import sys

import numpy as np
import pytest

pytestmark = pytest.mark.marmot


#: Relative bound for two evaluations that differ only in summation order.
roundOff = 1e-11


def assertClose(computed, reference, message=""):
    """Assert agreement up to round-off, scaled by the magnitude of the reference."""

    scale = max(1.0, float(np.abs(reference).max()))

    assert np.allclose(computed, reference, rtol=roundOff, atol=roundOff * scale), message


def evaluate(stencil, U, dU) -> tuple:
    """One kernel evaluation of a stencil, returning its tangent and flux."""

    K = np.zeros((stencil.nDof, stencil.nDof), order="F")
    P = np.zeros(stencil.nDof)

    stencil.computeKernels(K, P, U, dU, 1.0, 1.0)

    return K, P


def buildStencilBothPaths(stencilClass, spacings, makeMaterial, **options) -> tuple:
    """One stencil on one cell, built once with the compiled kernel and once without.

    Parameters
    ----------
    stencilClass
        The stencil class.
    spacings
        The grid spacing per direction.
    makeMaterial
        A callable returning a *fresh* material, since no two stencils may share one.
    options
        Further keyword arguments for the stencil.
    """

    from tests.test_tangents import buildStencil

    module = sys.modules[stencilClass.__module__]

    built = {}

    for compiled in (True, False):
        module.useCompiledKernel = compiled
        try:
            built[compiled] = buildStencil(stencilClass, spacings, makeMaterial(), **options)
        finally:
            module.useCompiledKernel = True

    assert built[False]._kernel is None

    return built[True], built[False]


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
def test_theDisplacementKernelAgrees(spacings, stressState, linearElasticProperties):
    """Every stress state, since each goes to a different Marmot routine with a differently
    reduced tangent: the full six by six, the three by three plane stress one and the scalar
    uniaxial one."""

    from edelweissfe.materials.marmot.marmothypoelastic import MarmotHypoElasticMaterial

    from edelweissfd.stencils.displacementstencil import DisplacementStencil

    if (
        DisplacementStencil.__module__
        and getattr(sys.modules[DisplacementStencil.__module__], "DisplacementKernel", None) is None
    ):
        pytest.skip("EdelweissFD was built without the compiled displacement kernel")

    def makeMaterial():
        return MarmotHypoElasticMaterial("LINEARELASTIC", linearElasticProperties)

    compiled, interpreted = buildStencilBothPaths(DisplacementStencil, spacings, makeMaterial, stressState=stressState)

    rng = np.random.default_rng(5)
    dU = rng.normal(scale=1e-3, size=compiled.nDof)

    KCompiled, PCompiled = evaluate(compiled, dU, dU)
    KInterpreted, PInterpreted = evaluate(interpreted, dU, dU)

    assertClose(KCompiled, KInterpreted, "tangents disagree for " + stressState)
    assertClose(PCompiled, PInterpreted, "fluxes disagree for " + stressState)
    assertClose(compiled._stateVarsTemp, interpreted._stateVarsTemp, "state variables disagree")


def test_theDisplacementKernelAcceptsAMixedCaseMaterialName(linearElasticProperties):
    """EdelweissFE's material accepts a name Marmot only registers upper-case, and keeps it as
    given; the compiled kernel has to accept every material the Python path accepts."""

    from edelweissfe.materials.marmot.marmothypoelastic import MarmotHypoElasticMaterial

    from edelweissfd.stencils.displacementstencil import DisplacementStencil

    if getattr(sys.modules[DisplacementStencil.__module__], "DisplacementKernel", None) is None:
        pytest.skip("EdelweissFD was built without the compiled displacement kernel")

    def makeMaterial():
        return MarmotHypoElasticMaterial("LinearElastic", linearElasticProperties)

    compiled, interpreted = buildStencilBothPaths(
        DisplacementStencil, [1.0, 2.0], makeMaterial, stressState="plane strain"
    )

    assert compiled._kernel is not None

    dU = np.random.default_rng(3).normal(scale=1e-3, size=compiled.nDof)

    KCompiled, PCompiled = evaluate(compiled, dU, dU)
    KInterpreted, PInterpreted = evaluate(interpreted, dU, dU)

    assertClose(KCompiled, KInterpreted, "tangents disagree")
    assertClose(PCompiled, PInterpreted, "fluxes disagree")


def test_theDisplacementKernelAgreesForAPlasticMaterial(vonMisesProperties):
    """A yielding material, so the state variable round trip through the compiled kernel is
    exercised rather than only the elastic tangent."""

    from edelweissfe.materials.marmot.marmothypoelastic import MarmotHypoElasticMaterial

    from edelweissfd.stencils.displacementstencil import DisplacementStencil

    if getattr(sys.modules[DisplacementStencil.__module__], "DisplacementKernel", None) is None:
        pytest.skip("EdelweissFD was built without the compiled displacement kernel")

    def makeMaterial():
        return MarmotHypoElasticMaterial("VONMISES", np.asarray(vonMisesProperties, dtype=float))

    compiled, interpreted = buildStencilBothPaths(
        DisplacementStencil, [1.0, 1.0], makeMaterial, stressState="plane strain"
    )

    rng = np.random.default_rng(11)
    dU = rng.normal(scale=2e-2, size=compiled.nDof)

    KCompiled, PCompiled = evaluate(compiled, dU, dU)
    KInterpreted, PInterpreted = evaluate(interpreted, dU, dU)

    assertClose(KCompiled, KInterpreted, "tangents disagree")
    assertClose(PCompiled, PInterpreted, "fluxes disagree")

    # the material really yielded, so this is not an elastic comparison in disguise
    equivalentPlasticStrain = compiled.getResultArray("kappa", 0)

    assert equivalentPlasticStrain[0] > 0.0


@pytest.mark.parametrize("spacings, stressState", [([1.0, 2.0], "plane strain"), ([1.0, 2.0, 0.5], "3d")])
def test_theGradientEnhancedKernelAgrees(spacings, stressState, at2PhaseFieldProperties):
    """The AT2 phase field couples both fields in every tangent block, so this exercises the
    stress, the local driving variable, the nonlocal parameter and the gradient term at once."""

    from edelweissfe.materials.marmot.marmotgradientenhancedhypoelastic import (
        MarmotGradientEnhancedHypoElasticMaterial,
    )

    from edelweissfd.stencils.gradientenhanceddisplacementstencil import (
        GradientEnhancedDisplacementStencil,
    )

    if (
        getattr(sys.modules[GradientEnhancedDisplacementStencil.__module__], "GradientEnhancedDisplacementKernel", None)
        is None
    ):
        pytest.skip("EdelweissFD was built without the compiled gradient enhanced kernel")

    def makeMaterial():
        return MarmotGradientEnhancedHypoElasticMaterial("AT2PHASEFIELD", at2PhaseFieldProperties)

    compiled, interpreted = buildStencilBothPaths(
        GradientEnhancedDisplacementStencil, spacings, makeMaterial, stressState=stressState
    )

    rng = np.random.default_rng(13)

    U = np.zeros(compiled.nDof)
    U[compiled._displacementDofs] = rng.normal(scale=1e-3, size=compiled._displacementDofs.size)
    U[compiled._nonlocalDofs] = np.abs(rng.normal(scale=0.2, size=compiled._nonlocalDofs.size))

    dU = 0.5 * U

    KCompiled, PCompiled = evaluate(compiled, U, dU)
    KInterpreted, PInterpreted = evaluate(interpreted, U, dU)

    assertClose(KCompiled, KInterpreted, "tangents disagree for " + stressState)
    assertClose(PCompiled, PInterpreted, "fluxes disagree for " + stressState)
    assertClose(compiled._stateVarsTemp, interpreted._stateVarsTemp, "state variables disagree")
