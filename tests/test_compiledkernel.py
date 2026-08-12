#!/usr/bin/env python3
"""The compiled gradient plasticity kernel must agree with the Python one exactly.

The stencil delegates its inner loop to EdelweissFE's compiled kernel when that is available and
falls back to the Python implementation otherwise, so both paths are live and both have to give
the same answer. The bound is round-off rather than an accuracy tolerance: the two evaluate the
same expressions over the same material and differ only in the order the sums are accumulated --
explicit C loops against numpy's BLAS calls -- so they agree to about 1e-13 relative and anything
looser than that would be a real discrepancy.
"""

import numpy as np
import pytest

from edelweissfd.stencils import gradientplasticitystencil as gps
from edelweissfd.stencils.gradientplasticitystencil import GradientPlasticityStencil

from tests.test_gradientplasticity import buildSingleCellGrid

pytestmark = pytest.mark.marmot


def kernelAvailable() -> bool:
    return gps.GradientPlasticityKernel is not None


needsKernel = pytest.mark.skipif(not kernelAvailable(), reason="EdelweissFE was built without the compiled kernel")


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


def buildBothPaths(nGridPoints=(5, 5), cellPosition="interior", **properties) -> tuple:
    """The same cell of the same grid, once with the compiled kernel and once without."""

    stencils = {}

    for compiled in (True, False):
        _, _, allStencils = buildSingleCellGrid(nGridPoints=nGridPoints, **properties)

        if cellPosition == "corner":
            stencil = min(allStencils, key=lambda s: s.nNodes)
        else:
            stencil = max(allStencils, key=lambda s: s.nNodes)

        # the flag is read where the kernel is built, so it has to be set around that call
        gps.useCompiledKernel = compiled
        try:
            stencil.initializeStencil()
        finally:
            gps.useCompiledKernel = True

        stencils[compiled] = stencil

    assert (stencils[True]._kernel is not None) == kernelAvailable()
    assert stencils[False]._kernel is None

    return stencils[True], stencils[False]


def randomIncrement(stencil, scale=4e-3, seed=7) -> np.ndarray:
    """A displacement and multiplier increment large enough to yield."""

    rng = np.random.default_rng(seed)

    dU = np.zeros(stencil.nDof)
    dU[stencil._displacementDofs] = rng.normal(scale=scale, size=stencil._displacementDofs.size)
    dU[stencil._multiplierDofs] = np.abs(rng.normal(scale=5e-4, size=stencil._multiplierDofs.size))

    return dU


@needsKernel
@pytest.mark.parametrize("cellPosition", ["interior", "corner"])
def test_theTwoPathsAgreeToMachinePrecision(cellPosition):
    """The whole point. Both the interior molecule and a boundary one, where the ghost node
    mirroring makes the molecule smaller and the index bookkeeping differs."""

    compiled, interpreted = buildBothPaths(cellPosition=cellPosition)

    dU = randomIncrement(compiled)

    KCompiled, PCompiled = evaluate(compiled, dU, dU)
    KInterpreted, PInterpreted = evaluate(interpreted, dU, dU)

    assertClose(KCompiled, KInterpreted, "tangents disagree")
    assertClose(PCompiled, PInterpreted, "fluxes disagree")


@needsKernel
def test_theStateVariablesAgreeAsWell():
    """Not just the returned tangent and flux: the stress, strain, multiplier and yield value the
    kernel leaves behind in the state variables are what the next increment starts from and what
    the field output reports."""

    compiled, interpreted = buildBothPaths()

    dU = randomIncrement(compiled)

    evaluate(compiled, dU, dU)
    evaluate(interpreted, dU, dU)

    assertClose(compiled._stateVarsTemp, interpreted._stateVarsTemp, "state variables disagree")

    # and the plastic state really was reached, so this is not comparing two elastic evaluations
    kappa = np.array([compiled.getResultArray("kappa", p)[0] for p in range(compiled._nMaterialPoints)])

    assert kappa.max() > 0.0


@needsKernel
def test_repeatedEvaluationsAgree():
    """A Newton iteration evaluates the same cell many times from the same accepted state. Both
    paths reset the trial state variables at the top of every evaluation, so repeating one has to
    give the same answer -- a kernel that forgot the reset would drift."""

    compiled, interpreted = buildBothPaths()

    dU = randomIncrement(compiled)

    first = evaluate(compiled, dU, dU)
    second = evaluate(compiled, dU, dU)

    # repeating a compiled evaluation must be bit identical: same code, same inputs
    assert np.array_equal(first[0], second[0]), "the compiled kernel drifts between evaluations"
    assert np.array_equal(first[1], second[1]), "the compiled kernel drifts between evaluations"

    assertClose(second[0], evaluate(interpreted, dU, dU)[0], "the paths disagree after repetition")


@needsKernel
def test_acceptingAndResettingStillWork():
    """The state variable bookkeeping stayed in Python, so it must still act on what the compiled
    kernel wrote."""

    compiled, _ = buildBothPaths()

    dU = randomIncrement(compiled)

    evaluate(compiled, dU, dU)

    trial = np.array(compiled._stateVarsTemp)

    compiled.acceptLastState()
    assert np.array_equal(compiled._stateVars, trial)

    compiled._stateVarsTemp[:] = 0.0
    compiled.resetToLastValidState()
    assert np.array_equal(compiled._stateVarsTemp, trial)


@needsKernel
@pytest.mark.parametrize("cellPosition", ["interior", "corner"])
def test_theCompiledTangentIsConsistent(cellPosition):
    """The numerical tangent checker drives the compiled path through the same contract, so it
    covers the kernel as well -- an independent check that does not rely on the Python
    implementation being right."""

    compiled, _ = buildBothPaths(cellPosition=cellPosition)

    dU = randomIncrement(compiled, scale=4e-3)

    compiled.assertTangentConsistent(dU, dU, 1.0, 1.0, perturbation=1e-9, relativeTolerance=1e-5)


def test_theStencilRunsWithoutTheCompiledKernel():
    """The package has to stay usable when EdelweissFE was installed without its extensions."""

    _, _, stencils = buildSingleCellGrid(nGridPoints=(4, 4))
    stencil = stencils[0]

    gps.useCompiledKernel = False
    try:
        stencil.initializeStencil()

        assert stencil._kernel is None

        K, P = evaluate(stencil, np.zeros(stencil.nDof), np.zeros(stencil.nDof))

        assert np.all(np.isfinite(K))
    finally:
        gps.useCompiledKernel = True
