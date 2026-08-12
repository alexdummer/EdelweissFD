#!/usr/bin/env python3
"""The Fischer-Burmeister smoothing has to be scaled to the stress, or the tangent stops being usable.

Marmot's ``GradientVonMises`` enforces the yield condition through the smoothed complementarity
function ``phi(a, b) = sqrt(a^2 + b^2 + eps) - (a + b)``, where ``a = -f`` is a stress and
``b = dLambda * 1e4``. The derivative ``dphi/da`` multiplies the *whole* yield condition row of the
algorithmic tangent, so its stability decides whether Newton can converge.

At a converged plastic point ``a`` vanishes, so the distance to the corner of ``phi`` is ``b``, which
is proportional to the load increment. With a fixed absolute ``eps`` the corner is arbitrarily sharp
relative to the stress scale, and once ``b`` drops below the residual the solver works at, a
residual sized perturbation of ``a`` swings ``dphi/da`` from -1 to nearly zero: the yield condition
decouples from the displacements and the iteration oscillates. Worse, the usual remedy makes it
worse, because cutting the increment shrinks ``b`` further.

These tests pin the property that fixes it: ``eps`` scales with the square of the yield strength, so
the corner has a fixed radius *relative to the stress scale* and the swing is bounded no matter how
small the increment becomes.

Note that the tangent was *consistent* all along -- verified to 1e-8 against central differences up
to 95 % of exhaustion -- so a consistency check alone cannot catch this. What was wrong was its
stability, which is what is measured here.

The price of the smoothing, that the yield condition holds only to ``eps / (2 b)`` rather than
exactly, is asserted where a genuinely converged plastic state exists rather than a hand built one:
``test_gradientplasticity.py::test_homogeneousStateReproducesTheSofteningLaw``.
"""

import numpy as np
import pytest

from edelweissfe.materials.base.basegradientplasticityhypoelasticmaterial import (
    GradientPlasticityIncrement,
    GradientPlasticityResponse,
    GradientPlasticityTangents,
)

pytestmark = pytest.mark.marmot

shearModulus = 4000.0
poissonsRatio = 0.49
youngsModulus = 2.0 * shearModulus * (1.0 + poissonsRatio)
yieldStrength = 100.0
hardeningModulus = -400.0
gradientParameter = 3600.0

#: The scale Marmot applies to the plastic multiplier inside the complementarity function.
multiplierScale = 1e4


def makeMaterial(smoothing=None):
    """A GRADIENTVONMISES material, optionally with an explicit relative smoothing."""

    from edelweissfe.materials.marmot.marmotgradientplasticityhypoelastic import (
        MarmotGradientPlasticityHypoElasticMaterial,
    )

    properties = [
        youngsModulus,
        poissonsRatio,
        yieldStrength,
        hardeningModulus,
        gradientParameter,
        1.0,
        2.4e-9,
        0.0,
    ]

    if smoothing is not None:
        properties.append(smoothing)

    return MarmotGradientPlasticityHypoElasticMaterial("GRADIENTVONMISES", np.array(properties))


def evaluate(material, stateVars, stress, dStrain, dLambda, laplaceDLambda=0.0):
    """One evaluation from a pristine state, since the material mutates its own state variables."""

    scratch = stateVars.copy()
    material.assignCurrentStateVars(scratch)

    response = GradientPlasticityResponse.createZero(1)
    tangents = GradientPlasticityTangents.createZero(1)
    increment = GradientPlasticityIncrement.createZero(1)

    response.stress[:] = stress
    increment.dStrain[:] = dStrain
    increment.dLambda[0] = dLambda
    increment.laplaceDLambda[0] = laplaceDLambda

    material.computeStress(response, tangents, increment, 1.0, 1.0)

    return response, tangents


def onYieldSurface(material, kappa=0.05):
    """State variables and a stress sitting on the current yield surface, so that ``a`` vanishes."""

    stateVars = np.zeros(material.getNumberOfRequiredStateVars())
    material.assignCurrentStateVars(stateVars)
    material.initializeYourself()
    stateVars[0] = kappa

    # uniaxial compression, for which the Mises stress is the magnitude of the axial component
    yieldStress = yieldStrength + hardeningModulus * kappa

    return stateVars, np.array([0.0, -yieldStress, 0.0, 0.0, 0.0, 0.0])


@pytest.mark.parametrize("dLambda", [1e-4, 1e-6, 1e-8, 1e-10])
def test_theYieldRowDoesNotCollapseAsTheIncrementShrinks(dLambda):
    """The regression. ``dF_dStrain`` carries the factor ``dphi/da``; with an unscaled smoothing it
    collapses towards zero as the increment shrinks, which is what decoupled the two fields."""

    material = makeMaterial()
    stateVars, stress = onYieldSurface(material)

    dStrain = np.array([0.0, -1e-6, 0.0, 0.0, 0.0, 0.0])

    _, tangents = evaluate(material, stateVars, stress, dStrain, dLambda)

    yieldRow = np.abs(tangents.dF_dStrain[0, :]).max()

    # with the stress scaled smoothing the row keeps the magnitude of the elastic stiffness times
    # the flow direction, irrespective of how small the multiplier increment is
    assert yieldRow > 1e-3 * youngsModulus, "the yield condition row collapsed at dLambda={:}".format(dLambda)


def test_theDefaultAppliesWithoutTheOptionalProperty():
    """Eight properties must still work, and give the same answer as passing the default explicitly."""

    withDefault = makeMaterial()
    withExplicit = makeMaterial(smoothing=1e-5)

    results = []
    for material in (withDefault, withExplicit):
        stateVars, stress = onYieldSurface(material)
        response, _ = evaluate(material, stateVars, stress, np.zeros(6), 1e-3)
        results.append(float(response.f[0]))

    assert results[0] == pytest.approx(results[1], rel=1e-12)
