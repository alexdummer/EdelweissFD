#!/usr/bin/env python3
"""The two field gradient plasticity stencil.

The material of this family owns its second balance equation, the yield condition, and asks to
be told the Laplacian of the plastic multiplier. EdelweissFD supplies it directly by second
difference quotients, so the problem stays a two field problem: displacement and plastic
multiplier, no auxiliary field for the gradient and no penalty.

The reference implementation, Marmot's ``C0GradientPlasticityFiniteElement``, needs all three
because a C0 shape function cannot produce a second derivative.
"""

import numpy as np
import pytest

from edelweissfd.drivers.pythonscriptedsimulation import FDSimulation
from edelweissfd.stencils.gradientplasticitystencil import GradientPlasticityStencil


def gradientVonMisesProperties(
    youngsModulus=20000.0,
    poissonsRatio=0.3,
    yieldStrength=100.0,
    hardeningModulus=-200.0,
    internalLength=8.0,
    useFischerBurmeister=1.0,
):
    """The property vector of Marmot's ``GRADIENTVONMISES``.

    The yield stress is ``fy0 + H kappa - g laplace(kappa)``, so the gradient parameter follows
    from the requested internal length as ``g = |H| l^2``.
    """

    return [
        youngsModulus,
        poissonsRatio,
        yieldStrength,
        hardeningModulus,
        abs(hardeningModulus) * internalLength**2,
        useFischerBurmeister,
        2.4e-9,
        0.0,
    ]


def buildSingleCellGrid(nGridPoints=(5, 5), lengths=(12.0, 12.0), stressState="plane strain", **properties):
    """A small grid covered with gradient plasticity stencils.

    Returns
    -------
    tuple
        The simulation, the grid and the stencils.
    """

    sim = FDSimulation(domainSize=len(nGridPoints), name="gp", verbose=False)

    grid = sim.createStructuredGrid(lengths=list(lengths), nGridPoints=list(nGridPoints))

    material = sim.createMaterial(
        "GradientVonMises", gradientVonMisesProperties(**properties), baseClass="gradientPlasticityHypoElastic"
    )

    stencils = sim.assignStencils(GradientPlasticityStencil, grid, material=material, stressState=stressState)

    return sim, grid, stencils


@pytest.mark.marmot
def test_onlyTwoFieldsAreInvolved():
    """The stencil must not introduce a field for the gradient of the plastic multiplier."""

    sim, grid, stencils = buildSingleCellGrid()

    fieldsOnCorners = stencils[0].fields[0]

    assert fieldsOnCorners == ["displacement", "plastic multiplier"]

    sim.model.prepareYourself(sim.journal)

    assert set(sim.model.nodeFields) == {"displacement", "plastic multiplier"}


@pytest.mark.marmot
def test_moleculeReachesBeyondTheCellAndCarriesOnlyTheMultiplier():
    """The Laplacian reaches one grid point past the cell. Those outer grid points take part
    through the plastic multiplier alone, which is what keeps the block small."""

    sim, grid, stencils = buildSingleCellGrid(nGridPoints=(5, 5))

    interior = [s for s in stencils if s.nNodes == max(t.nNodes for t in stencils)][0]

    assert interior.nCorners == 4
    assert interior.nNodes == 12, "four corners plus the eight outer grid points of the cross"

    for node in range(interior.nCorners):
        assert interior.fields[node] == ["displacement", "plastic multiplier"]

    for node in range(interior.nCorners, interior.nNodes):
        assert interior.fields[node] == ["plastic multiplier"]

    # four corners with three degrees of freedom, eight outer grid points with one
    assert interior.nDof == 4 * 3 + 8 * 1

    # a corner cell has a smaller molecule, because the ghost nodes mirror back onto it
    corner = min(stencils, key=lambda s: s.nNodes)

    assert corner.nNodes < interior.nNodes


@pytest.mark.marmot
@pytest.mark.parametrize("cellPosition", ["corner", "interior"])
def test_tangentIsConsistent(cellPosition):
    """All four coupling blocks, including the ones going through the Laplacian."""

    sim, grid, stencils = buildSingleCellGrid(nGridPoints=(5, 5))

    if cellPosition == "corner":
        stencil = min(stencils, key=lambda s: s.nNodes)
    else:
        stencil = max(stencils, key=lambda s: s.nNodes)

    stencil.initializeStencil()

    rng = np.random.default_rng(7)

    dU = np.zeros(stencil.nDof)
    dU[stencil._displacementDofs] = rng.normal(scale=4e-3, size=stencil._displacementDofs.size)
    dU[stencil._multiplierDofs] = np.abs(rng.normal(scale=5e-4, size=stencil._multiplierDofs.size))

    stencil.assertTangentConsistent(dU, dU, 1.0, 1.0, perturbation=1e-9, relativeTolerance=1e-5)


@pytest.mark.marmot
def test_homogeneousStateReproducesTheSofteningLaw():
    """A sharp check of the whole chain.

    Under a homogeneous state the Laplacian of the plastic multiplier vanishes, so the yield
    condition reduces to the local one and the stress has to sit exactly on

        sigma_mises = fy0 + H kappa

    which also confirms that neither the Laplacian nor its boundary treatment disturbs a
    uniform field.

    Not to machine precision, though, and the bound is worth stating because it is a property of the
    formulation rather than of this discretisation. Fischer-Burmeister enforces the yield condition
    through the smoothed complementarity function
    ``phi(a, b) = sqrt(a^2 + b^2 + eps) - (a + b)`` with ``a = -f`` and ``b`` proportional to the
    plastic multiplier increment. At a converged plastic point ``a`` vanishes, so ``phi`` reduces to
    ``sqrt(b^2 + eps) - b``, which is ``eps / (2 b)`` rather than zero: the yield condition is
    satisfied only to that, and ``eps`` cannot be made arbitrarily small because it is also what
    keeps the algorithmic tangent from swinging near the corner -- see ``fbSmoothingRelative`` in
    Marmot's ``GradientVonMises``. With the default smoothing the violation is of order 1e-9
    relative to the yield strength, which is what the tolerance below allows for.
    """

    yieldStrength = 100.0
    hardeningModulus = -200.0
    sideLength = 10.0
    axialStrain = -0.02

    sim, grid, stencils = buildSingleCellGrid(
        nGridPoints=(5, 5),
        lengths=(sideLength, sideLength),
        yieldStrength=yieldStrength,
        hardeningModulus=hardeningModulus,
    )

    # a homogeneous state of uniaxial strain: prescribe the displacement on the whole boundary
    step = sim.createStep(stepLength=1.0, maxInc=2e-2, minInc=1e-9, maxNumInc=3000, maxIter=20)
    for boundary in ("left", "right"):
        step.addDirichlet(boundary, grid.nodeSets[boundary], "displacement", {0: 0.0})
    step.addDirichlet("bottom", grid.nodeSets["bottom"], "displacement", {0: 0.0, 1: 0.0})
    step.addDirichlet("top", grid.nodeSets["top"], "displacement", {0: 0.0, 1: axialStrain * sideLength})

    sim.addStencilFieldOutput("stress", "stress", stencils=stencils)
    sim.addStencilFieldOutput("kappa", "kappa", stencils=stencils)

    model, fieldOutputs = sim.run()

    stress = np.asarray(fieldOutputs.fieldOutputs["stress"].getLastResult()).reshape(-1, 6)
    kappa = fieldOutputs.fieldOutputs["kappa"].getLastResult().flatten()

    assert kappa.min() > 1e-4, "the state should be plastic"

    # every material point sees the same state
    assert np.allclose(kappa, kappa[0], rtol=1e-8)
    assert np.allclose(stress, stress[0], atol=1e-8 * np.max(np.abs(stress)))

    s11, s22, s33, s12 = stress[0, 0], stress[0, 1], stress[0, 2], stress[0, 3]

    misesStress = np.sqrt(0.5 * ((s11 - s22) ** 2 + (s22 - s33) ** 2 + (s33 - s11) ** 2) + 3.0 * s12**2)

    assert misesStress == pytest.approx(yieldStrength + hardeningModulus * kappa[0], rel=1e-7)


@pytest.mark.marmot
def test_plasticMultiplierStaysZeroWhileElastic():
    """Below yield the plastic multiplier must not be activated anywhere, which is what the
    complementarity formulation of the material has to deliver."""

    sideLength = 10.0

    sim, grid, stencils = buildSingleCellGrid(nGridPoints=(4, 4), lengths=(sideLength, sideLength))

    # a tenth of the strain needed to yield
    step = sim.createStep(stepLength=1.0, maxInc=0.5, minInc=1e-9, maxNumInc=100, maxIter=20)
    step.addDirichlet("bottom", grid.nodeSets["bottom"], "displacement", {0: 0.0, 1: 0.0})
    step.addDirichlet("top", grid.nodeSets["top"], "displacement", {0: 0.0, 1: -0.0005 * sideLength})

    sim.addNodeFieldOutput("multiplier", "plastic multiplier", "U", nodeSet=grid.nodeSets["all"])
    sim.addStencilFieldOutput("kappa", "kappa", stencils=stencils)

    model, fieldOutputs = sim.run()

    multiplier = fieldOutputs.fieldOutputs["multiplier"].getLastResult().flatten()
    kappa = fieldOutputs.fieldOutputs["kappa"].getLastResult().flatten()

    assert np.max(np.abs(multiplier)) < 1e-10
    assert np.max(np.abs(kappa)) < 1e-10
