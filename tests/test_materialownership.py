#!/usr/bin/env python3
"""Every stencil must own its material instance.

A point-wise material carries the state variable storage it operates on as mutable state, so it
can serve only one material point at a time. Within a stencil that is fine -- material points are
evaluated one after another -- but the solvers hand chunks of entities to different threads, so
two stencils sharing one material instance re-point it from two threads at once.

On a free threaded interpreter that aborts the process outright:

    Fatal Python error: __pyx_fatalerror: Acquisition count is -1

because the Cython memoryview holding the state variables is released twice. Under the global
interpreter lock the very same race is merely unlikely and silent, and then a material point is
evaluated against another point's state variables. Neither failure shows up in a serial run, which
is why this is tested by identity rather than by outcome.
"""

import pytest

from edelweissfd.drivers.pythonscriptedsimulation import FDSimulation
from edelweissfd.materials.provider import MaterialProvider, materialInstanceFrom
from edelweissfd.stencils.displacementstencil import DisplacementStencil


def buildGrid(nGridPoints=(4, 4), **materialOptions):
    """A small grid covered with displacement stencils."""

    sim = FDSimulation(domainSize=len(nGridPoints), name="ownership", verbose=False)

    grid = sim.createStructuredGrid(lengths=[6.0] * len(nGridPoints), nGridPoints=list(nGridPoints))

    material = sim.createMaterial("linearelastic", [20000.0, 0.3], provider="edelweiss", **materialOptions)

    stencils = sim.assignStencils(DisplacementStencil, grid, material=material, stressState="plane strain")

    return sim, grid, stencils


def test_createMaterialReturnsAProviderRatherThanAMaterial():
    """The description is handed out, not an instance -- otherwise there is nothing to mint from."""

    sim = FDSimulation(domainSize=2, name="ownership", verbose=False)

    provider = sim.createMaterial("linearelastic", [20000.0, 0.3], provider="edelweiss")

    assert isinstance(provider, MaterialProvider)

    # not callable, so assignStencils cannot mistake it for the spatially varying form
    assert not callable(provider)

    first = provider.createMaterial()
    second = provider.createMaterial()

    assert first is not second


def test_noTwoStencilsShareAMaterialInstance():
    """The invariant itself. One instance per stencil, all distinct."""

    sim, grid, stencils = buildGrid(nGridPoints=(4, 4))

    assert len(stencils) == 9

    materials = [stencil._material for stencil in stencils]

    assert len({id(material) for material in materials}) == len(stencils)


def test_aSpatiallyVaryingMaterialAlsoYieldsDistinctInstances():
    """The callable form returns a provider per location, and each stencil still gets its own
    instance -- a weak region shared between neighbouring stencils would otherwise share state."""

    sim = FDSimulation(domainSize=2, name="ownership", verbose=False)

    grid = sim.createStructuredGrid(lengths=[6.0, 6.0], nGridPoints=[4, 4])

    strong = sim.createMaterial("linearelastic", [20000.0, 0.3], provider="edelweiss")
    weak = sim.createMaterial("linearelastic", [2000.0, 0.3], provider="edelweiss")

    def materialAt(coordinates):
        return weak if coordinates[0] < 3.0 else strong

    stencils = sim.assignStencils(DisplacementStencil, grid, material=materialAt, stressState="plane strain")

    materials = [stencil._material for stencil in stencils]

    assert len({id(material) for material in materials}) == len(stencils)

    # and the two regions really did get different properties
    youngsModuli = {float(stencil._material._E) for stencil in stencils}

    assert youngsModuli == {20000.0, 2000.0}


def test_aReadyMadeMaterialIsPassedThrough():
    """The tangent tests build a single stencil by hand and evaluate it serially, so handing over
    an instance directly stays legal."""

    material = object()

    assert materialInstanceFrom(material) is material


@pytest.mark.marmot
def test_marmotMaterialsAreNotSharedEither():
    """The failure this guards against was reported for a Marmot material, so cover that path too
    rather than only the native one."""

    sim = FDSimulation(domainSize=2, name="ownership", verbose=False)

    grid = sim.createStructuredGrid(lengths=[6.0, 6.0], nGridPoints=[4, 4])

    material = sim.createMaterial("LinearElastic", [20000.0, 0.3])

    stencils = sim.assignStencils(DisplacementStencil, grid, material=material, stressState="plane strain")

    materials = [stencil._material for stencil in stencils]

    assert len({id(material) for material in materials}) == len(stencils)
