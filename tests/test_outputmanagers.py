#!/usr/bin/env python3
"""Output managers requested through :meth:`FDSimulation.addOutputManager`.

They are created through EdelweissFE's registry and validated against their option schema, so
both a manager with a required option and a label (``monitor``) and one without a label
(``statusfile``) are covered, as is the rejection of an unknown option.
"""

import pytest

from edelweissfd.drivers.pythonscriptedsimulation import FDSimulation
from edelweissfd.stencils.displacementstencil import DisplacementStencil

pytestmark = pytest.mark.marmot


def buildBar(outputManagers: dict) -> FDSimulation:
    """A one dimensional bar under tension, with the requested output managers."""

    sim = FDSimulation(domainSize=1, name="bar1D", verbose=False)

    grid = sim.createStructuredGrid(lengths=[100.0], nGridPoints=[5])
    material = sim.createMaterial("LinearElastic", [20000.0, 0.2])

    sim.assignStencils(DisplacementStencil, grid, material=material, stressState="uniaxial stress")

    step = sim.createStep(stepLength=1.0)
    step.addDirichlet("clamped", grid.nodeSets["left"], "displacement", {0: 0.0})
    step.addNeumann("pulled", grid, "right", "displacement", [10.0])

    sim.addNodeFieldOutput("U", "displacement", "U", nodeSet=grid.nodeSets["right"])

    for name, (managerType, options) in outputManagers.items():
        sim.addOutputManager(managerType, name=name, **options)

    return sim


def test_monitorAndStatusFileAreCreatedAndRun(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    sim = buildBar(
        {
            "tipMonitor": ("monitor", {"fieldOutput": "U"}),
            "status": ("statusfile", {"filename": "bar.sta"}),
        }
    )

    model, _ = sim.run()

    # the monitor takes its name from its label, which defaults to the requested name
    assert set(model.outputManagers) == {"tipMonitor", "status"}

    monitor = model.outputManagers["tipMonitor"]
    assert [job["fieldOutput"].name for job in monitor.monitorJobs] == ["U"]

    assert (tmp_path / "bar.sta").read_text().strip()


def test_monitorWithoutItsRequiredOptionRaises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    sim = buildBar({"tipMonitor": ("monitor", {})})

    with pytest.raises(ValueError, match="fieldOutput"):
        sim.run()


def test_unknownOptionRaises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    sim = buildBar({"status": ("statusfile", {"fileName2": "bar.sta"})})

    with pytest.raises(ValueError, match="fileName2"):
        sim.run()
