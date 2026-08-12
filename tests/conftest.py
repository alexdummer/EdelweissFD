#!/usr/bin/env python3
"""Shared pytest fixtures and markers of the EdelweissFD test suite.

Tests which need a Marmot material are marked ``marmot`` and are skipped automatically if
the point-wise Marmot material interfaces of EdelweissFE were not compiled, mirroring the
split of the EdelweissFE regression suite into ``testfiles/edelweiss-only`` and
``testfiles/marmot``.
"""

import numpy as np
import pytest
from edelweissfe.utils.misc import checkSuccessfulExtension

#: Whether the point-wise Marmot material interfaces of EdelweissFE are available.
marmotMaterialsAvailable = checkSuccessfulExtension(
    "edelweissfe.materials.marmot.marmothypoelastic"
) and checkSuccessfulExtension("edelweissfe.materials.marmot.marmotgradientenhancedhypoelastic")


def pytest_collection_modifyitems(config, items):
    """Skip the Marmot dependent tests if the extensions are missing."""

    if marmotMaterialsAvailable:
        return

    skip = pytest.mark.skip(reason="the point-wise Marmot material interfaces of EdelweissFE are not compiled")

    for item in items:
        if "marmot" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def linearElasticProperties():
    """Young's modulus and Poisson's ratio of the reference linear elastic material."""

    return np.array([20000.0, 0.2])


@pytest.fixture
def at2PhaseFieldProperties():
    """Young's modulus, Poisson's ratio, fracture energy and length scale of the AT2 phase
    field material, matching ``testfiles/marmot/AT2PhaseField`` of EdelweissFE."""

    return np.array([20000.0, 0.2, 0.1, 5.0])


@pytest.fixture
def vonMisesProperties():
    """Young's modulus, Poisson's ratio and the hardening parameters of Marmot's VONMISES,
    matching ``testfiles/marmot`` of EdelweissFE."""

    return np.array([210000.0, 0.3, 550.0, 1000.0, 200.0, 1400.0])
