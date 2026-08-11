#!/usr/bin/env python3
"""The mesh study's convergence diagnostic, which has to refuse as often as it reports.

Three widths can always be fitted by *something*. The value of the diagnostic is that it declines
to attach a convergence order to a sequence that is not converging, because such a number would
read as evidence of grid independence where there is none. These are the four sequences actually
met while building the compressed panel example.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples" / "CompressedPanel2D"))

from run import convergenceOfWidth  # noqa: E402

spacings = [3.0, 2.0, 1.5]


def test_aResolvedSequenceGetsAnOrderAndAPositiveLimit():
    """Measured with an internal length of six, where the grid resolves it: the order comes out at
    the formal order of the difference operators and the limit at a few internal lengths."""

    result = convergenceOfWidth(spacings, [20.857, 17.149, 15.838])

    assert result["diagnosis"] == ""
    assert result["order"] == pytest.approx(2.0, abs=0.1)
    assert result["extrapolated"] == pytest.approx(14.1, abs=0.5)


def test_aLockedSequenceIsRejectedForGrowingDifferences():
    """Measured before the volumetric averaging: the width moved *further* with each refinement,
    because locking gets worse as the grid is refined."""

    result = convergenceOfWidth(spacings, [36.246, 34.797, 30.248])

    assert np.isnan(result["order"])
    assert "grows under refinement" in result["diagnosis"]


def test_aStillNarrowingSequenceIsRejectedForAnImpossibleLimit():
    """Measured after the volumetric averaging with an internal length of three, where the grid
    only just reaches it: the differences do shrink, but a fit gives an order below one and a
    negative limit, so the band is still narrowing rather than settling."""

    result = convergenceOfWidth(spacings, [25.639, 20.355, 17.142])

    assert np.isnan(result["order"])
    assert "still fall steeply" in result["diagnosis"]


def test_aWidthSetByTheGridExtrapolatesToNothing():
    """The control case: a width proportional to the spacing, i.e. set by the grid and not by any
    internal length, must never be reported as converged."""

    result = convergenceOfWidth(spacings, [2.0 * h for h in spacings])

    assert np.isnan(result["order"])
    assert result["diagnosis"]


def test_aNonMonotoneSequenceIsRejected():
    """Noise rather than convergence."""

    result = convergenceOfWidth(spacings, [20.0, 17.0, 18.0])

    assert np.isnan(result["order"])
    assert "not monotone" in result["diagnosis"]
