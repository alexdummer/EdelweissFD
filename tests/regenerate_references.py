#!/usr/bin/env python3
"""Regenerate the stored reference solutions of the EdelweissFD test suite.

This is the equivalent of ``run_tests_edelweissfe --create`` of EdelweissFE: it writes the
concatenated nodal solution vector of every regression case with :func:`numpy.savetxt`, so
that the stored references stay comparable between the two packages.

Run it from the repository root::

    python -m tests.regenerate_references
"""

import numpy as np

from tests.test_at2phasefieldbar import (
    referenceDirectory,
    solutionVector,
    solveBarUnderCrackOpeningControl,
)


def regenerateAT2PhaseFieldBar1D():
    """Write the reference of the one dimensional AT2 phase field bar."""

    properties = np.array([20000.0, 0.2, 0.1, 5.0])

    model, fieldOutputs, grid = solveBarUnderCrackOpeningControl(51, properties)

    referenceDirectory.mkdir(parents=True, exist_ok=True)

    target = referenceDirectory / "AT2PhaseFieldBar1D.ref"

    np.savetxt(target, solutionVector(model))

    print("wrote {:} values to {:}".format(len(solutionVector(model)), target))


def main():
    regenerateAT2PhaseFieldBar1D()


if __name__ == "__main__":
    main()
