#!/usr/bin/env python3
"""
A simply supported beam in bending, solved by finite differences.

This is the finite difference counterpart of the EdelweissFE regression test
``testfiles/edelweiss-only/SimpleBeamQuad4``: a 2000 x 150 beam of thickness 25, pinned at one
end and on a roller at the other, carrying a uniformly distributed load on its top edge.

Where the tension bar is reproduced exactly on any grid, because a linear displacement field
lies in the span of the difference operators, a bending field does not. This example is
therefore the one that shows the *accuracy* of the scheme, by refining towards the
Euler-Bernoulli deflection

    w = 5 q L^4 / (384 E I) = 22.792 mm,     I = t h^3 / 12

The approach is from below, i.e. the discretization is too stiff on coarse grids. That is the
classical locking of linear kinematics in bending and affects the bilinear ``CPE4`` of the FE
test in the same way.

Note that the load is applied as a *traction*, which the driver converts into grid point
forces weighted by their tributary length. The FE test instead prescribes a fixed force on
every top grid point, which is equivalent only on its own 75 x 4 mesh -- refining that
definition would silently increase the total load.

The material is the native EdelweissFE ``linearelastic``, so this example needs no Marmot.

Run it with::

    python run.py
"""

import numpy as np

from edelweissfd.drivers.pythonscriptedsimulation import FDSimulation
from edelweissfd.stencils.displacementstencil import DisplacementStencil

youngsModulus = 13000.0
poissonsRatio = 0.15

span = 2000.0
depth = 150.0
thickness = 25.0

#: The distributed load on the top edge, as force per unit length of the span.
distributedLoad = 10.0


def analyticalDeflection() -> float:
    """The mid span deflection of Euler-Bernoulli beam theory.

    Returns
    -------
    float
        The deflection.
    """

    secondMomentOfArea = thickness * depth**3 / 12.0

    return 5.0 * distributedLoad * span**4 / (384.0 * youngsModulus * secondMomentOfArea)


def run(nCells=(75, 4), verbose: bool = True) -> tuple:
    """Solve the beam.

    Parameters
    ----------
    nCells
        The number of cells along the span and over the depth.
    verbose
        Be verbose during the simulation.

    Returns
    -------
    tuple
        The model and the field output controller.
    """

    sim = FDSimulation(domainSize=2, name="SimpleBeam2D", verbose=verbose)

    grid = sim.createStructuredGrid(lengths=[span, depth], nGridPoints=[nCells[0] + 1, nCells[1] + 1])

    material = sim.createMaterial("linearelastic", [youngsModulus, poissonsRatio], provider="edelweiss")

    sim.assignStencils(DisplacementStencil, grid, material=material, stressState="plane strain", thickness=thickness)

    step = sim.createStep(stepLength=100.0, maxInc=1.0, minInc=1e-8, maxNumInc=1000, maxIter=25)
    step.addDirichlet("pin", grid.nodeSets["leftBottom"], "displacement", {0: 0.0, 1: 0.0})
    step.addDirichlet("roller", grid.nodeSets["rightBottom"], "displacement", {1: 0.0})
    step.addNeumann("load", grid, "top", "displacement", [0.0, -distributedLoad])

    sim.addNodeFieldOutput("displacement", "displacement", "U", nodeSet=grid.nodeSets["all"])
    sim.addNodeFieldOutput("reaction", "displacement", "P", nodeSet=grid.nodeSets["bottom"])
    sim.addStencilFieldOutput("stress", "stress")

    return sim.run()


def midSpanDeflection(model, fieldOutputs) -> float:
    """The largest downward deflection on the grid line closest to mid span."""

    displacements = fieldOutputs.fieldOutputs["displacement"].getLastResult()
    coordinates = np.array([node.coordinates for node in model.nodeFields["displacement"].nodes])

    distanceToMidSpan = np.abs(coordinates[:, 0] - 0.5 * span)
    atMidSpan = distanceToMidSpan <= distanceToMidSpan.min() + 1e-9

    return float(np.max(np.abs(displacements[atMidSpan, 1])))


def main():
    reference = analyticalDeflection()

    print()
    print("Euler-Bernoulli mid span deflection : {:.4f} mm".format(reference))
    print("total applied load                  : {:.1f}".format(distributedLoad * span))
    print()
    print("{:>12}  {:>10}  {:>9}  {:>12}".format("cells", "w [mm]", "error", "sum(P_y)"))

    previousError = None

    for nCells in [(20, 2), (40, 4), (80, 8), (160, 16), (320, 32)]:
        model, fieldOutputs = run(nCells=nCells, verbose=False)

        deflection = midSpanDeflection(model, fieldOutputs)
        error = deflection / reference - 1.0

        totalReaction = fieldOutputs.fieldOutputs["reaction"].getLastResult()[:, 1].sum()

        rate = "" if previousError is None else "  rate {:.2f}".format(np.log2(abs(previousError / error)))

        print(
            "{:>12}  {:>10.4f}  {:>8.2f}%  {:>12.3f}{:}".format(
                "{:}x{:}".format(*nCells), deflection, 100.0 * error, totalReaction, rate
            )
        )

        previousError = error


if __name__ == "__main__":
    main()
