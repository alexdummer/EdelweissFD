#!/usr/bin/env python3
"""
Gradient plasticity of a compressed panel with a weak corner, solved by finite differences.

A 60 x 120 panel in plane strain is compressed vertically between frictionless platens. A small
region in the bottom left corner is given a reduced yield strength, which is enough to break the
symmetry of the problem: the material softens, so the plastic zone does not spread but
concentrates, and it concentrates starting from the weak spot.

The platens are frictionless on purpose. With rough platens the material is triaxially confined
near them and least confined at mid height, so plasticity starts at mid height regardless of how
weak the corner is made -- the weak spot then sits in the region that is hardest to yield and
loses against the boundary conditions. Frictionless platens leave the stress state uniform and
make the imperfection the only symmetry breaking feature. The grid point pinned against rigid
body translation is deliberately the one at the *opposite* bottom corner, so that its local
constraint does not interfere with the weak spot.

Two fields live on every grid point, the displacement and the plastic multiplier, see
:mod:`edelweissfd.stencils.gradientplasticitystencil`. The Laplacian of the plastic multiplier
is formed directly by second difference quotients, with homogeneous Neumann conditions at the
boundary through ghost nodes, so no auxiliary field and no penalty are involved. The
constitutive model is Marmot's ``GRADIENTVONMISES``, whose yield stress is

    sigma_Y = fy0 + H kappa - g laplace(kappa)

With ``H < 0`` the response softens, and without the gradient term the plastic zone would
collapse onto a single row of cells, i.e. the answer would be set by the grid. The gradient term
regularises it and gives the zone an internal length

    l = sqrt(g / |H|)

which is what the mesh study in :func:`main` checks: refining the grid has to leave the width of
the localized zone alone.

Two properties of the softening law set the scale of everything else and are worth stating
explicitly, because both bit hard while this example was built:

* The material is exhausted at ``kappa = fy0 / |H|``, where the local yield stress reaches zero.
  Newton stops converging at any increment size somewhat before that, so the loading has to be
  targeted as a fraction of that range rather than as an absolute number.
* The width of the localized zone comes out at a few times ``l``. For the zone to be visibly
  narrower than the panel, ``l`` has to be a small fraction of the panel width -- and the grid
  still has to resolve it. Those two requirements pull in opposite directions and are what makes
  the cost of this example.

Run it with::

    python run.py
"""

from dataclasses import dataclass

import numpy as np
from edelweissfd.drivers.pythonscriptedsimulation import FDSimulation
from edelweissfd.stencils.gradientplasticitystencil import GradientPlasticityStencil


@dataclass(frozen=True)
class Panel:
    """The panel, its material and its imperfection.

    Everything the simulation needs beyond the grid resolution and the loading. Grouped so that
    the parameter study in :func:`main` can vary it without touching module state.
    """

    #: The width and the height of the panel.
    width: float = 60.0
    height: float = 120.0

    youngsModulus: float = 20000.0
    poissonsRatio: float = 0.3

    #: The initial yield strength.
    yieldStrength: float = 100.0

    #: The softening modulus. Negative, otherwise nothing localises.
    hardeningModulus: float = -2000.0

    #: The internal length of the regularisation. The gradient parameter follows from it as
    #: ``g = |H| l^2``. Six is a compromise: the localized zone comes out a few times ``l`` wide,
    #: so a much larger value fills the whole 60 wide panel and nothing recognisable localizes,
    #: while a much smaller one needs a grid fine enough to resolve it and the cost of the mesh
    #: study explodes.
    internalLength: float = 6.0

    #: The yield strength of the weak corner, as a fraction of the initial one. A shallow
    #: imperfection does not win against the rest of the panel: at 0.9 the plastic zone comes out
    #: symmetric about the vertical centre line and merely sits at the bottom edge, at 0.8 it is
    #: anchored at the corner. Going deeper still, to 0.7, localizes so sharply that Newton gives
    #: up before the zone is fully developed.
    weakeningFactor: float = 0.8

    #: The radius of the weakened region around the bottom left corner.
    weakCornerRadius: float = 6.0

    #: Marmot's smooth Fischer-Burmeister complementarity formulation, which is what makes the
    #: elastic to plastic switch differentiable and thus Newton friendly.
    useFischerBurmeister: float = 1.0

    density: float = 2.4e-9
    nonlocalViscosity: float = 0.0

    @property
    def gradientParameter(self) -> float:
        """The gradient parameter ``g`` belonging to the requested internal length."""

        return abs(self.hardeningModulus) * self.internalLength**2

    @property
    def exhaustionMultiplier(self) -> float:
        """The plastic multiplier at which the local yield stress reaches zero.

        The whole softening branch is contained in ``kappa`` between zero and this value, so it
        is the natural yardstick for how far the panel can be driven.
        """

        return self.yieldStrength / abs(self.hardeningModulus)

    @property
    def misesFactor(self) -> float:
        """The Mises stress per unit axial stress, for uniaxial stress in plane strain.

        With ``sigma_22 = -sigma``, ``sigma_11 = 0`` and ``sigma_33 = nu sigma_22`` from
        ``eps_33 = 0``.
        """

        nu = self.poissonsRatio

        return float(np.sqrt(0.5 * (1.0 + (1.0 - nu) ** 2 + nu**2)))

    def shorteningAtYielding(self) -> float:
        """The end shortening at which the weak corner first yields, in closed form.

        Frictionless platens and free lateral faces leave a state of uniaxial stress, so the
        axial strain at yielding is ``-sigma (1 - nu^2) / E`` with
        ``sigma = weakeningFactor fy0 / misesFactor``. Used to size the displacement controlled
        part of the loading, and worth having as an independent check on the computed peak.
        """

        stress = self.weakeningFactor * self.yieldStrength / self.misesFactor

        return -stress * (1.0 - self.poissonsRatio**2) / self.youngsModulus * self.height


#: The panel the mesh study uses.
defaultPanel = Panel()

#: How far past the closed form onset of yielding the displacement controlled part of the loading
#: goes. It only has to pass yielding, so that the plastic multiplier at the weak corner is
#: non-zero and can serve as the control quantity: while everything is still elastic it is
#: identically zero, a trial load increment produces no change in it at all, and the arc length
#: controller has nothing to divide by.
yieldingSafetyFactor = 1.15

#: How far into the softening range the weak corner is driven under indirect control, as a
#: fraction of :attr:`Panel.exhaustionMultiplier`. Beyond roughly one half the band is close
#: enough to zero strength that Newton stops converging at any increment size.
fractionOfExhaustion = 0.45

#: The end shortening the arc length parameter scales. Only an upper bound -- the controller
#: decides how much of it is actually applied, and on the softening branch that turns out to be
#: almost nothing, i.e. the response snaps back.
referenceShortening = -12.0


def createMaterialFactory(sim: FDSimulation, panel: Panel):
    """A material factory weakening a corner region of the panel.

    Parameters
    ----------
    sim
        The simulation, used to create the materials.
    panel
        The panel and its material.

    Returns
    -------
    callable
        A callable taking the coordinates of a material point and returning a material.
    """

    def createMaterial(yieldStress):
        return sim.createMaterial(
            "GradientVonMises",
            [
                panel.youngsModulus,
                panel.poissonsRatio,
                yieldStress,
                panel.hardeningModulus,
                panel.gradientParameter,
                panel.useFischerBurmeister,
                panel.density,
                panel.nonlocalViscosity,
            ],
            baseClass="gradientPlasticityHypoElastic",
        )

    strongMaterial = createMaterial(panel.yieldStrength)
    weakMaterial = createMaterial(panel.weakeningFactor * panel.yieldStrength)

    def materialAt(coordinates):
        distanceToCorner = np.linalg.norm(coordinates[:2])

        return weakMaterial if distanceToCorner <= panel.weakCornerRadius else strongMaterial

    return materialAt


def weakCornerControlPoints(grid, panel: Panel) -> list:
    """The two grid points closest to the weak corner, used as the indirect control.

    The zone localizes there, so the plastic multiplier at those points is the quantity that
    grows monotonically through the whole loading history.

    Parameters
    ----------
    grid
        The grid of the panel.
    panel
        The panel, for the radius of the weakened region.

    Returns
    -------
    list
        Two grid points.
    """

    inWeakCorner = sorted(
        (node for node in grid.nodes.values() if np.linalg.norm(node.coordinates[:2]) <= panel.weakCornerRadius),
        key=lambda node: np.linalg.norm(node.coordinates[:2]),
    )

    if len(inWeakCorner) < 2:
        raise ValueError("The grid is too coarse to place two control points inside the weak corner.")

    return inWeakCorner[:2]


def run(
    nCells=(20, 40),
    panel: Panel = defaultPanel,
    control: str = "indirect",
    verbose: bool = True,
) -> tuple:
    """Compress the panel.

    Parameters
    ----------
    nCells
        The number of cells across the width and over the height.
    panel
        The panel and its material.
    control
        ``indirect`` drives the plastic multiplier at the weak corner with the arc length solver,
        which is what makes the softening branch reachable. ``displacement`` drives the end
        shortening directly, which is simpler but only survives until the response snaps back.
    verbose
        Be verbose during the simulation.

    Returns
    -------
    tuple
        The model, the field output controller and the grid.
    """

    sim = FDSimulation(domainSize=2, name="CompressedPanel2D", verbose=verbose)

    grid = sim.createStructuredGrid(lengths=[panel.width, panel.height], nGridPoints=[nCells[0] + 1, nCells[1] + 1])

    stencils = sim.assignStencils(
        GradientPlasticityStencil,
        grid,
        material=createMaterialFactory(sim, panel),
        stressState="plane strain",
        thickness=1.0,
    )

    sim.createSolver("NIST")

    if control == "indirect":
        sim.createSolver("NISTPArcLength")

    firstStep = sim.createStep(solver="NIST", stepLength=1.0, maxInc=5e-2, minInc=1e-9, maxNumInc=3000, maxIter=25)

    # Frictionless platens: only the vertical displacement is prescribed on the loaded faces, the
    # panel is free to slide sideways, and a single grid point is pinned against rigid body
    # translation. Every bottom grid point being held vertically already rules out rotation.
    firstStep.addDirichlet("bottom", grid.nodeSets["bottom"], "displacement", {1: 0.0})
    firstStep.addDirichlet("pin", grid.nodeSets["rightBottom"], "displacement", {0: 0.0})

    if control == "displacement":
        # Enough to pass the peak, if the solver survives that far.
        firstStep.addDirichlet("top", grid.nodeSets["top"], "displacement", {1: 4.0 * panel.shorteningAtYielding()})
    else:
        # Displacement control up to just past the onset of yielding, then hand over to the arc
        # length solver driving the plastic multiplier of the weak corner.
        firstStep.addDirichlet(
            "top",
            grid.nodeSets["top"],
            "displacement",
            {1: yieldingSafetyFactor * panel.shorteningAtYielding()},
        )

        secondStep = sim.createStep(
            solver="NISTPArcLength", stepLength=1.0, maxInc=1e-2, minInc=1e-9, maxNumInc=3000, maxIter=25
        )
        secondStep.addDirichlet("top", grid.nodeSets["top"], "displacement", {1: referenceShortening})

        controlPoints = weakCornerControlPoints(grid, panel)

        secondStep.addIndirectControl(
            controlPoints[0],
            [0.5],
            controlPoints[1],
            [0.5],
            fractionOfExhaustion * panel.exhaustionMultiplier,
            field="plastic multiplier",
        )

    sim.addNodeFieldOutput("displacement", "displacement", "U", nodeSet=grid.nodeSets["all"])
    sim.addNodeFieldOutput("plasticMultiplier", "plastic multiplier", "U", nodeSet=grid.nodeSets["all"])

    sim.addNodeFieldOutput(
        "reactionForce",
        "displacement",
        "P",
        nodeSet=grid.nodeSets["top"],
        saveHistory=True,
        f_x=lambda x: np.sum(x[:, 1]),
        export="reactionForce",
    )
    sim.addNodeFieldOutput(
        "shortening",
        "displacement",
        "U",
        nodeSet=grid.nodeSets["top"],
        saveHistory=True,
        f_x=lambda x: np.mean(x[:, 1]),
        export="shortening",
    )

    sim.addStencilFieldOutput("kappa", "kappa", stencils=stencils)
    sim.addStencilFieldOutput("stress", "stress", stencils=stencils)

    model, fieldOutputs = sim.run()

    return model, fieldOutputs, grid


def plasticZone(model, fieldOutputs) -> tuple:
    """The coordinates and the plastic multiplier of the grid points.

    Returns
    -------
    tuple
        The coordinates and the plastic multiplier per grid point.
    """

    coordinates = np.array([node.coordinates for node in model.nodeFields["plastic multiplier"].nodes])
    multiplier = fieldOutputs.fieldOutputs["plasticMultiplier"].getLastResult().flatten()

    return coordinates, multiplier


def bandGeometry(coordinates: np.ndarray, multiplier: np.ndarray) -> dict:
    """The shape of the plastic zone, from the second moments of the plastic multiplier.

    The plastic multiplier is treated as a mass distribution. Its weighted covariance has two
    principal directions; twice the square root of the larger eigenvalue is a length along the
    zone, twice the square root of the smaller one is a width across it. Both are threshold free
    and, for a resolved field, independent of the grid.

    Additionally the participation ratio ``(sum l)^2 / sum l^2`` is reported as a fraction of all
    grid points. It is close to one while the plastic zone is diffuse and drops towards the area
    fraction of the zone once the deformation localizes, which makes it the honest indicator of
    *whether* localization happened at all -- the second moments alone happily report a plausible
    looking width for a completely diffuse field.

    Parameters
    ----------
    coordinates
        The grid point coordinates.
    multiplier
        The plastic multiplier per grid point.

    Returns
    -------
    dict
        The centre, the length, the width, the inclination and the plastic fraction.
    """

    weights = np.clip(multiplier, 0.0, None)

    if weights.sum() <= 0.0:
        return dict(centre=np.full(2, np.nan), length=np.nan, width=np.nan, angle=np.nan, fraction=0.0)

    inPlane = coordinates[:, :2]

    centre = np.average(inPlane, axis=0, weights=weights)
    deviation = inPlane - centre

    covariance = np.einsum("i,ij,ik->jk", weights, deviation, deviation) / weights.sum()

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    length = 2.0 * np.sqrt(max(eigenvalues[-1], 0.0))
    width = 2.0 * np.sqrt(max(eigenvalues[0], 0.0))

    majorAxis = eigenvectors[:, -1]
    angle = np.degrees(np.arctan2(abs(majorAxis[1]), abs(majorAxis[0])))

    participation = weights.sum() ** 2 / np.sum(weights**2)

    return dict(
        centre=centre,
        length=float(length),
        width=float(width),
        angle=float(angle),
        fraction=float(participation / weights.size),
    )


def plot(fileName: str, model, fieldOutputs, grid, panel: Panel = defaultPanel):
    """Draw the plastic multiplier over the panel next to the load displacement curve.

    Parameters
    ----------
    fileName
        Where to write the figure.
    model
        The model.
    fieldOutputs
        The field output controller.
    grid
        The grid of the panel.
    panel
        The panel, for the weak corner outline.
    """

    import matplotlib.pyplot as plt

    coordinates, multiplier = plasticZone(model, fieldOutputs)

    # sort the flat field onto the tensor product grid rather than trusting the order in which
    # the node field happens to list its grid points
    shape = tuple(int(n) for n in grid.shape)
    nodes = model.nodeFields["plastic multiplier"].nodes

    x = np.empty(shape)
    y = np.empty(shape)
    field = np.empty(shape)

    for node, coordinate, value in zip(nodes, coordinates, multiplier):
        index = tuple(int(i) for i in grid.gridIndexOf(node))

        x[index] = coordinate[0]
        y[index] = coordinate[1]
        field[index] = value

    force = np.abs(np.array(fieldOutputs.fieldOutputs["reactionForce"].getResultHistory()).flatten())
    shortening = np.abs(np.array(fieldOutputs.fieldOutputs["shortening"].getResultHistory()).flatten())

    figure, axes = plt.subplots(1, 2, figsize=(9.0, 5.0))

    contour = axes[0].contourf(x, y, np.clip(field, 0.0, None), levels=24, cmap="inferno")
    figure.colorbar(contour, ax=axes[0], label="plastic multiplier")

    axes[0].add_patch(plt.Circle((0.0, 0.0), panel.weakCornerRadius, fill=False, color="deepskyblue", lw=1.5))

    axes[0].set_aspect("equal")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title("{:} x {:} cells,  l = {:.1f}".format(shape[0] - 1, shape[1] - 1, panel.internalLength))

    axes[1].plot(shortening, force, "-")
    axes[1].axvline(abs(panel.shorteningAtYielding()), color="grey", ls="--", lw=1.0, label="closed form onset")
    axes[1].set_xlabel("end shortening")
    axes[1].set_ylabel("reaction force")
    axes[1].set_title("load displacement")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    figure.tight_layout()
    figure.savefig(fileName, dpi=110)
    plt.close(figure)


def report(name: str, model, fieldOutputs, panel: Panel = defaultPanel) -> dict:
    """Print the peak load and where the plastic deformation concentrated."""

    force = np.abs(np.array(fieldOutputs.fieldOutputs["reactionForce"].getResultHistory()).flatten())
    shortening = np.abs(np.array(fieldOutputs.fieldOutputs["shortening"].getResultHistory()).flatten())

    coordinates, multiplier = plasticZone(model, fieldOutputs)

    peak = int(np.argmax(force))
    hottest = int(np.argmax(multiplier))

    band = bandGeometry(coordinates, multiplier)

    print()
    print("=== {:} ===".format(name))
    print("increments solved      : {:}".format(len(force)))
    print("peak load              : {:.4f}".format(force[peak]))
    print(
        "shortening at peak     : {:.5f}   (closed form onset {:.5f})".format(
            shortening[peak], abs(panel.shorteningAtYielding())
        )
    )
    print("final load             : {:.4f}  ({:.1f} % of peak)".format(force[-1], 100.0 * force[-1] / force[peak]))
    print(
        "max plastic multiplier : {:.6f}  ({:.1f} % of exhaustion)".format(
            multiplier.max(), 100.0 * multiplier.max() / panel.exhaustionMultiplier
        )
    )
    print("hottest grid point     : ({:.2f}, {:.2f})".format(*coordinates[hottest][:2]))
    print("plastic fraction       : {:.3f}".format(band["fraction"]))
    print("zone centre            : ({:.2f}, {:.2f})".format(*band["centre"]))
    print(
        "zone length x width    : {:.3f} x {:.3f}   (internal length {:.3f})".format(
            band["length"], band["width"], panel.internalLength
        )
    )
    print("zone inclination       : {:.1f} deg".format(band["angle"]))

    return dict(
        peakLoad=force[peak],
        finalLoad=force[-1],
        maxMultiplier=multiplier.max(),
        hottest=coordinates[hottest][:2],
        **band,
    )


def convergenceOfWidth(spacings, widths) -> dict:
    """Fit ``w = w_inf + C h^p`` to three widths and report the order and the limit.

    Three grids are exactly enough to solve for the three unknowns, so this is a fit with no
    freedom left -- which is the point: it turns "the width looks grid independent" into a number.
    A width that were set by the grid rather than by the internal length would extrapolate to
    zero, and a genuinely regularised one to a finite multiple of the internal length.

    Parameters
    ----------
    spacings
        The three grid spacings, decreasing.
    widths
        The corresponding widths.

    Returns
    -------
    dict
        The observed order and the extrapolated width, both ``nan`` if the fit is not meaningful.
    """

    (h1, h2, h3), (w1, w2, w3) = spacings, widths

    firstDifference = w1 - w2
    secondDifference = w2 - w3

    if not firstDifference * secondDifference > 0.0:
        # not monotone, so there is no order to speak of
        return dict(order=np.nan, extrapolated=np.nan)

    # solve (h1^p - h2^p) / (h2^p - h3^p) = firstDifference / secondDifference for p
    target = firstDifference / secondDifference

    def residual(order):
        return (h1**order - h2**order) / (h2**order - h3**order) - target

    orders = np.linspace(0.2, 6.0, 2000)
    residuals = np.array([residual(order) for order in orders])

    signChanges = np.nonzero(np.diff(np.sign(residuals)))[0]

    if signChanges.size == 0:
        return dict(order=np.nan, extrapolated=np.nan)

    order = float(
        np.interp(0.0, residuals[signChanges[0] : signChanges[0] + 2], orders[signChanges[0] : signChanges[0] + 2])
    )

    coefficient = firstDifference / (h1**order - h2**order)

    return dict(order=order, extrapolated=float(w3 - coefficient * h3**order))


def main():
    panel = defaultPanel

    print("internal length      : {:.3f}".format(panel.internalLength))
    print("weak corner radius   : {:.3f}".format(panel.weakCornerRadius))
    print("exhaustion at kappa  : {:.4f}".format(panel.exhaustionMultiplier))
    print("onset of yielding at : {:.4f}".format(abs(panel.shorteningAtYielding())))

    grids = [(20, 40), (30, 60), (40, 80)]
    results = []

    for nCells in grids:
        model, fieldOutputs, grid = run(nCells=nCells, panel=panel, verbose=False)

        results.append(report("{:}x{:} cells".format(*nCells), model, fieldOutputs, panel))

        plot("plasticMultiplier_{:}x{:}.png".format(*nCells), model, fieldOutputs, grid, panel)

    print()
    print("mesh study: the width of the localized zone has to be set by the internal length")
    print(
        "  {:>8}  {:>7}  {:>10}  {:>10}  {:>8}  {:>7}  {:>7}  {:>16}".format(
            "cells", "h/l", "peak", "final", "width", "angle", "frac", "hottest"
        )
    )
    for nCells, result in zip(grids, results):
        print(
            "  {:>8}  {:7.2f}  {:10.4f}  {:10.4f}  {:8.3f}  {:7.1f}  {:7.3f}  ({:5.1f},{:6.1f})".format(
                "{:}x{:}".format(*nCells),
                panel.width / nCells[0] / panel.internalLength,
                result["peakLoad"],
                result["finalLoad"],
                result["width"],
                result["angle"],
                result["fraction"],
                *result["hottest"],
            )
        )

    convergence = convergenceOfWidth(
        [panel.width / nCells[0] for nCells in grids], [result["width"] for result in results]
    )

    print()
    print("observed order of the width  : {:.2f}".format(convergence["order"]))
    print(
        "extrapolated width           : {:.3f}  = {:.2f} l".format(
            convergence["extrapolated"], convergence["extrapolated"] / panel.internalLength
        )
    )
    print(
        "spread of the peak load      : {:.2f} %".format(
            100.0
            * (max(r["peakLoad"] for r in results) - min(r["peakLoad"] for r in results))
            / min(r["peakLoad"] for r in results)
        )
    )


if __name__ == "__main__":
    main()
