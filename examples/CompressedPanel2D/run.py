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
from edelweissfe.utils.exceptions import StepFailed

from edelweissfd.drivers.pythonscriptedsimulation import FDSimulation
from edelweissfd.stencils.gradientplasticitystencil import GradientPlasticityStencil


@dataclass(frozen=True)
class Panel:
    """The panel, its material and its imperfection.

    Everything the simulation needs beyond the grid resolution and the loading. Grouped so that
    the parameter study in :func:`main` can vary it without touching module state.
    """

    #: The width, the height and the out of plane thickness of the panel. The thickness only
    #: enters as a factor on the reaction force, and hence on its normalisation by ``fy0 b t``.
    width: float = 60.0
    height: float = 120.0
    thickness: float = 1.0

    shearModulus: float = 4000
    poissonsRatio: float = 0.49
    youngsModulus: float = 2.0 * shearModulus * (1.0 + poissonsRatio)

    #: The initial yield strength.
    yieldStrength: float = 100.0

    #: The softening modulus. Negative, otherwise nothing localises.
    hardeningModulus: float = -400.0

    #: The internal length of the regularisation. The gradient parameter follows from it as
    #: ``g = |H| l^2``. Six is a compromise: the localized zone comes out a few times ``l`` wide,
    #: so a much larger value fills the whole 60 wide panel and nothing recognisable localizes,
    #: while a much smaller one needs a grid fine enough to resolve it and the cost of the mesh
    #: study explodes.
    gradientParameter: float = 3600
    internalLength: float = np.sqrt(gradientParameter / abs(hardeningModulus))

    #: The yield strength of the weak corner, as a fraction of the initial one. A shallow
    #: imperfection does not win against the rest of the panel: at 0.9 the plastic zone comes out
    #: symmetric about the vertical centre line and merely sits at the bottom edge, at 0.8 it is
    #: anchored at the corner. Going deeper still, to 0.7, localizes so sharply that Newton gives
    #: up before the zone is fully developed.
    weakeningFactor: float = 0.9

    #: Where the centre of the weakened region sits. ``None`` puts it at the origin, i.e. the
    #: bottom left corner, where only a quarter of the disc lies inside the panel.
    #:
    #: Placing it at the centre of the panel is a materially different problem, not just a moved
    #: flaw: a spot on the vertical centre line leaves the left-right symmetry of the panel intact,
    #: so the band running up-right and the band running up-left are equally favourable and the
    #: solution has to pick one. The corner spot breaks that symmetry outright. It also detaches the
    #: band from the loaded edges, so the inclination is no longer influenced by the platens.
    weakSpotCentre: tuple = None

    #: The radius of the weakened region.
    weakCornerRadius: float = 12.0

    #: Which complementarity formulation of ``GRADIENTVONMISES`` to use: zero selects Marmot's
    #: ``standard`` return mapping, anything else its smooth ``fischer_burmeister`` variant, see
    #: ``modules/materials/GradientVonMises/src/GradientVonMises.cpp`` where the constructor reads
    #: property five.
    #:
    #: Fischer-Burmeister is the default because the standard one **cannot solve this problem**.
    #: Measured: with ``implementation = 0.0`` the Newton-Raphson solver fails at a step progress of
    #: 0.344828, which is exactly ``1 / shorteningFactor``, i.e. precisely the closed form onset of
    #: yielding, and it fails at every increment size down to 1e-9. Three findings, in order:
    #:
    #: 1. The elastic branch of ``GradientVonMises::computeStressStandard`` is internally
    #:    inconsistent: it reports the yield function as identically zero -- so its true derivatives
    #:    are zero -- while returning ``dF_ddStrain = dF_dStress^T C`` and ``dF_dKappa = E``. The
    #:    numerical tangent checker puts the relative deviation at 1.0; reporting ``f = E dLambda``
    #:    with zero strain and Laplacian derivatives instead brings it to 3e-10.
    #: 2. Fixing that is not enough. The panel still fails at the same progress, and both branches
    #:    are then individually consistent (elastic 3e-10, plastic 5e-9), so the obstacle is the
    #:    switch between them rather than either branch.
    #: 3. The two branches carry multiplier diagonals of *opposite sign* -- ``+E`` elastic against a
    #:    negative value plastic. A grid point flipping between them from one Newton iteration to
    #:    the next reverses the sign of its diagonal, so the iteration chatters instead of settling.
    #:    The smooth variant has no such switch: through the same transition its diagonal runs
    #:    monotonically from -1.0e4 to -3.3e4, never changing sign.
    #:
    #: The standard implementation is not wrong in general -- for a spatially *homogeneous* state,
    #: where every material point switches at the same instant, the two agree to six digits and both
    #: land exactly on ``fy0 + H kappa``. It is unsuitable here because a localising panel is by
    #: construction a mixed elastic and plastic field.
    implementation: float = 1.0

    density: float = 2.4e-9
    nonlocalViscosity: float = 0.0

    @property
    def weakSpot(self) -> np.ndarray:
        """The centre of the weakened region."""

        return np.zeros(2) if self.weakSpotCentre is None else np.asarray(self.weakSpotCentre, dtype=float)

    @property
    def weakSpotArea(self) -> float:
        """The exact area of the weakened region, clipped to the panel.

        A spot at a corner contributes a quarter of its disc, one on an edge a half, one in the
        interior all of it -- which matters when comparing placements, since the same radius then
        weakens four times as much material in the middle as in a corner.
        """

        centre = self.weakSpot
        radius = self.weakCornerRadius

        fractionX = 0.5 if centre[0] <= 0.0 or centre[0] >= self.width else 1.0
        fractionY = 0.5 if centre[1] <= 0.0 or centre[1] >= self.height else 1.0

        return fractionX * fractionY * np.pi * radius**2 * self.thickness

    def isWeak(self, coordinates) -> bool:
        """Whether a point lies inside the weakened region."""

        return bool(np.linalg.norm(np.asarray(coordinates)[:2] - self.weakSpot) <= self.weakCornerRadius)

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

    @property
    def referenceLoad(self) -> float:
        """The load the reaction force is normalised by, ``fy0 b t``.

        The reaction force divided by this is the mean axial stress over the cross section in
        units of the initial yield strength, so a normalised load of one means the whole cross
        section carrying exactly ``fy0``.

        The peak comes out *above* one, and by a predictable amount: a uniaxial stress state
        reaches the Mises surface only at ``sigma = fy0 / misesFactor``, so a panel of the strong
        material alone would peak at ``1 / misesFactor``. That makes the normalised peak an
        independent check on the whole computation -- it has to sit just below that value, short
        of it because the weak corner has already yielded and softened.
        """

        return self.yieldStrength * self.width * self.thickness

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

#: How far past the closed form onset of yielding the loading goes, as a multiple of it. Under
#: direct displacement control this is the whole loading history, so it has to reach well past the
#: peak for the softening branch to be visible.
#:
#: It also has to be reachable on *every* grid of the mesh study, because comparing the band on
#: grids that stopped at different end shortenings is not a mesh study -- the band keeps narrowing
#: as it develops, so that would confound refinement with loading. Four was too far: with the
#: volumetric averaging in place the band localises sharply enough that 30 x 60 cells stalls at
#: 96 percent of it, close to exhausting the material inside the band.
shorteningFactor = 2.9

#: How far past the closed form onset of yielding the *displacement controlled* part goes when the
#: arc length solver takes over afterwards. It only has to pass yielding, so that the plastic
#: multiplier at the weak corner is non-zero and can serve as the control quantity: while
#: everything is still elastic it is identically zero, a trial load increment produces no change in
#: it at all, and the arc length controller has nothing to divide by.
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
                panel.implementation,
                panel.density,
                panel.nonlocalViscosity,
            ],
            baseClass="gradientPlasticityHypoElastic",
        )

    strongMaterial = createMaterial(panel.yieldStrength)
    weakMaterial = createMaterial(panel.weakeningFactor * panel.yieldStrength)

    def materialAt(coordinates):
        return weakMaterial if panel.isWeak(coordinates) else strongMaterial

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
        (node for node in grid.nodes.values() if panel.isWeak(node.coordinates)),
        key=lambda node: np.linalg.norm(node.coordinates[:2] - panel.weakSpot),
    )

    if len(inWeakCorner) < 2:
        raise ValueError("The grid is too coarse to place two control points inside the weak corner.")

    return inWeakCorner[:2]


def run(
    nCells=(20, 40),
    panel: Panel = defaultPanel,
    control: str = "displacement",
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
        ``displacement`` drives the end shortening directly, which is what this softening law
        allows: the response has no snap back, so the end shortening stays a monotone parameter
        along the whole equilibrium path. ``indirect`` instead drives the plastic multiplier of the
        weak corner with the arc length solver, which is only needed for softening sharp enough to
        snap back -- with a stiffer ``hardeningModulus`` the arc length parameter comes out at
        essentially zero, i.e. the end shortening freezes and displacement control cannot follow.
    verbose
        Be verbose during the simulation.

    Returns
    -------
    tuple
        The model, the field output controller, the grid and the stencils.
    """

    sim = FDSimulation(domainSize=2, name="CompressedPanel2D", verbose=verbose)

    grid = sim.createStructuredGrid(lengths=[panel.width, panel.height], nGridPoints=[nCells[0] + 1, nCells[1] + 1])

    stencils = sim.assignStencils(
        GradientPlasticityStencil,
        grid,
        material=createMaterialFactory(sim, panel),
        stressState="plane strain",
        thickness=panel.thickness,
    )

    sim.createSolver("NIST")

    if control == "indirect":
        sim.createSolver("NISTPArcLength")

    # under displacement control this single step traces the whole curve, so it needs enough
    # increments to resolve the peak and the softening branch rather than just to reach yielding
    maxInc = 5e-2 if control == "indirect" else 1e-2

    firstStep = sim.createStep(solver="NIST", stepLength=1.0, maxInc=maxInc, minInc=1e-9, maxNumInc=3000, maxIter=25)

    # Frictionless platens: only the vertical displacement is prescribed on the loaded faces, the
    # panel is free to slide sideways, and a single grid point is pinned against rigid body
    # translation. Every bottom grid point being held vertically already rules out rotation.
    firstStep.addDirichlet("bottom", grid.nodeSets["bottom"], "displacement", {1: 0.0})
    firstStep.addDirichlet("pin", grid.nodeSets["rightBottom"], "displacement", {0: 0.0})

    if control == "displacement":
        firstStep.addDirichlet(
            "top", grid.nodeSets["top"], "displacement", {1: shorteningFactor * panel.shorteningAtYielding()}
        )
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

    return model, fieldOutputs, grid, stencils


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

    .. warning::
        One ellipse can only describe **one** band. For a field that is symmetric about the vertical
        centre line -- which is what a weak spot on that line produces, see :func:`antisymmetry` --
        both conjugate shear directions develop equally and form an X. The major axis of the fitted
        ellipse is then the vertical bisector of that X, so this reports an inclination of 90
        degrees and a width spanning the whole pattern, neither of which is a band. Always read the
        inclination together with :func:`antisymmetry`.

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


def reportWeakRegion(stencils, panel: Panel = defaultPanel) -> dict:
    """Verify that the stencils inside the weak corner really carry the weakened material.

    Reads the yield strength back off each stencil's own material instance rather than trusting
    that the factory was wired up correctly. Worth doing for two reasons: the material is selected
    by the *cell centre*, which for this stencil is the average over its cell corners only and not
    over its whole molecule -- the Laplacian reaches a grid point further, and a molecule average
    would be off exactly at the boundary cells, which is where the weak corner sits. And every
    stencil now owns its own material instance, so a factory returning a shared one would no
    longer be caught by a difference in results.

    Parameters
    ----------
    stencils
        The stencils covering the grid.
    panel
        The panel, for the weakened region.

    Returns
    -------
    dict
        The number of weak stencils, their area fraction and the number of misassigned ones.
    """

    weakYieldStrength = panel.weakeningFactor * panel.yieldStrength

    weak, misassigned = [], []

    for stencil in stencils:
        # index 2 of Marmot's GRADIENTVONMISES property vector is the initial yield strength
        yieldStrength = float(stencil._material.materialProperties[2])
        centre = stencil.getCoordinatesAtCenter()

        shouldBeWeak = panel.isWeak(centre)
        isWeak = yieldStrength < panel.yieldStrength

        if isWeak:
            weak.append(centre)

        if isWeak != shouldBeWeak or (isWeak and yieldStrength != weakYieldStrength):
            misassigned.append((centre, yieldStrength, shouldBeWeak))

    weakArea = sum(
        stencil._totalVolume
        for stencil in stencils
        if float(stencil._material.materialProperties[2]) < panel.yieldStrength
    )
    exactArea = panel.weakSpotArea

    print(
        "weak region            : {:} of {:} stencils, fy {:.3f} vs {:.3f}, "
        "area {:.1f} vs {:.1f} exact ({:+.1f} %), misassigned {:}".format(
            len(weak),
            len(stencils),
            weakYieldStrength,
            panel.yieldStrength,
            weakArea,
            exactArea,
            100.0 * (weakArea / exactArea - 1.0),
            len(misassigned),
        )
    )

    for centre, yieldStrength, shouldBeWeak in misassigned:
        print(
            "  MISASSIGNED at ({:.2f}, {:.2f}): fy = {:.3f}, expected {:}".format(
                centre[0], centre[1], yieldStrength, "weak" if shouldBeWeak else "strong"
            )
        )

    return dict(nWeak=len(weak), weakArea=weakArea, exactArea=exactArea, nMisassigned=len(misassigned))


def onGrid(model, grid, field: str, values: np.ndarray) -> np.ndarray:
    """Sort a flat nodal field onto the tensor product grid.

    Done by asking the grid for each node's index rather than by reshaping, so the result does not
    depend on the order in which the node field happens to list its grid points.

    Parameters
    ----------
    model
        The model.
    grid
        The grid of the panel.
    field
        The name of the node field the values belong to.
    values
        The values, one row per grid point.

    Returns
    -------
    np.ndarray
        The values with the grid shape leading.
    """

    values = np.asarray(values)
    shape = tuple(int(n) for n in grid.shape)

    arranged = np.empty(shape + values.shape[1:])

    for node, value in zip(model.nodeFields[field].nodes, values):
        arranged[tuple(int(i) for i in grid.gridIndexOf(node))] = value

    return arranged


def antisymmetry(model, grid, multiplier: np.ndarray) -> float:
    """How one sided the plastic field is about the vertical centre line of the panel.

    Zero means perfectly symmetric, so both conjugate shear directions developed equally and the
    pattern is an X rather than a band; one means the field lives entirely on one side, so a single
    band was selected. This is the measure that distinguishes the two regimes, because
    :func:`bandGeometry` cannot: it fits one ellipse and reports the bisector of a symmetric X as a
    vertical band.

    A weak spot on the centre line leaves the panel's left-right symmetry intact and the measure
    comes out at machine precision, a few times 1e-14. A spot in a corner breaks that symmetry and
    it comes out at a few tenths.

    Parameters
    ----------
    model
        The model.
    grid
        The grid of the panel.
    multiplier
        The plastic multiplier per grid point.

    Returns
    -------
    float
        The ratio of the antisymmetric to the symmetric part, in the L1 sense.
    """

    field = onGrid(model, grid, "plastic multiplier", multiplier)
    mirrored = field[::-1, :]

    symmetric = np.abs(field + mirrored).sum()

    return float(np.abs(field - mirrored).sum() / symmetric) if symmetric > 0.0 else float("nan")


def plot(fileName: str, model, fieldOutputs, grid, panel: Panel = defaultPanel, magnification: float = None):
    """Draw the plastic multiplier, the deformed shape and the load displacement curve.

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
        The panel, for the weak corner outline and the normalisation of the curve.
    magnification
        The factor the displacements are exaggerated by in the deformed shape. Defaults to
        whatever makes the largest displacement a tenth of the panel height, because at true
        scale the shortening is around one percent of the height and the kink across the shear
        band would be invisible. The factor used is stated in the panel title.

    Notes
    -----
    The displacement magnitude is measured from the grid point pinned laterally, which is the
    *bottom right* corner. Frictionless platens let the panel expand sideways, so a large part of
    ``|u|`` is that lateral expansion relative to the pin rather than anything local: it grows
    smoothly away from the pinned corner and makes the deformed shape look as though the panel
    were leaning. Read the localisation off the plastic multiplier panel, and the deformed shape
    for the kink across the band, not for the overall tilt.
    """

    import matplotlib.pyplot as plt

    coordinates, multiplier = plasticZone(model, fieldOutputs)
    displacement = np.asarray(fieldOutputs.fieldOutputs["displacement"].getLastResult())

    shape = tuple(int(n) for n in grid.shape)

    x = onGrid(model, grid, "plastic multiplier", coordinates[:, 0])
    y = onGrid(model, grid, "plastic multiplier", coordinates[:, 1])
    field = onGrid(model, grid, "plastic multiplier", multiplier)

    u = onGrid(model, grid, "displacement", displacement[:, :2])
    magnitude = np.linalg.norm(u, axis=-1)

    if magnification is None:
        largest = float(magnitude.max())
        magnification = 0.1 * panel.height / largest if largest > 0.0 else 1.0

    force = np.abs(np.array(fieldOutputs.fieldOutputs["reactionForce"].getResultHistory()).flatten())
    shortening = np.abs(np.array(fieldOutputs.fieldOutputs["shortening"].getResultHistory()).flatten())

    figure, axes = plt.subplots(1, 3, figsize=(14.0, 5.0))

    contour = axes[0].contourf(x, y, np.clip(field, 0.0, None), levels=24, cmap="inferno")
    figure.colorbar(contour, ax=axes[0], label="plastic multiplier")

    axes[0].add_patch(
        plt.Circle(tuple(panel.weakSpot), panel.weakCornerRadius, fill=False, color="deepskyblue", lw=1.5)
    )

    axes[0].set_aspect("equal")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title("{:} x {:} cells,  l = {:.2f}".format(shape[0] - 1, shape[1] - 1, panel.internalLength))

    # the deformed shape, i.e. the same contour drawn on the displaced grid points
    deformed = axes[1].contourf(
        x + magnification * u[..., 0], y + magnification * u[..., 1], magnitude, levels=24, cmap="viridis"
    )
    figure.colorbar(deformed, ax=axes[1], label="|u|")

    # the undeformed outline for reference, so the magnification can be read off the figure
    axes[1].plot(
        [0.0, panel.width, panel.width, 0.0, 0.0],
        [0.0, 0.0, panel.height, panel.height, 0.0],
        color="grey",
        ls="--",
        lw=1.0,
    )

    axes[1].set_aspect("equal")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_title("deformed shape, displacements x {:.0f}".format(magnification))

    axes[2].plot(shortening / panel.height, force / panel.referenceLoad, "-")
    axes[2].axvline(
        abs(panel.shorteningAtYielding()) / panel.height,
        color="grey",
        ls="--",
        lw=1.0,
        label="closed form onset",
    )
    axes[2].set_xlabel("$u\\,/\\,H$")
    axes[2].set_ylabel("$F\\,/\\,(f_{y0}\\,b\\,t)$")
    axes[2].set_title("load displacement")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    figure.tight_layout()
    figure.savefig(fileName, dpi=110)
    plt.close(figure)


def report(name: str, model, fieldOutputs, panel: Panel = defaultPanel, grid=None) -> dict:
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
    print(
        "peak load              : {:.4f}   F/(fy0 b t) = {:.4f}   (strong material alone: {:.4f})".format(
            force[peak], force[peak] / panel.referenceLoad, 1.0 / panel.misesFactor
        )
    )
    print(
        "shortening at peak     : {:.5f}   u/H = {:.5f}   (closed form onset u/H = {:.5f})".format(
            shortening[peak],
            shortening[peak] / panel.height,
            abs(panel.shorteningAtYielding()) / panel.height,
        )
    )
    print(
        "final load             : {:.4f}   F/(fy0 b t) = {:.4f}  ({:.1f} % of peak)".format(
            force[-1], force[-1] / panel.referenceLoad, 100.0 * force[-1] / force[peak]
        )
    )
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

    if grid is not None:
        measure = antisymmetry(model, grid, multiplier)
        print(
            "antisymmetry           : {:.3e}   ({:})".format(
                measure,
                (
                    "a symmetric X, so the inclination above is its bisector, not a band"
                    if measure < 1e-6
                    else "a single band was selected"
                ),
            )
        )

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
        The observed order and the extrapolated width, both ``nan`` if the widths do not converge,
        together with a ``diagnosis`` saying why. Refusing to report an order is the whole value of
        this function: three points can always be fitted by *something*, and a number attached to a
        sequence that is not converging would read as evidence of grid independence where there is
        none.
    """

    (h1, h2, h3), (w1, w2, w3) = spacings, widths

    firstDifference = w1 - w2
    secondDifference = w2 - w3

    notConverged = dict(order=np.nan, extrapolated=np.nan)

    if not firstDifference * secondDifference > 0.0:
        return dict(notConverged, diagnosis="the widths are not monotone, so there is no order to speak of")

    if abs(secondDifference) >= abs(firstDifference):
        return dict(
            notConverged,
            diagnosis=(
                "the change in width grows under refinement ({:+.3f} then {:+.3f}), i.e. the width is "
                "not settling. Expect this whenever the grid does not resolve the internal length: "
                "the regularisation cannot act below one cell, so the answer is still set by the grid"
            ).format(-firstDifference, -secondDifference),
        )

    # solve (h1^p - h2^p) / (h2^p - h3^p) = firstDifference / secondDifference for p
    target = firstDifference / secondDifference

    def residual(order):
        return (h1**order - h2**order) / (h2**order - h3**order) - target

    orders = np.linspace(0.2, 6.0, 2000)
    residuals = np.array([residual(order) for order in orders])

    signChanges = np.nonzero(np.diff(np.sign(residuals)))[0]

    if signChanges.size == 0:
        return dict(notConverged, diagnosis="no convergence order between 0.2 and 6 reproduces these widths")

    order = float(
        np.interp(0.0, residuals[signChanges[0] : signChanges[0] + 2], orders[signChanges[0] : signChanges[0] + 2])
    )

    coefficient = firstDifference / (h1**order - h2**order)
    extrapolated = float(w3 - coefficient * h3**order)

    # An order below one is below the formal order of the difference operators, and an
    # extrapolated width of zero or less is not a width at all. Either says the widths are still
    # falling too fast to have a limit in view: they are consistent with a band that keeps
    # narrowing under refinement, which is what an unresolved internal length looks like. Reporting
    # the fit anyway would dress that up as a convergence statement.
    if order < 1.0 or extrapolated <= 0.0:
        return dict(
            notConverged,
            diagnosis=(
                "the widths still fall steeply -- fitting them gives an order of {:.2f} and a limit "
                "of {:.2f}, i.e. below the formal order of the operators and not a positive width. "
                "The band is still narrowing with every refinement, so its width is not yet set by "
                "the internal length alone"
            ).format(order, extrapolated),
        )

    return dict(order=order, extrapolated=extrapolated, diagnosis="")


def main():
    panel = defaultPanel

    print("internal length      : {:.3f}".format(panel.internalLength))
    print(
        "weak spot            : centre ({:.1f}, {:.1f}), radius {:.3f}, area {:.1f}".format(
            *panel.weakSpot, panel.weakCornerRadius, panel.weakSpotArea
        )
    )
    print("exhaustion at kappa  : {:.4f}".format(panel.exhaustionMultiplier))
    print("onset of yielding at : {:.4f}".format(abs(panel.shorteningAtYielding())))

    grids = [(20, 40), (30, 60), (40, 80)]
    results = []

    completed = []

    for nCells in grids:
        # a grid that gives up must not take the whole study with it: the ones that did converge
        # are still worth reporting, and which grid failed is itself the useful information
        try:
            model, fieldOutputs, grid, stencils = run(nCells=nCells, panel=panel, verbose=True)
        except StepFailed:
            print()
            print("=== {:}x{:} cells: FAILED to reach the end shortening ===".format(*nCells))
            continue

        results.append(report("{:}x{:} cells".format(*nCells), model, fieldOutputs, panel, grid))
        completed.append(nCells)

        reportWeakRegion(stencils, panel)

        plot("plasticMultiplier_{:}x{:}.png".format(*nCells), model, fieldOutputs, grid, panel)

    grids = completed

    if not results:
        print("no grid completed, nothing to compare")
        return

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

    if len(results) < 3:
        print()
        print("only {:} of 3 grids completed, so there is nothing to extrapolate from".format(len(results)))
        return

    convergence = convergenceOfWidth(
        [panel.width / nCells[0] for nCells in grids], [result["width"] for result in results]
    )

    print()
    print(
        "spread of the peak load      : {:.2f} %".format(
            100.0
            * (max(r["peakLoad"] for r in results) - min(r["peakLoad"] for r in results))
            / min(r["peakLoad"] for r in results)
        )
    )

    if convergence["diagnosis"]:
        print("width does NOT converge      : {:}".format(convergence["diagnosis"]))
        print(
            "                               here h/l is {:}, and resolving l takes roughly h/l <= 0.5".format(
                ", ".join("{:.2f}".format(panel.width / nCells[0] / panel.internalLength) for nCells in grids)
            )
        )
    else:
        print("observed order of the width  : {:.2f}".format(convergence["order"]))
        print(
            "extrapolated width           : {:.3f}  = {:.2f} l".format(
                convergence["extrapolated"], convergence["extrapolated"] / panel.internalLength
            )
        )


if __name__ == "__main__":
    main()
