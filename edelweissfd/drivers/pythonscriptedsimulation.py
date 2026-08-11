#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  ---------------------------------------------------------------------
#
#  _____    _      _              _         _____ _____ ____
# | ____|__| | ___| |_      _____(_)___ ___|  ___|  ___|  _ \
# |  _| / _` |/ _ \ \ \ /\ / / _ \ / __/ __| |_  | |_  | | | |
# | |__| (_| |  __/ |\ V  V /  __/ \__ \__ \  _| |  _| | |_| |
# |_____\__,_|\___|_| \_/\_/ \___|_|___/___/_|   |_|   |____/
#
#
#  Unit of Strength of Materials and Structural Analysis
#  University of Innsbruck,
#  2017 - today
#
#  Alexander Dummer alexander.dummer@uibk.ac.at
#
#  This file is part of EdelweissFD.
#
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#
#  The full text of the license can be found in the file LICENSE.md at
#  the top level directory of EdelweissFD.
#  ---------------------------------------------------------------------

"""
The Python scripting interface of EdelweissFD.

:class:`FDSimulation` is the finite difference counterpart of
:func:`edelweissfe.drivers.inputfiledrivensimulation.finiteElementSimulation`. It follows
the very same sequence -- build the model, prepare the variables and fields, create the
field outputs and output managers, load the solver configuration, then dequeue and solve
the steps -- but it constructs the objects directly instead of parsing an input file.

A simulation is written as an ordinary Python script::

    sim = FDSimulation(domainSize=1, name="TensionBar1D")

    grid = sim.createStructuredGrid(lengths=[100.0], nGridPoints=[101])
    material = sim.createMaterial("LinearElastic", [20000.0, 0.2])

    sim.assignStencils(DisplacementStencil, grid, material=material,
                       stressState="uniaxial stress")

    step = sim.createStep(stepLength=1.0, maxInc=0.5)
    step.addDirichlet("clamped", grid.nodeSets["left"], "displacement", {0: 0.0})
    step.addNeumann("pulled", grid, "right", "displacement", [10.0])

    sim.addNodeFieldOutput("displacement", "displacement", "U",
                           nodeSet=grid.nodeSets["all"])

    model, fieldOutputs = sim.run()
"""

from time import time as getCurrentTime

import numpy as np
from edelweissfe.config.configurator import loadConfiguration
from edelweissfe.config.materiallibrary import getMaterialClass
from edelweissfe.config.outputmanagers import getOutputManagerFactoryByName
from edelweissfe.config.solvers import getSolverByName
from edelweissfe.journal.journal import Journal
from edelweissfe.points.node import Node
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.steps.stepmanager import (
    StepActionDefinition,
    StepDefinition,
    StepManager,
)
from edelweissfe.utils.exceptions import StepFailed
from edelweissfe.utils.fieldoutput import FieldOutputController

from edelweissfd.grids.structuredgrid import StructuredGrid, boundarySetNames
from edelweissfd.models.fdmodel import FDModel
from edelweissfd.sets.stencilset import StencilSet
from edelweissfd.stepactions import tractionnodalload

#: The component keys every prescribed-value step action of EdelweissFE reads unconditionally.
componentDefaults = {str(component): None for component in range(1, 7)}

#: The defaults of a Dirichlet step action definition; every key is read unconditionally
#: by :mod:`edelweissfe.stepactions.dirichlet`.
dirichletDefaults = {
    "components": None,
    "analyticalField": None,
    "f(t)": None,
    **componentDefaults,
}

#: The defaults of a nodal force step action definition, cf.
#: :mod:`edelweissfe.stepactions.nodeforces`, which knows no analytical field.
nodeForcesDefaults = {
    "components": None,
    "f(t)": None,
    **componentDefaults,
}

#: The default options of an adaptive step.
stepDefaults = {
    "stepLength": 1.0,
    "maxInc": 1.0,
    "minInc": 1e-5,
    "maxNumInc": 1000,
    "maxIter": 25,
    "criticalIter": 5,
}


class FDStep:
    """A single step of a finite difference simulation.

    Instances are created by :meth:`FDSimulation.createStep`; do not construct them
    directly.

    Parameters
    ----------
    number
        The number of this step.
    solverName
        The name of the solver to be used.
    options
        The options of the adaptive step, e.g. ``stepLength`` or ``maxInc``.
    """

    def __init__(self, number: int, solverName: str, options: dict):
        self.number = number
        self.solverName = solverName
        self.options = dict(stepDefaults)
        self.options.update(options)

        self._actionDefinitions = []
        self._prebuiltActions = []

        #: Stencil sets which have to exist in ``model.elementSets`` before the step actions
        #: of this step are created, since a body force refers to its set by name.
        self._stencilSetsToRegister = []

    def addDirichlet(
        self,
        name: str,
        nodeSet: NodeSet,
        field: str,
        components: dict,
        amplitudeExpression: str = None,
        analyticalField: str = None,
    ):
        """Prescribe field components on a node set.

        .. important::
            Following EdelweissFE, the prescribed values are the **increments applied over
            this step**, not absolute targets. A value of ``0.2`` in the first step and again
            ``0.2`` in the second one therefore ends at ``0.4``, and unloading back to zero
            takes a value of ``-0.2``. A boundary condition which is not mentioned again in a
            later step holds the value it has reached.

        Parameters
        ----------
        name
            The name of the boundary condition. Reusing a name in a later step updates that
            boundary condition instead of creating a new one.
        nodeSet
            The node set the boundary condition acts on.
        field
            The field the boundary condition acts on, e.g. ``displacement``.
        components
            The increments applied over this step, keyed by the zero based component index.
        amplitudeExpression
            An optional expression in the step progress ``t``, evaluated on ``[0...1]``.
        analyticalField
            The name of an analytical field scaling the prescribed values.
        """

        action = dict(dirichletDefaults)
        action["name"] = name
        action["nSet"] = nodeSet.name
        action["field"] = field
        action["f(t)"] = amplitudeExpression
        action["analyticalField"] = analyticalField

        for component, value in components.items():
            action[str(int(component) + 1)] = value

        self._actionDefinitions.append(StepActionDefinition(name, "dirichlet", action))

        return self

    def addNodeForces(
        self,
        name: str,
        nodeSet: NodeSet,
        field: str,
        components: dict,
        amplitudeExpression: str = None,
    ):
        """Apply the same nodal force to every node of a node set.

        Like :meth:`addDirichlet`, the values are the increments applied over this step, and a
        load which is not mentioned again in a later step is held constant.

        Parameters
        ----------
        name
            The name of the load.
        nodeSet
            The node set the load acts on.
        field
            The field the load acts on.
        components
            The force increments applied over this step, keyed by the zero based component index.
        amplitudeExpression
            An optional expression in the step progress ``t``, evaluated on ``[0...1]``.
        """

        action = dict(nodeForcesDefaults)
        action["name"] = name
        action["nSet"] = nodeSet.name
        action["field"] = field
        action["f(t)"] = amplitudeExpression

        for component, value in components.items():
            action[str(int(component) + 1)] = value

        self._actionDefinitions.append(StepActionDefinition(name, "nodeforces", action))

        return self

    def addNeumann(
        self,
        name: str,
        grid: StructuredGrid,
        boundary: str,
        field: str,
        traction,
        amplitudeExpression: str = None,
    ):
        """Apply a traction on a boundary plane of a grid.

        The traction is converted into consistent nodal forces using the tributary areas of
        the grid points, see
        :meth:`~edelweissfd.grids.structuredgrid.StructuredGrid.boundaryNodeAreas`. In contrast
        to a fixed force per grid point, this is mesh independent: refining the grid keeps the
        total load the same.

        Like :meth:`addDirichlet`, the value is the increment applied over this step.

        Parameters
        ----------
        name
            The name of the load.
        grid
            The grid whose boundary is loaded.
        boundary
            The name of the boundary node set, e.g. ``right`` or ``top``.
        field
            The field the load acts on.
        traction
            The traction increment applied over this step, one entry per field component.
        amplitudeExpression
            An optional expression in the step progress ``t``, evaluated on ``[0...1]``.
        """

        direction, index = _locateBoundary(grid, boundary)

        nodeSet = grid.nodeSets[boundary]
        areas = grid.boundaryNodeAreas(direction, index)

        traction = np.asarray(traction, dtype=float)

        nodalLoads = np.array([areas[node] * traction for node in nodeSet])

        self._prebuiltActions.append(
            (
                tractionnodalload.stepActionModule,
                name,
                tractionnodalload.TractionNodalLoad(name, field, nodeSet, nodalLoads, amplitudeExpression),
            )
        )

        return self

    def addBodyForce(
        self,
        name: str,
        stencils,
        forceVector,
        amplitudeExpression: str = None,
    ):
        """Apply a force per unit volume, e.g. gravity, to a set of stencils.

        Like :meth:`addDirichlet`, the value is the increment applied over this step.

        Parameters
        ----------
        name
            The name of the load. It doubles as the name of the stencil set registered in the
            model.
        stencils
            The stencils the load acts on, as returned by
            :meth:`FDSimulation.assignStencils`.
        forceVector
            The increment of the force per unit volume, one entry per spatial direction.
        amplitudeExpression
            An optional expression in the step progress ``t``, evaluated on ``[0...1]``.
        """

        forceVector = np.asarray(forceVector, dtype=float)

        action = {
            "name": name,
            "elSet": name,
            # the step action parses this with numpy.fromstring, so plain decimals only
            "forceVector": ", ".join("{:.17g}".format(float(component)) for component in forceVector),
            "f(t)": amplitudeExpression,
            "delta": None,
        }

        self._stencilSetsToRegister.append((name, list(stencils)))
        self._actionDefinitions.append(StepActionDefinition(name, "bodyforce", action))

        return self

    def addIndirectControl(
        self,
        firstNode: Node,
        firstWeights,
        secondNode: Node,
        secondWeights,
        finalValue: float,
        field: str = "displacement",
        name: str = "indirectcontrol",
    ):
        """Control the step by a prescribed relative displacement instead of by a load or a
        boundary displacement.

        This is what allows a softening structure to be followed past a snap-back, where the
        equilibrium path turns back in the controlled displacement and plain displacement
        control has no solution. It requires the arc-length solver, i.e.
        ``createSolver("NISTPArcLength")``, and switches its controller to
        ``indirectcontrol``.

        The controlled quantity is
        ``firstWeights . u(firstNode) + secondWeights . u(secondNode)``, which is driven from
        zero to ``finalValue`` over the step; the classical choice is a pair of nodes on
        either side of the localizing zone with opposite weights, i.e. their relative opening.

        Parameters
        ----------
        firstNode
            The first control grid point.
        firstWeights
            The weights of the first control grid point, one per field component.
        secondNode
            The second control grid point.
        secondWeights
            The weights of the second control grid point, one per field component.
        finalValue
            The value the controlled quantity reaches at the end of the step.
        field
            The field the control acts on.
        name
            The name of the step action.
        """

        action = {
            "name": name,
            "dof1": 'model.nodes[{:}].fields["{:}"]'.format(firstNode.label, field),
            "dof2": 'model.nodes[{:}].fields["{:}"]'.format(secondNode.label, field),
            "cVector1": repr(list(np.asarray(firstWeights, dtype=float))),
            "cVector2": repr(list(np.asarray(secondWeights, dtype=float))),
            "L": float(finalValue),
            "exportCVector": "",
            "absolute": True,
        }

        self._actionDefinitions.append(StepActionDefinition(name, "indirectcontrol", action))

        self.addSolverOptions("NISTArcLength", arcLengthController="indirectcontrol", stopCondition=None)

        return self

    def addSolverOptions(self, category: str, **options):
        """Set solver specific options for this step.

        Parameters
        ----------
        category
            The option category, e.g. ``NISTSolver``.
        options
            The options, e.g. ``extrapolation="off"``.
        """

        action = dict(options)
        action["name"] = category
        action["category"] = category

        self._actionDefinitions.append(StepActionDefinition(category, "options", action))

        return self

    def toStepDefinition(self) -> StepDefinition:
        """The :class:`~edelweissfe.steps.stepmanager.StepDefinition` of this step."""

        stepOptions = dict(self.options)
        stepOptions["solver"] = self.solverName

        return StepDefinition("adaptive", stepOptions, list(self._actionDefinitions))


def _locateBoundary(grid: StructuredGrid, boundary: str) -> tuple:
    """The direction normal to a named boundary of a grid and the grid index of that plane.

    Parameters
    ----------
    grid
        The grid.
    boundary
        The name of the boundary node set.

    Returns
    -------
    tuple
        The normal direction and the grid index along it.
    """

    for direction, (lowerName, upperName) in enumerate(boundarySetNames[: grid.nDim]):
        if boundary == lowerName:
            return direction, 0
        if boundary == upperName:
            return direction, grid.shape[direction] - 1

    validNames = [name for names in boundarySetNames[: grid.nDim] for name in names]

    raise ValueError("'{:}' is not a boundary of this grid; valid are {:}".format(boundary, ", ".join(validNames)))


class FDSimulation:
    """A finite difference simulation, set up and run from a Python script.

    Parameters
    ----------
    domainSize
        The spatial dimension of the simulation.
    name
        The name of the job.
    verbose
        Be verbose during the simulation.
    startTime
        The time the simulation starts at.
    """

    identification = "fdCore"

    def __init__(self, domainSize: int, name: str = "edelweissfd", verbose: bool = True, startTime: float = 0.0):
        self.name = name
        self.startTime = startTime

        self.journal = Journal(verbose=verbose)
        self.model = FDModel(domainSize)

        self.solvers = dict()
        self.steps = []

        self._nodeFieldOutputRequests = []
        self._stencilFieldOutputRequests = []
        self._outputManagerRequests = []

        self._nextStencilNumber = 1
        self._nextNodeLabel = 1

        # The default tolerances and solver options; solvers read them at construction, so
        # the configuration has to be in place before the first solver is created.
        self.jobInfo = loadConfiguration({"name": name, "domainSize": domainSize, "computationTime": 0.0})

    # -- model construction -------------------------------------------------------------

    def createStructuredGrid(self, lengths, nGridPoints, origin=None, name: str = None) -> StructuredGrid:
        """Create a structured grid and register it in the model.

        Parameters
        ----------
        lengths
            The extent of the grid per direction.
        nGridPoints
            The number of grid points per direction.
        origin
            The coordinates of the lower left grid point.
        name
            The name of the grid. Defaults to ``grid`` for the first grid.

        Returns
        -------
        StructuredGrid
            The grid.
        """

        if name is None:
            name = "grid" if not self.model.grids else "grid{:}".format(len(self.model.grids) + 1)

        grid = StructuredGrid(
            name,
            lengths,
            nGridPoints,
            origin=origin,
            firstNodeLabel=self._nextNodeLabel,
        )

        self._nextNodeLabel += len(grid.nodes)

        self.model.addGrid(grid)

        self.journal.message("Created {:}".format(grid), self.identification)

        return grid

    def createNodeSet(self, name: str, nodes) -> NodeSet:
        """Create a node set and register it in the model, so boundary conditions can refer to
        it.

        The grids already provide their boundary sets; this is for anything else, e.g. the
        grid points along a symmetry line or a single control point::

            centre = sim.createNodeSet("centre", [n for n in grid.nodes.values()
                                                  if abs(n.coordinates[1] - 50.0) < 1e-12])

        Parameters
        ----------
        name
            The name of the node set. It must not collide with an existing one.
        nodes
            The grid points of the set.

        Returns
        -------
        NodeSet
            The node set.
        """

        if name in self.model.nodeSets:
            raise ValueError("A node set named '{:}' already exists in this model.".format(name))

        nodeSet = NodeSet(name, list(nodes))

        self.model.nodeSets[name] = nodeSet

        return nodeSet

    def createMaterial(
        self,
        materialName: str,
        materialProperties,
        provider: str = "marmotmaterialpoint",
        baseClass: str = "hypoelastic",
    ):
        """Create a material.

        Parameters
        ----------
        materialName
            The name of the material, e.g. ``LinearElastic`` for a Marmot material or
            ``linearelastic`` for a native one.
        materialProperties
            The material properties.
        provider
            ``marmotmaterialpoint`` for a Marmot material evaluated point-wise, or
            ``edelweiss`` for a native EdelweissFE material.
        baseClass
            For ``marmotmaterialpoint``, the Marmot material base class, i.e.
            ``hypoelastic`` or ``gradientEnhancedHypoElastic``.

        Returns
        -------
        The material instance.
        """

        materialProperties = np.asarray(materialProperties, dtype=float)

        if provider == "marmotmaterialpoint":
            materialClass = getMaterialClass(baseClass, provider)

            return materialClass(materialName.upper(), materialProperties)

        materialClass = getMaterialClass(materialName, provider)

        return materialClass(materialProperties)

    def assignStencils(self, stencilClass, grid: StructuredGrid, material, **stencilOptions) -> list:
        """Place one stencil on every cell of a grid.

        Parameters
        ----------
        stencilClass
            The stencil class, e.g.
            :class:`~edelweissfd.stencils.displacementstencil.DisplacementStencil`.
        grid
            The grid to be covered.
        material
            The material, or a callable taking the cell centre coordinates and returning a
            material. The callable form allows a spatially varying material.
        stencilOptions
            Further keyword arguments passed on to the stencil constructor.

        Returns
        -------
        list
            The created stencils.
        """

        stencils = []

        materialFactory = material if callable(material) else None

        for cellIndex in grid.cellIndices():
            stencil = stencilClass(self._nextStencilNumber, grid.spacings, **stencilOptions)

            # setCell rather than setNodes, so that a stencil needing more than the corners of
            # its cell -- a Laplacian reaches one grid point further -- can widen its molecule
            stencil.setCell(grid, cellIndex)

            if materialFactory is not None:
                stencil.setMaterial(materialFactory(stencil.getCoordinatesAtCenter()))
            else:
                stencil.setMaterial(material)

            self.model.addStencil(stencil)

            stencils.append(stencil)

            self._nextStencilNumber += 1

        self.journal.message(
            "Assigned {:} {:} stencils on grid '{:}'".format(len(stencils), stencilClass.__name__, grid.name),
            self.identification,
        )

        return stencils

    # -- steps --------------------------------------------------------------------------

    def createSolver(self, solverName: str = "NIST", name: str = None, **solverOptions):
        """Create a solver.

        Parameters
        ----------
        solverName
            The name of the solver, e.g. ``NIST``, ``NISTParallel`` or ``NISTPArcLength``.
        name
            The name to register the solver under. Defaults to ``solverName``.
        solverOptions
            Options passed on to the solver.

        Returns
        -------
        The solver instance.
        """

        if name is None:
            name = solverName

        solverClass = getSolverByName(solverName)

        self.solvers[name] = solverClass(self.jobInfo, self.journal, **solverOptions)

        return self.solvers[name]

    def createStep(self, solver: str = None, **options) -> FDStep:
        """Create a simulation step.

        Parameters
        ----------
        solver
            The name of the solver to be used. Defaults to the first solver, creating a
            :class:`~edelweissfe.solvers.nonlinearimplicitstatic.NIST` if none exists yet.
        options
            The options of the adaptive step, e.g. ``stepLength``, ``maxInc``, ``minInc``,
            ``maxNumInc``, ``maxIter``.

        Returns
        -------
        FDStep
            The step.
        """

        if solver is None:
            if not self.solvers:
                self.createSolver("NIST")

            solver = next(iter(self.solvers))

        step = FDStep(len(self.steps), solver, options)

        self.steps.append(step)

        return step

    # -- output -------------------------------------------------------------------------

    def addNodeFieldOutput(
        self,
        name: str,
        field: str,
        result: str = "U",
        nodeSet: NodeSet = None,
        saveHistory: bool = False,
        f_x=None,
        export: str = None,
    ):
        """Request a field output of nodal values.

        Parameters
        ----------
        name
            The name of the field output.
        field
            The field, e.g. ``displacement``.
        result
            The result entry of the node field, i.e. ``U`` or ``P``.
        nodeSet
            The node set to restrict the output to. Defaults to all nodes.
        saveHistory
            Save the complete history instead of only the last result.
        f_x
            A callable post-processing the result.
        export
            The base name of a csv file the result is exported to.
        """

        self._nodeFieldOutputRequests.append(
            dict(
                name=name,
                field=field,
                result=result,
                nodeSet=nodeSet,
                saveHistory=saveHistory,
                f_x=f_x,
                export=export,
            )
        )

    def addStencilFieldOutput(
        self,
        name: str,
        result: str,
        stencils: list = None,
        materialPoints=None,
        saveHistory: bool = False,
        f_x=None,
        export: str = None,
    ):
        """Request a field output of stencil results, e.g. the stress.

        .. note::
            The result is shaped ``(nStencils, nMaterialPoints, resultSize)`` when more than one
            material point is collected, and squeezed to ``(nStencils, resultSize)`` when only
            one is, which is the case for every one dimensional grid. Reshaping with
            ``.reshape(-1, resultSize)`` covers both.

        Parameters
        ----------
        name
            The name of the field output.
        result
            The result the stencils provide, e.g. ``stress`` or a material state name.
        stencils
            The stencils to collect from. Defaults to all stencils.
        materialPoints
            The indices of the material points to collect from. Defaults to *all* material
            points of a stencil, which in two and three dimensions is more than one.
        saveHistory
            Save the complete history instead of only the last result.
        f_x
            A callable post-processing the result.
        export
            The base name of a csv file the result is exported to.
        """

        self._stencilFieldOutputRequests.append(
            dict(
                name=name,
                result=result,
                stencils=stencils,
                materialPoints=materialPoints,
                saveHistory=saveHistory,
                f_x=f_x,
                export=export,
            )
        )

    def addOutputManager(self, managerType: str, name: str = None, **options):
        """Request an output manager, e.g. a monitor printing a field output every increment.

        Parameters
        ----------
        managerType
            The type of the output manager, e.g. ``monitor`` or ``statusfile``.
        name
            The name of the output manager.
        options
            Options passed on to the output manager, e.g. ``fieldOutput`` for a monitor.
        """

        self._outputManagerRequests.append(
            dict(managerType=managerType, name=name if name is not None else managerType, options=options)
        )

    # -- execution ----------------------------------------------------------------------

    def run(self) -> tuple:
        """Prepare and solve the simulation.

        Returns
        -------
        tuple
            The final model tree and the field output controller holding all results.
        """

        journal = self.journal
        model = self.model

        journal.printSeperationLine()
        journal.message("Setting up finite difference model", self.identification, 0)

        tic = getCurrentTime()
        model.prepareYourself(journal)
        model.advanceToTime(self.startTime)
        toc = getCurrentTime()

        journal.printTable([("Model setup time ", "{:10.4f}s".format(toc - tic))], self.identification, level=0)

        self._printModelSummary()

        jobInfo = self.jobInfo
        jobInfo["computationTime"] = 0.0

        # the entries every solver and output manager expects on a node field
        for nodeField in model.nodeFields.values():
            nodeField.createFieldValueEntry("U")
            nodeField.createFieldValueEntry("P")

        model._linkFieldVariableObjects(model.nodeSets["all"])

        fieldOutputController = self._createFieldOutputController()
        fieldOutputController.initializeJob()

        # kept accessible, so that the results gathered up to a failed step can be inspected
        self.fieldOutputController = fieldOutputController

        outputManagers = self._createOutputManagers(fieldOutputController)
        for outputManager in outputManagers:
            outputManager.initializeJob()

        if not self.solvers:
            self.createSolver("NIST")

        solvers = dict(self.solvers)
        solvers["default"] = next(iter(self.solvers.values()))

        stepManager = StepManager()
        for step in self.steps:
            stepManager.enqueueStepDefinition(step.toStepDefinition())

            for setName, stencils in step._stencilSetsToRegister:
                model.elementSets[setName] = StencilSet(setName, stencils)

            for module, name, action in step._prebuiltActions:
                stepManager.stepActions[module][name] = action

        try:
            for step in stepManager.dequeueStep(
                jobInfo, model, fieldOutputController, journal, solvers, outputManagers
            ):
                tic = getCurrentTime()
                step.solve()
                toc = getCurrentTime()

                jobInfo["computationTime"] += toc - tic

                journal.printTable(
                    [("Step computation time", "{:10.4f}s".format(toc - tic))], self.identification, level=0
                )

        except KeyboardInterrupt:
            journal.errorMessage("Interrupted by user", self.identification)

        except StepFailed:
            journal.errorMessage("Simulation failed", self.identification)
            raise

        finally:
            journal.printTable(
                [("Job computation time", "{:10.4f}s".format(jobInfo["computationTime"]))],
                self.identification,
                level=0,
            )

            fieldOutputController.finalizeJob()
            for outputManager in outputManagers:
                outputManager.finalizeJob()

        return model, fieldOutputController

    def _createFieldOutputController(self) -> FieldOutputController:
        """Materialize the requested field outputs, now that the node fields exist."""

        fieldOutputController = FieldOutputController(self.model, self.journal)

        for request in self._nodeFieldOutputRequests:
            nodeField = self.model.nodeFields[request["field"]]

            if request["nodeSet"] is not None:
                nodeField = nodeField.subset(request["nodeSet"])

            fieldOutputController.addPerNodeFieldOutput(
                name=request["name"],
                nodeField=nodeField,
                result=request["result"],
                saveHistory=request["saveHistory"],
                f_x=request["f_x"],
                export=request["export"],
            )

        for request in self._stencilFieldOutputRequests:
            stencils = request["stencils"]
            if stencils is None:
                stencils = list(self.model.stencils.values())

            materialPoints = request["materialPoints"]
            if materialPoints is None:
                materialPoints = list(range(stencils[0].getNumberOfQuadraturePoints()))

            fieldOutputController.addPerElementFieldOutput(
                name=request["name"],
                elSet=StencilSet(request["name"] + "_stencils", stencils),
                result=request["result"],
                saveHistory=request["saveHistory"],
                f_x=request["f_x"],
                export=request["export"],
                quadraturePoints=list(materialPoints),
            )

        return fieldOutputController

    def _createOutputManagers(self, fieldOutputController: FieldOutputController) -> list:
        """Materialize the requested output managers."""

        outputManagers = []

        for request in self._outputManagerRequests:
            factory = getOutputManagerFactoryByName(request["managerType"])

            options = dict(request["options"])
            options.setdefault("label", request["name"])
            moduleOptions = options.pop("moduleOptions", dict())

            outputManagers.append(
                factory(
                    request["name"],
                    self.model,
                    fieldOutputController,
                    moduleOptions,
                    self.journal,
                    None,
                    **options,
                )
            )

        return outputManagers

    def _printModelSummary(self):
        """Log the size of the model."""

        model = self.model

        self.journal.printTable(
            [("Grids", len(model.grids)), ("Grid points", len(model.nodes)), ("Stencils", len(model.stencils))]
            + [
                ("Field '{:}'".format(name), "{:} grid points".format(len(nodeField.nodes)))
                for name, nodeField in model.nodeFields.items()
            ],
            self.identification,
            level=0,
        )
