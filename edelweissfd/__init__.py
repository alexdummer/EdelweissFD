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
EdelweissFD, a finite difference framework for multifield problems.

See :mod:`edelweissfd.drivers.pythonscriptedsimulation` for the scripting interface.
"""

from edelweissfe.utils.inputlanguage import InputLanguage

# EdelweissFE registers its input language keywords as an import side effect of its input
# file parser, and several of its modules -- among them the solvers and every step action --
# look those keywords up at *import* time. The registry therefore has to be populated before
# any of them is imported, which holds even though EdelweissFD never reads an input file.
#
# This has to happen from a clean import context, i.e. before the first `edelweissfe` module
# that the parser itself imports is loaded, which is why it lives here rather than in the
# driver module.
InputLanguage().ensureParserLoaded()
