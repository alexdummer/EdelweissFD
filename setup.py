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
Compiles the optional cell kernels of the finite difference stencils.

EdelweissFD is pure Python otherwise: the kernels exist only because a few stencils are small
and evaluated often enough that a numpy call per cell costs more in dispatch than in arithmetic,
see the individual kernel modules for the measurements that motivate each one. Every stencil
falls back to an equivalent pure-Python ``computeKernels`` when its kernel is not built, so a
failed or skipped extension only costs speed and never a missing feature.
"""

import os
import sys
from os.path import expanduser, join

import numpy
from Cython.Build import build_ext, cythonize
from setuptools import setup
from setuptools.extension import Extension

directives = {
    "boundscheck": False,
    "wraparound": False,
    "nonecheck": False,
    "initializedcheck": False,
}

default_install_prefix = sys.prefix
print("*" * 80)
print("EdelweissFD setup")
print("System prefix: " + sys.prefix)
print("*" * 80)

marmot_dir = expanduser(os.environ.get("MARMOT_INSTALL_DIR", default_install_prefix))
eigen_include = expanduser(os.environ.get("EIGEN_INCLUDE_DIR", join(default_install_prefix, "include/eigen3")))
arch_flags = os.environ.get("EDELWEISSFD_ARCH_FLAGS", "-march=native").split()
print("Marmot install directory (overwrite via environment var. MARMOT_INSTALL_DIR):")
print(marmot_dir)
print("Eigen include directory (overwrite via environment var. EIGEN_INCLUDE_DIR):")
print(eigen_include)
print("Architecture compile flags (overwrite via environment var. EDELWEISSFD_ARCH_FLAGS):")
print(arch_flags)

# The kernels cimport the point-wise Marmot material declarations of EdelweissFE and include
# the C++ shims living next to them, so both have to be locatable in the *installed* EdelweissFE
# -- editable or not -- rather than assumed to sit in a sibling checkout.
try:
    import edelweissfe

    # The parent of the installed edelweissfe package, so Cython's own cimport resolution (which
    # is independent of the C compiler's include_dirs below) can walk
    # edelweissfe/materials/marmot/*.pxd the same way it would inside a checkout of EdelweissFE
    # itself.
    edelweissfe_root_dir = os.path.dirname(os.path.dirname(edelweissfe.__file__))
    edelweissfe_marmot_materials_dir = join(os.path.dirname(edelweissfe.__file__), "materials", "marmot")
except ImportError:
    edelweissfe_root_dir = None
    edelweissfe_marmot_materials_dir = None
    print("!" * 80)
    print("WARNING: EdelweissFE is not installed. The compiled kernels will NOT be built.")
    print("!" * 80)

print("*" * 80)

extensions = []

if edelweissfe_marmot_materials_dir is not None:
    print("Gather the compiled finite difference cell kernels")
    for kernel_source in [
        "displacementkernel.pyx",
        "gradientenhanceddisplacementkernel.pyx",
        "gradientplasticitykernel.pyx",
    ]:
        extensions += [
            Extension(
                "*",
                sources=[join("edelweissfd", "kernels", kernel_source)],
                include_dirs=[
                    join(marmot_dir, "include"),
                    numpy.get_include(),
                    eigen_include,
                    edelweissfe_marmot_materials_dir,
                ],
                libraries=["Marmot"],
                library_dirs=[join(marmot_dir, "lib")],
                runtime_library_dirs=[join(marmot_dir, "lib")],
                language="c++",
                extra_compile_args=["-O3", "-std=c++20", *arch_flags],
            )
        ]

print("Now compile!")


class optional_build_ext(build_ext):
    def build_extensions(self):
        for ext in self.extensions:
            try:
                self.build_extension(ext)
                print(f"[OK] Built extension: {ext.name}")
            except Exception as e:
                print(f"[FAIL] Could not build {ext.name}: {e}")


setup(
    cmdclass={"build_ext": optional_build_ext},
    ext_modules=cythonize(
        extensions,
        compiler_directives=directives,
        annotate=True,
        language_level=3,
        include_path=[edelweissfe_root_dir] if edelweissfe_root_dir is not None else [],
    ),
)

print("Finish!")
