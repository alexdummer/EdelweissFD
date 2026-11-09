Installation
============

Prerequisites
-------------

EdelweissFD stands on two compiled projects which have to be installed first: Marmot for the
constitutive models, and EdelweissFE for the solution machinery. EdelweissFD's own compiled cell
kernels -- an optional speedup, every stencil falls back to an equivalent pure-Python
implementation when one is missing -- need both too, since they reach Marmot through the
point-wise material interfaces EdelweissFE declares.

1. `Marmot <https://github.com/MAteRialMOdelingToolbox/Marmot>`_, providing the constitutive
   models. Install it into the environment prefix::

       cd Marmot
       cmake -B build -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
       cmake --build build -j
       cmake --install build

2. `EdelweissFE <https://github.com/Edelweiss-Numerics/EdelweissFE>`_ (release ``v26.11``, or branch ``master`` for development),
   providing everything but the discretization. Its point-wise Marmot material interfaces need
   the Eigen headers, so point ``EIGEN_INCLUDE_DIR`` at them if they are not in
   ``$CONDA_PREFIX/include/eigen3``, and ``MARMOT_INSTALL_DIR`` at Marmot if it is not installed
   into the environment prefix::

       cd EdelweissFE
       MARMOT_INSTALL_DIR=/path/to/marmot pip install --no-build-isolation .

   Afterwards, check that the material interfaces really were built; ``setup.py`` tolerates
   failing extensions on purpose::

       grep marmot $(python -c "import edelweissfe, os; \
           print(os.path.join(os.path.dirname(edelweissfe.__file__), 'built_extensions.log'))")

   The list has to contain ``edelweissfe.materials.marmot.marmothypoelastic`` and
   ``edelweissfe.materials.marmot.marmotgradientenhancedhypoelastic``.

   The gradient plasticity material family (``marmotgradientplasticityhypoelastic``), which
   :class:`~edelweissfd.stencils.gradientplasticitystencil.GradientPlasticityStencil` and
   :class:`~edelweissfd.stencils.gfdmgradientplasticitystencil.GFDMGradientPlasticityStencil`
   need, is not yet part of EdelweissFE and Marmot ``next_v26.11``. Without it, these two stencils
   raise on construction, their compiled kernel is not built, and their tests are skipped.

EdelweissFD
-----------

Its own compiled kernels need the same Eigen headers and Marmot install as EdelweissFE's
material interfaces above, since they cimport EdelweissFE's C++ shims directly; the
architecture flags follow EdelweissFE's ``EDELWEISSFE_ARCH_FLAGS``::

    cd EdelweissFD
    MARMOT_INSTALL_DIR=/path/to/marmot pip install --no-build-isolation -e .

Check that they really were built the same way as above, this time against EdelweissFD's own
stencils::

    python -c "from edelweissfd.stencils import displacementstencil as ds; \
        assert ds.DisplacementKernel is not None"

Tests
-----

.. code-block:: bash

    pytest

Tests requiring the compiled Marmot material interfaces are marked ``marmot``, and those
requiring the gradient plasticity material family additionally ``gradientplasticity``; both are
skipped automatically if the respective interfaces are unavailable:

.. code-block:: bash

    pytest -m "not marmot"
