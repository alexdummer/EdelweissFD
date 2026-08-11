Installation
============

Prerequisites
-------------

EdelweissFD is pure Python, but it stands on two compiled projects which have to be
installed first.

1. `Marmot <https://github.com/MAteRialMOdelingToolbox/Marmot>`_, providing the constitutive
   models. Install it into the environment prefix::

       cd Marmot
       cmake -B build -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
       cmake --build build -j
       cmake --install build

2. `EdelweissFE <https://github.com/EdelweissFE/EdelweissFE>`_, providing everything but the
   discretization. Its point-wise Marmot material interfaces need the Eigen headers, so
   point ``EIGEN_INCLUDE_DIR`` at them if they are not in the environment prefix::

       cd EdelweissFE
       EIGEN_INCLUDE_DIR=/path/to/include/eigen3 pip install .

   Afterwards, check that the material interfaces really were built; ``setup.py`` tolerates
   failing extensions on purpose::

       grep marmot $(python -c "import edelweissfe, os; \
           print(os.path.join(os.path.dirname(edelweissfe.__file__), 'built_extensions.log'))")

   The list has to contain ``edelweissfe.materials.marmot.marmothypoelastic`` and
   ``edelweissfe.materials.marmot.marmotgradientenhancedhypoelastic``.

EdelweissFD
-----------

.. code-block:: bash

    cd EdelweissFD
    pip install -e .

Tests
-----

.. code-block:: bash

    pytest

Tests requiring the compiled Marmot material interfaces are marked ``marmot`` and are
skipped automatically if those are unavailable:

.. code-block:: bash

    pytest -m "not marmot"
