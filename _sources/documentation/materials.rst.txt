Materials
=========

EdelweissFD does not implement constitutive models. It consumes those of EdelweissFE, which
are either implemented natively in Python or provided point-wise by Marmot.

One material instance per stencil
---------------------------------

.. automodule:: edelweissfd.materials.provider
   :members:
   :undoc-members:

Small strain materials
----------------------

.. autoclass:: edelweissfe.materials.base.basehypoelasticmaterial.BaseHypoElasticMaterial
   :members:

Gradient-enhanced materials
---------------------------

.. automodule:: edelweissfe.materials.base.basegradientenhancedhypoelasticmaterial
   :members:
   :undoc-members:
