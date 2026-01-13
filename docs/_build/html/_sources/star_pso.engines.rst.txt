Engines
=======
This package contains a set of **PSO** variants. Bellow we can see a summary of them along with the options that they
support. The variable types, refers to the kind of data the algorithm supports and operates on. The *Jack of all trades*
algorithm supports mixed types, which means that the particle can have any kind of data, in any order.

    - **G_best**: is the default option for the positional updates, since it is the simplest and supported
      by all algorithms.
    - **FIPSO**: Refers to the *Fully Informed PSO* and is also supported by all algorithms, albeit with
      slightly higher computation cost.
    - **Multimodal**: is a new feature that allows the PSO to look for local optima values and not stuck to
      a single global optimal value. It is restricted to continuous variable types (Float), since it is using
      a 'distance' measure to calculate the proximity among particles.
    - **Parallel**: is supported natively by all algorithms. Note however that is implemented in the evaluation
      of the objective function. So, depending on the computational cost it might be faster to leave it as 'False'
      (default parameter).

+--------------------+------------------+----------------------+------------+-----------+----------------+--------------+
| **Algorithm**      | **Var. Type(s)** | **Adapt parameters** | **G_best** | **FIPSO** | **Multimodal** | **Parallel** |
+====================+==================+======================+============+===========+================+==============+
| Standard           |      Float       |         Yes          |    Yes     |    Yes    |      Yes       |     Yes      |
+--------------------+------------------+----------------------+------------+-----------+----------------+--------------+
| Binary             |   Int. (0, 1)    |         Yes          |    Yes     |    Yes    |       No       |     Yes      |
+--------------------+------------------+----------------------+------------+-----------+----------------+--------------+
| Categorical        |   Set {...}      |         Yes          |    Yes     |    Yes    |       No       |     Yes      |
+--------------------+------------------+----------------------+------------+-----------+----------------+--------------+
| Integer            |     Integer      |         Yes          |    Yes     |    Yes    |       No       |     Yes      |
+--------------------+------------------+----------------------+------------+-----------+----------------+--------------+
| Jack of all trades |      Mixed       |         Yes          |    Yes     |    Yes    |       No       |     Yes      |
+--------------------+------------------+----------------------+------------+-----------+----------------+--------------+
| Quantum            |      Float       |         Yes          |    Yes     |    Yes    |      Yes       |     Yes      |
+--------------------+------------------+----------------------+------------+-----------+----------------+--------------+

Submodules
----------

star_pso.engines.binary\_pso module
-----------------------------------

.. automodule:: star_pso.engines.binary_pso
   :members:
   :undoc-members:
   :show-inheritance:

star_pso.engines.quantum\_pso module
------------------------------------

.. automodule:: star_pso.engines.quantum_pso
   :members:
   :undoc-members:
   :show-inheritance:

star_pso.engines.categorical\_pso module
----------------------------------------

.. automodule:: star_pso.engines.categorical_pso
   :members:
   :undoc-members:
   :show-inheritance:

star_pso.engines.generic\_pso module
------------------------------------

.. automodule:: star_pso.engines.generic_pso
   :members:
   :undoc-members:
   :show-inheritance:

star_pso.engines.integer\_pso module
------------------------------------

.. automodule:: star_pso.engines.integer_pso
   :members:
   :undoc-members:
   :show-inheritance:

star_pso.engines.jack\_of\_all\_trades\_pso module
--------------------------------------------------

.. automodule:: star_pso.engines.jack_of_all_trades_pso
   :members:
   :undoc-members:
   :show-inheritance:

star_pso.engines.standard\_pso module
-------------------------------------

.. automodule:: star_pso.engines.standard_pso
   :members:
   :undoc-members:
   :show-inheritance:

Module contents
---------------

.. automodule:: star_pso.engines
   :members:
   :undoc-members:
   :show-inheritance:
