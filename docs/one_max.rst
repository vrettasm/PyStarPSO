OneMax
======

Description:

    - Optimization (max)
    - Single-objective
    - Constraints (no)

The general problem statement is given by:

We have a state vector :math:`\mathbf{x} \in [0, 1]^M` of bits, such as: :math:`\mathbf{x} = (0, 0, 1, 1, 0, ..., 1)`.

- The optimal solution is the one where each variable has the value of '1'.

    .. math::
        f(\mathbf{x}) = \sum_{i=1}^{M} x_i, \text{ with } x_i \in \{0, 1\}

- Global maximum is found at:

    .. math::
        f(1, 1, ..., 1) = M


Step 1: Import python libraries and StarPSO classes
---------------------------------------------------

.. code-block:: python

    import numpy as np
    from star_pso.population.swarm import Swarm
    from star_pso.population.particle import Particle
    from star_pso.engines.binary_pso import BinaryPSO
    from star_pso.utils.auxiliary import cost_function

Step 2: Define the objective function
-------------------------------------

.. code-block:: python

    @cost_function
    def fun_one_max(x: np.ndarray, **kwargs) -> tuple[float, bool]:

        # Compute the function value.
        f_val = np.sum(x)

        # Condition for termination.
        solution_found = f_val == len(x)

        # Return the solution tuple.
        return f_val, solution_found

Step 3: Set the PSO parameters
------------------------------

.. code-block:: python

    # Set a seed for reproducible initial population.
    SEED = 1821

    # Random number generator.
    rng = np.random.default_rng(SEED)

    # Define the number of optimizing variables.
    n_dim = 30

    # Define the number of particles.
    n_pop = 60

    # Sample the initial points randomly.
    X_t0 = rng.integers(low=0, high=1, endpoint=True, size=(n_pop, n_dim))

    # Initial population (swarm).
    swarm_t0 = Swarm([Particle(x) for x in X_t0])

    # Create a BinaryPSO object that will perform the optimization.
    test_PSO = BinaryPSO(initial_swarm = swarm_t0, obj_func = fun_one_max)

Step 4: Run the optimization
----------------------------

.. code-block:: python

    test_PSO.run(max_it = 1000,
                 options = {"w0": 0.70, "c1": 1.50, "c2": 1.50, "mode": "g_best"},
                 reset_swarm = False, verbose = False, adapt_params = False)

Step 5: Final output
--------------------

.. code-block:: python

    # Get the optimal solution from the PSO.
    i_opt, f_opt, x_opt = test_PSO.get_optimal_values()

    # Display the (final) optimal value.
    print(f"Optimum Found: {f_opt} / {D}, at iteration {i_opt}.\n")

    # Display each particle position value.
    print(f"Optimum position: {x_opt}")