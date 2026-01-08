import unittest
import numpy as np

from star_pso.population.swarm import Swarm
from star_pso.population.particle import Particle
from star_pso.engines.integer_pso import IntegerPSO
from star_pso.utils.auxiliary import cost_function


@cost_function(minimize=True)
def fun_sum_abs(x: np.ndarray, **kwargs) -> tuple[float, bool]:
    """
    The cost function to test the Integer_PSO algorithm is
    simply the sum(abs()) that sums the absolute values of
    the input vector (array).

    :param x: the input array we want to test.

    CAUTION: The 'kwargs' is added for compatibility.
             The code will fail if it is removed.
    """
    # Compute the function value.
    f_value = np.sum(np.abs(x))

    # Condition for termination.
    # If the sum of all elements is '0' we stop.
    solution_found = f_value == 0.0

    # Return the solution tuple.
    return f_value, solution_found
# _end_def_


class TestIntegerPSO(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Set a seed for reproducible initial population.
        SEED = 1821

        # Random number generator.
        rng = np.random.default_rng(SEED)

        # Define the number of optimizing variables.
        D = 15

        # Define the number of particles.
        N = 60

        # Sample the initial points randomly.
        X_t0 = rng.integers(low=-100, high=100, endpoint=True, size=(N, D))

        # Initial population.
        cls.swarm_t0 = Swarm([Particle(x) for x in X_t0])
    # _end_def_

    def test_run(self):
        """
        Test the run() method.
        """
        # Fix the seed (for reproducibility).
        IntegerPSO.set_seed(2026)

        # Create the IntegerPSO object that will perform the optimization.
        test_PSO = IntegerPSO(initial_swarm = TestIntegerPSO.swarm_t0,
                              obj_func = fun_sum_abs,
                              copy = True, x_min = -100, x_max = +100)

        # Run the PSO.
        test_PSO.run(max_it=100,
                     options={"w0": 0.70, "c1": 1.50, "c2": 1.50, "mode": "fipso"},
                     reset_swarm=False, verbose=False, adapt_params=False)

        # Get the optimal solution from the PSO.
        _, f_opt, _ = test_PSO.get_optimal_values()

        # This assumes the optimization was successful.
        self.assertEqual(0.0, f_opt)
    # _end_def_

# _end_class_


if __name__ == '__main__':
    unittest.main()
