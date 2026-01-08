import unittest
import numpy as np

from star_pso.population.swarm import Swarm
from star_pso.population.particle import Particle
from star_pso.engines.binary_pso import BinaryPSO
from star_pso.utils.auxiliary import cost_function


@cost_function
def fun_one_max(x: np.ndarray, **kwargs) -> tuple[float, bool]:
    """
    The cost function to test the Binary_PSO algorithm is
    simply the OneMax that sums the numbers of '1' in the
    input vector (array).

    :param x: the input array we want to test.

    CAUTION: The 'kwargs' is added for compatibility.
             The code will fail if it is removed.
    """
    # Compute the function value.
    f_value = np.sum(x)

    # Condition for termination.
    # If all elements are '1' we stop.
    solution_found = f_value == len(x)

    # Return the solution tuple.
    return f_value, solution_found
# _end_def_


class TestBinaryPSO(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Set a seed for reproducible initial population.
        SEED = 1821

        # Random number generator.
        rng = np.random.default_rng(SEED)

        # Define the number of optimizing variables.
        D = 20

        # Define the number of particles.
        N = 40

        # Sample the initial points randomly.
        X_t0 = rng.integers(low=0, high=1, endpoint=True, size=(N, D))

        # Initial population.
        cls.swarm_t0 = Swarm([Particle(x) for x in X_t0])
    # _end_def_

    def test_run(self):
        """
        Test the run() method.
        """
        # Fix the seed (for reproducibility).
        BinaryPSO.set_seed(2026)

        # Create a BinaryPSO object that will perform the optimization.
        test_PSO = BinaryPSO(initial_swarm=TestBinaryPSO.swarm_t0,
                             obj_func=fun_one_max)

        # Run the optimization.
        test_PSO.run(max_it=100,
                     options={"w0": 0.70, "c1": 1.50, "c2": 1.50, "mode": "g_best"},
                     reset_swarm=False, verbose=False, adapt_params=False)

        # Get the optimal solution from the PSO.
        _, f_opt, _ = test_PSO.get_optimal_values()

        # This assumes the optimization was successful.
        self.assertEqual(test_PSO.n_cols, int(f_opt))
    # _end_def_

# _end_class_


if __name__ == '__main__':
    unittest.main()
