import numpy as np
from star_pso.population.particle import Particle
from star_pso.benchmarks.test_function import TestFunction


class FiveUnevenPeakTrap(TestFunction):
    """
    This function was originally proposed in:

    J.-P. Li, M. E. Balazs, G. T. Parks, and P. J. Clarkson,
    “A species conserving genetic algorithm for multimodal function optimization”
    Evolutionary Computation, vol. 10, no. 3, pp. 207–234, 2002.
    """

    def __init__(self) -> None:
        """
        Default initializer of the FiveUnevenPeakTrap class.
        """

        # Call the super initializer with the name and the limits.
        super().__init__(name="Five_Uneven_Peak_Trap", x_min=0.0, x_max=30.0)
    # _end_def_

    def func(self, x_pos: np.ndarray) -> np.ndarray:
        """
        This is 1D function. There are two global and one local optima.

        :param x_pos: the current position(s) of the function.

        :return: the function value(s).
        """
        # Initialize function values to NaN.
        f_value = np.nan

        # Conditions for the different ranges.
        if (0.0 <= x_pos) & (x_pos < 2.5):
            f_value = 80 * (2.50 - x_pos)
        elif (2.5 <= x_pos) & (x_pos < 5.0):
            f_value = 64 * (x_pos - 2.50)
        elif (5.0 <= x_pos) & (x_pos < 7.5):
            f_value = 64 * (7.50 - x_pos)
        elif (7.5 <= x_pos) & (x_pos < 12.5):
            f_value = 28 * (x_pos - 7.50)
        elif (12.5 <= x_pos) & (x_pos < 17.5):
            f_value = 28 * (17.5 - x_pos)
        elif (17.5 <= x_pos) & (x_pos < 22.5):
            f_value = 32 * (x_pos - 17.5)
        elif (22.5 <= x_pos) & (x_pos < 27.5):
            f_value = 32 * (27.5 - x_pos)
        elif (27.5 <= x_pos) & (x_pos <= 30.0):
            f_value = 80 * (x_pos - 27.5)
        # _end_if_
        
        # Return the value.
        return f_value
    # _end_def_

    def sample_random_positions(self, n_pos: int = 50) -> np.ndarray:
        """
        Generate an initial set of uniformly random sampled positions
        within the minimum / maximum bounds of the test problem.

        :param n_pos: the number of positions to generate.

        :return: a uniformly sampled set of random positions.
        """
        # Draw uniform random samples for the initial points.
        return self.rng.uniform(self._x_min, self._x_max, size=(n_pos, 1))
    # _end_def_

    def search_for_optima(self, population: list[Particle],
                          epsilon: float = 1.0e-4) -> tuple[int, int]:
        """
        Searches the input population for the global optimum values
        of the specific test function, using default (problem specific)
        parameters.

        :param population: the population to search the global optimum.

        :param epsilon: accuracy level of the global optimal solution.

        :return: a tuple with the number of global optima found and the
        total number that exist.
        """
        # Get the global optima particles.
        found_optima = self.search_global_optima(population, epsilon=epsilon,
                                                 radius=0.01, f_opt=200.0)
        # Find the number of optima.
        num_optima = len(found_optima)

        # Return the tuple (number of found, total number)
        return num_optima, 2
    # _end_def_

# _end_class_
