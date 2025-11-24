import numpy as np
from star_pso.population.particle import Particle
from star_pso.benchmarks.test_function import TestFunction


class SixHumpCamelBack(TestFunction):
    """
    This function was originally proposed in:

    Z. Michalewicz, Genetic Algorithms + Data Structures = Evolution Programs.
    New York: Springer-Verlag, New York, 1996.
    """

    def __init__(self) -> None:
        """
        Default initializer of the SixHumpCamelBack class.
        """

        # Call the super initializer with the name and the limits.
        super().__init__(name="Six_Hump_Camel_Back",
                         x_min=np.array([-1.9, -1.1]),
                         x_max=np.array([+1.9, +1.1]))
    # _end_def_

    def func(self, x_pos: np.ndarray) -> np.ndarray:
        """
        This is a 2D function with 2 global and 2 local optima.

        :param x_pos: the current position(s) of the function.

        :return: the function value(s).
        """

        # Initialize function values to NaN.
        f_value = float("NaN")

        # Vectorized calculations based on the condition.
        if np.all((self.x_min <= x_pos) & (x_pos <= self.x_max)):
            # Separate the two variables.
            x, y = x_pos

            # Calculate the function value.
            f_value = -4 * ((4 - 2.1 * x**2 + (x**4)/3) * x**2 +
                            x*y + 4*(y**2 - 1)*(y**2))
        # Return the ndarray.
        return f_value
    # _end_def_

    def initial_random_positions(self, n_pos: int = 50) -> np.ndarray:
        """
        Generate an initial set of uniformly random sampled positions
        within the minimum / maximum bounds of the test problem.

        :param n_pos: the number of positions to generate.

        :return: a uniformly sampled set of random positions.
        """
        # Draw uniform random samples for the initial points.
        return self.rng.uniform(self._x_min, self._x_max, size=(n_pos, 2))
    # _end_def_

    def global_optima(self, population: list[Particle]) -> (int, int):
        """
        Calculates the global optimum found in the input population.

        :param population: the population to search the global optimum.

        :return: a tuple with the number of global optima found and the
        total number that exist.
        """
        # Get the global optima particles.
        found_optima = self.global_optima_found(population, epsilon=1.0E-3,
                                                radius=0.5, f_opt=4.12651381395951)
        # Find the number of optima.
        num_optima = len(found_optima)

        # Return the tuple (number of found, total number)
        return num_optima, 2
    # _end_def_

# _end_class_
