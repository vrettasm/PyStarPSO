import numpy as np
from star_pso.population.particle import Particle
from star_pso.benchmarks.test_function import TestFunction
from star_pso.utils.auxiliary import identify_global_optima


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
        super().__init__(name="Six_Hump_Camel_Back", n_dim=2,
                         x_min=np.array([-1.9, -1.1]),
                         x_max=np.array([+1.9, +1.1]))
    # _end_def_

    def func(self, x_pos: np.ndarray) -> float | np.ndarray:
        """
        This is a 2D function with 2 global and 2 local optima.

        :param x_pos: the current position(s) of the function.

        :return: the function value(s).
        """
        # Initialize function value to NaN.
        f_value = np.nan

        # Condition for the valid range.
        if np.all((self.x_min <= x_pos) & (x_pos <= self.x_max)):
            # Separate the two variables.
            x, y = x_pos

            # Calculate the function value.
            f_value = -((4 - 2.1 * x**2 + (x**4)/3) * x**2 +
                        x*y + 4*(y**2 - 1)*(y**2))

        # Return the value.
        return f_value
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
        found_optima = identify_global_optima(population, epsilon=epsilon,
                                              radius=0.5, f_opt=1.031628453)
        # Find the number of optima.
        num_optima = len(found_optima)

        # Return the tuple (number of found, total number)
        return num_optima, 2
    # _end_def_

# _end_class_
