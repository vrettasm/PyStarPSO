import numpy as np
from numpy.typing import NDArray

from star_pso.population.particle import Particle
from star_pso.benchmarks.test_function import TestFunction
from star_pso.utils.auxiliary import (identify_global_optima,
                                      calculate_dynamic_radius)


class EqualMaxima(TestFunction):
    """
    This function was originally proposed in:

    - K. Deb, “Genetic algorithms in multimodal function optimization
      (master thesis and tcga report no. 89002),” Ph.D. dissertation,
      Tuscaloosa: University of Alabama, The Clearinghouse for Genetic
      Algorithms, 1989.
    """

    def __init__(self, x_min: float = 0.0, x_max: float = 1.0) -> None:
        """
        Default initializer of the EqualMaxima class.

        :param x_min: (float) the lower bound values of the search space.

        :param x_max: (float) the upper bound values of the search space.

        :return: None.
        """
        # Call the super initializer.
        super().__init__(name="Equal_Maxima",
                         n_dim=1, x_min=x_min, x_max=x_max)
    # _end_def_

    def func(self, x_pos: NDArray) -> NDArray:
        """
        This is 1D function. There are 5 global optima.

        :param x_pos: the current position(s) of the function.

        :return: the function value(s).
        """
        # Convert input cleanly without re-allocation
        # if x_pos is already an array.
        x_pos: NDArray = np.asarray(x_pos)

        # Create a boolean mask for element-wise boundary checking.
        in_bounds: NDArray = (self.x_min <= x_pos) & (x_pos <= self.x_max)

        # Calculate values vectorized.
        f_value: NDArray = np.sin(5.0 * np.pi * x_pos) ** 6

        # Return f_value where true, else NaN.
        return np.where(in_bounds, f_value, np.nan)
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
        # Calculate the radius dynamically.
        radius: float = calculate_dynamic_radius(self.x_min, self.x_max)

        # Get the global optima particles.
        found_optima: list[Particle] = identify_global_optima(population,
                                                              f_opt=1.0,
                                                              epsilon=epsilon, radius=radius)
        # Find the number of optima.
        num_optima: int = len(found_optima)

        # Return the tuple (number of found, total number)
        return num_optima, 5
    # _end_def_

# _end_class_
