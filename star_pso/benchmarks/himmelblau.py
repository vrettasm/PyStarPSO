import numpy as np
from numpy.typing import NDArray

from star_pso.population.particle import Particle
from star_pso.benchmarks.test_function import TestFunction
from star_pso.utils.auxiliary import (identify_global_optima,
                                      calculate_dynamic_radius)


class Himmelblau(TestFunction):
    """
    This function was originally proposed in:

    - K. Deb, “Genetic algorithms in multimodal function optimization
      (master thesis and tcga report no. 89002)”,  Ph.D. dissertation,
      Tuscaloosa: University of Alabama, The Clearinghouse for Genetic
      Algorithms, 1989.
    """

    def __init__(self, x_min: float = -6.0, x_max: float = 6.0) -> None:
        """
        Default initializer of the Himmelblau class.

        :param x_min: (float) the lower bound values of the search space.

        :param x_max: (float) the upper bound values of the search space.

        :return: None.
        """
        # Call the super initializer.
        super().__init__(name="Himmelblau",
                         n_dim=2, x_min=x_min, x_max=x_max)
    # _end_def_

    def func(self, x_pos: NDArray) -> NDArray:
        """
        This is 2D function with is 4 global optima.

        :param x_pos: the current position(s) of the function.

        :return: the function value(s).
        """
        # Force array context cleanly.
        x_pos: NDArray = np.asarray(x_pos)

        # Branchless slicing: works for both 1D arrays and 2D matrices.
        x: NDArray = x_pos[..., 0]
        y: NDArray = x_pos[..., 1]

        # Create a boolean mask for element-wise boundary checking.
        in_bounds: NDArray = ((self.x_min <= x) & (x <= self.x_max) &
                              (self.x_min <= y) & (y <= self.x_max))

        # Calculate values vectorized.
        f_value: NDArray = 200.0 - (x ** 2 + y - 11) ** 2 - (x + y ** 2 - 7) ** 2

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
                                                              f_opt=200.0,
                                                              epsilon=epsilon,
                                                              radius=radius)
        # Find the number of optima.
        num_optima: int = len(found_optima)

        # Return the tuple (number of found, total number)
        return num_optima, 4
    # _end_def_

# _end_class_
