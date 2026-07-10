import numpy as np
from numpy.typing import NDArray

from star_pso.population.particle import Particle
from star_pso.benchmarks.test_function import TestFunction
from star_pso.utils.auxiliary import (identify_global_optima,
                                      calculate_dynamic_radius)


class Vincent(TestFunction):
    """
    This function was originally proposed in:

    - O. Shir and T. Back, “Niche radius adaptation in the cms-es niching algorithm”,
      in Parallel Problem-Solving from Nature - PPSN IX, 9th International Conference
      (LNCS 4193). Reykjavík, Iceland: Springer, 2006, pp. 142 – 151.
    """

    def __init__(self, n_dim: int = 2, x_min: float = 0.25, x_max: float = 10.0) -> None:
        """
        Default initializer of the Vincent D class.

        :param n_dim: (int) the number of dimension of the input space.

        :param x_min: (float) the lower bound values of the search space.

        :param x_max: (float) the upper bound values of the search space.

        :return: None.
        """
        # Ensure correct type.
        n_dim = int(n_dim)

        # Sanity check.
        if n_dim < 2:
            raise ValueError("Vincent needs at least 2 dimensions.")

        # Call the super initializer.
        super().__init__(name=f"Vincent_{n_dim}D",
                         n_dim=n_dim, x_min=x_min, x_max=x_max)
    # _end_def_

    def func(self, x_pos: NDArray) -> NDArray:
        """
        This is an n_dim function with 6^n_dim global optimal
        values.

        :param x_pos: the current position(s) of the function.

        :return: the function value(s).
        """
        # Number of dimensions.
        n_dim = x_pos.size if x_pos.ndim == 1 else x_pos.shape[1]

        # Ensure input is NDArray.
        x_pos = np.asarray(x_pos)

        # Evaluate boundaries element-by-element along the coordinate axis.
        in_bounds = np.all((self.x_min <= x_pos) &
                           (x_pos <= self.x_max), axis=-1)

        # Setup output array matching the layout of the points.
        f_value = np.full(np.shape(x_pos[..., 0]), np.nan, dtype=float)

        # Only calculate the expression for elements inside bounds.
        if np.any(in_bounds):
            # Extract only the valid points.
            valid_points = x_pos[in_bounds]

            f_value[in_bounds] = np.sum(np.sin(10.0 * np.log(valid_points)), axis=-1) / n_dim

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
        # Calculate the total global optima along with
        # the f_opt for the given number of dimensions.
        total_optima = int(6**self.n_dim)

        # Calculate the radius dynamically.
        radius = calculate_dynamic_radius(self.x_min, self.x_max)

        # Get the global optima particles.
        found_optima = identify_global_optima(population, f_opt=1.0,
                                              epsilon=epsilon, radius=radius)
        # Find the number of optima.
        num_optima = len(found_optima)

        # Return the tuple (number of found, total number)
        return num_optima, total_optima
    # _end_def_

# _end_class_
