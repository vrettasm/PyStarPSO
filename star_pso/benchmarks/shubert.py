import numpy as np
from numpy.typing import NDArray

from star_pso.population.particle import Particle
from star_pso.benchmarks.test_function import TestFunction
from star_pso.utils.auxiliary import (identify_global_optima,
                                      calculate_dynamic_radius)


class Shubert(TestFunction):
    """
    This function was originally proposed in:

    - Z. Michalewicz, Genetic Algorithms + Data Structures = Evolution Programs.
      New York: Springer-Verlag, New York, 1996.
    """

    def __init__(self, n_dim: int = 2, x_min: float = -10.0, x_max: float = 10.0) -> None:
        """
        Default initializer of the Shubert D class.

        :param n_dim: (int) the number of dimension of the input space.

        :param x_min: (float) the lower bound values of the search space.

        :param x_max: (float) the upper bound values of the search space.

        :return: None.
        """
        # Ensure correct type.
        n_dim = int(n_dim)

        # Sanity check.
        if n_dim < 2:
            raise ValueError("Shubert needs at least 2 dimensions.")

        # Call the super initializer.
        super().__init__(name=f"Shubert_{n_dim}D",
                         n_dim=n_dim, x_min=x_min, x_max=x_max)
    # _end_def_

    def func(self, x_pos: NDArray) -> NDArray:
        """
        This is a multidimensional function with 'n_dim * 3^n_dim'
        global optimal values.

        :param x_pos: the current position(s) of the function.

        :return: the function value(s).
        """
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

            # Prepare the constant array: i = [1, 2, 3, 4, 5]
            # Reshape it to (5, 1, ..., 1) so it broadcasts safely
            # over valid_points.
            i = np.arange(1, 6)
            i_broadcast = i.reshape((5,) + (1,) * valid_points.ndim)

            # Compute the internal Shubert sum per-dimension.
            dim_sums = np.sum(i_broadcast * np.cos((i_broadcast + 1) * valid_points + i_broadcast),
                              axis=0)

            # Product of the sums across all dimensions (axis=-1)
            f_value[in_bounds] = -np.prod(dim_sums, axis=-1)

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
        # Sanity check.
        if self.n_dim > 3:
            raise ValueError(f"Unknown 'f_opt' for D = {self.n_dim}")
        # _end_if_

        # Calculate the total global optima along with
        # the f_opt for the given number of dimensions.
        if self.n_dim == 2:
            total_optima, f_opt = 18, 186.7309088
        else:
            total_optima, f_opt = 81, 2709.093505
        # _end_if_

        # Calculate the radius dynamically.
        radius = calculate_dynamic_radius(self.x_min, self.x_max)

        # Get the global optima particles.
        found_optima = identify_global_optima(population, f_opt=f_opt,
                                              epsilon=epsilon, radius=radius)
        # Find the number of optima.
        num_optima = len(found_optima)

        # Return the tuple (number of found, total number)
        return num_optima, total_optima
    # _end_def_

# _end_class_
