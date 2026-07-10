import numpy as np
from numpy.typing import NDArray

from star_pso.population.particle import Particle
from star_pso.benchmarks.test_function import TestFunction
from star_pso.utils.auxiliary import (identify_global_optima,
                                      calculate_dynamic_radius)


class Rastrigin(TestFunction):
    """
    This function was originally proposed in:

    - A. Saha and K. Deb, “A bi-criterion approach to multimodal optimization:
      self-adaptive approach”, in Proceedings of the 8th international conference
      on Simulated evolution and learning, ser. SEAL-10. Berlin, Heidelberg:
      Springer-Verlag, 2010, pp. 95–104.
    """

    def __init__(self, n_dim: int = 2, x_min: float = 0.0, x_max: float = 1.0) -> None:
        """
        Default initializer of the D dimensional Rastrigin.

        :param n_dim: Number of dimensions of the problem.

        :param x_min: (float) the lower bound values of the search space.

        :param x_max: (float) the upper bound values of the search space.

        :return: None.
        """
        # Ensure correct type.
        n_dim: int = int(n_dim)

        # Sanity check.
        if n_dim < 2:
            raise ValueError("Rastrigin needs at least 2 dimensions.")

        # Call the super initializer with the name and the limits.
        super().__init__(name=f"Rastrigin_{n_dim}D",
                         n_dim=n_dim, x_min=x_min, x_max=x_max)

        # Set the 'kappa' coefficients (automatically).
        # Here we set them as: [1, 2, 1, 2, ...].
        self.kappa: NDArray = np.array([1 if i % 2 != 0 else 2
                                        for i in range(1, self.n_dim + 1)])

        # Compute the total number of optimal values
        # as the product of the 'kappa' coefficients.
        self.total_optima: int = np.prod(self.kappa, dtype=int)
    # _end_def_

    def func(self, x_pos: NDArray) -> NDArray:
        """
        This is a multidimensional function with 'M' global optimal values.

        :param x_pos: the current position(s) of the function.

        :return: the function value(s).
        """
        # Convert input cleanly without re-allocation
        # if x_pos is already an array.
        x_arr: NDArray = np.asarray(x_pos)

        # Fully vectorized calculation across all points simultaneously.
        raw_scores: NDArray = -np.sum(10.0 + 9.0 * np.cos(2.0 * np.pi * self.kappa * x_arr),
                                      axis=-1)

        # Check boundaries across the last dimension (D).
        in_bounds: NDArray = np.all((self.x_min <= x_arr) & (x_arr <= self.x_max),
                                    axis=-1)

        # Apply NaN mask directly to out-of-bounds array coordinates.
        return np.where(in_bounds, raw_scores, np.nan)
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
                                                              f_opt=-float(self.n_dim),
                                                              epsilon=epsilon, radius=radius)
        # Find the number of optima.
        num_optima: int = len(found_optima)

        # Return the tuple (number of found, total number)
        return num_optima, self.total_optima
    # _end_def_

# _end_class_
