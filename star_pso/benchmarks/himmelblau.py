import numpy as np
from star_pso.benchmarks.test_function import TestFunction


class Himmelblau(TestFunction):
    """
    This function was originally proposed in:

    K. Deb, “Genetic algorithms in multimodal function optimization
    (master thesis and tcga report no. 89002),” Ph.D. dissertation,
    Tuscaloosa: University of Alabama, The Clearinghouse for Genetic
    Algorithms, 1989.
    """

    def __init__(self) -> None:
        """
        Default initializer of the Himmelblau class.
        """

        # Call the super initializer with the name.
        super().__init__(name="Himmelblau",
                         x_min=-6.0, x_max=6.0)
    # _end_def_

    def func(self, x_pos: np.ndarray) -> np.ndarray:
        """
        This is 2D function with is 4 global optima.

        :param x_pos: the current position(s) of the function.

        :return: the function value(s).
        """

        # Initialize function values to NaN.
        f_value = np.full_like(x_pos, np.nan, dtype=float)

        # Make separate views for the two variables.
        x, y = x_pos[:, 0], x_pos[:, 1]

        # Conditions for the different ranges.
        x_range = (-6.0 <= x) & (x <= +6.0)
        y_range = (-6.0 <= y) & (y <= +6.0)

        # We need both variables to be within range.
        in_range = x_range & y_range

        # Vectorized calculations based on conditions.
        f_value[in_range] = (200.0 - (x[in_range]**2 + y[in_range] - 11)**2 -
                             (x[in_range] + y[in_range]**2 - 7)**2)
        # Return the ndarray.
        return f_value
    # _end_def_

    def initial_random_positions(self, n_pos: int = 50) -> np.ndarray:
        """
        Generate the initial set of random positions within the minimum
        / maximum bounds of the test problem.

        :param n_pos: the number of positions to generate.

        :return: a uniformly sampled set of random positions.
        """
        # Draw uniform random samples for the initial points.
        return self.rng.uniform(self._x_min, self._x_max, size=(n_pos, 2))
    # _end_def_

# _end_class_
