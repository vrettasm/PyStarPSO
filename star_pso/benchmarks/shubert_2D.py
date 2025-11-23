import numpy as np
from star_pso.benchmarks.test_function import TestFunction


class Shubert2D(TestFunction):
    """
    This function was originally proposed in:

    Z. Michalewicz, Genetic Algorithms + Data Structures = Evolution Programs.
    New York: Springer-Verlag, New York, 1996.
    """

    def __init__(self) -> None:
        """
        Default initializer of the SixHumpCamelBack class.
        """

        # Call the super initializer with the name.
        super().__init__(name="Shubert_2D",
                         x_min=-10.0, x_max=+10.0)
    # _end_def_

    def func(self, x_pos: np.ndarray) -> np.ndarray:
        """
        This is a 2D function with 18 global optimal values.

        :param x_pos: the current position(s) of the function.

        :return: the function value(s).
        """

        # Initialize function values to NaN.
        f_value = float("NaN")

        # Separate the two variables.
        x, y = x_pos

        # Conditions for the different ranges.
        x_range = (-10.0 <= x) & (x <= +10.0)
        y_range = (-10.0 <= y) & (y <= +10.0)

        # Vectorized calculations based on the condition.
        if x_range & y_range:
            # Range 1 to 6.
            i = np.arange(1, 6)

            # Calculate the first summation over each x.
            sum_x = np.sum(i * np.cos((i + 1) * x + i), axis=0)

            # Calculate the second summation over each y.
            sum_y = np.sum(i * np.cos((i + 1) * y + i), axis=0)

            # Get the product of both sums.
            f_value = sum_x * sum_y
        # _end_if_

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

# _end_class_
