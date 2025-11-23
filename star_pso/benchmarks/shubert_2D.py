import numpy as np
from star_pso.benchmarks.test_function import TestFunction


class Shubert2D(TestFunction):
    """
    This function was originally proposed in:

    Z. Michalewicz, Genetic Algorithms + Data Structures = Evolution Programs.
    New York: Springer-Verlag, New York, 1996.
    """

    def __init__(self, n_dim: int = 2) -> None:
        """
        Default initializer of the SixHumpCamelBack class.

        :param n_dim: Number of dimensions of the problem.

        :return: None.
        """

        # Call the super initializer with the name.
        super().__init__(name="Shubert_2D",
                         x_min=-10.0, x_max=+10.0)

        # Assign the number of dimensions.
        self.n_dim = max(int(n_dim), 2)
    # _end_def_

    def func(self, x_pos: np.ndarray) -> np.ndarray:
        """
        This is a multidimensional function with 'n_dim * 3^n_dim'
        global optimal values.

        :param x_pos: the current position(s) of the function.

        :return: the function value(s).
        """

        # Initialize function values to NaN.
        f_value = float("NaN")

        # Check the valid function range.
        if np.all((-10.0 <= x_pos) & (x_pos <= +10.0)):
            # Range 1 to 6.
            i = np.arange(1, 6)

            # Set the initial value to one.
            sum_x = np.ones(1)

            # Calculate the summation over each x.
            for xi in x_pos:
                sum_x *= np.sum(i * np.cos((i + 1) * xi + i), axis=0)

            # Get the product of both sums.
            f_value = -sum_x
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
