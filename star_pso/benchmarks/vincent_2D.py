import numpy as np
from star_pso.benchmarks.test_function import TestFunction


class Vincent2D(TestFunction):
    """
    This function was originally proposed in:

    O. Shir and T. Ba ̈ck, “Niche radius adaptation in the cms-es niching algorithm”,
    in Parallel Problem-Solving from Nature - PPSN IX, 9th International Conference
    (LNCS 4193). Reykjavík, Iceland: Springer, 2006, pp. 142 – 151.
    """

    def __init__(self) -> None:
        """
        Default initializer of the Vincent2D class.
        """

        # Call the super initializer with the name.
        super().__init__(name="Vincent_2D",
                         x_min=0.25, x_max=10.0)
    # _end_def_

    def func(self, x_pos: np.ndarray) -> np.ndarray:
        """
        This is a 2D function with 36 global optimal values.

        :param x_pos: the current position(s) of the function.

        :return: the function value(s).
        """

        # Initialize function values to NaN.
        f_value = float("NaN")

        # Separate the two variables.
        x, y = x_pos

        # Conditions for the different ranges.
        x_range = (0.25 <= x) & (x <= 10.0)
        y_range = (0.25 <= y) & (y <= 10.0)

        # Vectorized calculations based on the condition.
        if x_range & y_range:
            # Compute the function value.
            f_value = - (np.sin(10.0 * np.log(x)) + np.sin(10.0 * np.log(y))) / 2.0
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
