import numpy as np
from star_pso.benchmarks.test_function import TestFunction


class FiveUnevenPeakTrap(TestFunction):
    """
    This function was originally proposed in:

    J.-P. Li, M. E. Balazs, G. T. Parks, and P. J. Clarkson,
    “A species conserving genetic algorithm for multimodal function optimization”
    Evolutionary Computation, vol. 10, no. 3, pp. 207–234, 2002.
    """

    def __init__(self) -> None:
        """
        Default initializer of the FiveUnevenPeakTrap class.
        """

        # Call the super initializer with the name.
        super().__init__(name="Five_Uneven_Peak_Trap",
                         x_min=0.0, x_max=30.0)
    # _end_def_

    def func(self, x_pos: np.ndarray) -> np.ndarray:
        """
        This is 1D function. There are two global optima,
        whilst the number of local optima is 3.

        :param x_pos: the current position(s) of the function.

        :return: the function value(s).
        """

        # Initialize function values to NaN
        f_value = np.full_like(x_pos, np.nan, dtype=float)

        # Conditions for the different ranges.
        range_1 = (0.0 <= x_pos) & (x_pos < 2.5)
        range_2 = (2.5 <= x_pos) & (x_pos < 5.0)
        range_3 = (5.0 <= x_pos) & (x_pos < 7.5)
        range_4 = (7.5 <= x_pos) & (x_pos < 12.5)
        range_5 = (12.5 <= x_pos) & (x_pos < 17.5)
        range_6 = (17.5 <= x_pos) & (x_pos < 22.5)
        range_7 = (22.5 <= x_pos) & (x_pos < 27.5)
        range_8 = (27.5 <= x_pos) & (x_pos <= 30.0)

        # Vectorized calculations based on conditions.
        f_value[range_1] = 80 * (2.50 - x_pos[range_1])
        f_value[range_2] = 64 * (x_pos[range_2] - 2.50)
        f_value[range_3] = 64 * (7.50 - x_pos[range_3])
        f_value[range_4] = 28 * (x_pos[range_4] - 7.50)
        f_value[range_5] = 28 * (17.5 - x_pos[range_5])
        f_value[range_6] = 32 * (x_pos[range_6] - 17.5)
        f_value[range_7] = 32 * (27.5 - x_pos[range_7])
        f_value[range_8] = 80 * (x_pos[range_8] - 27.5)

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
        return self.rng.uniform(self._x_min,
                                self._x_max,
                                size=(n_pos, 1))
    # _end_def_

# _end_class_

