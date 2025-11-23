import numpy as np
from star_pso.benchmarks.test_function import TestFunction


class EqualMaxima(TestFunction):
    """
    This function was originally proposed in:

    K. Deb, “Genetic algorithms in multimodal function optimization
    (master thesis and tcga report no. 89002),” Ph.D. dissertation,
    Tuscaloosa: University of Alabama, The Clearinghouse for Genetic
    Algorithms, 1989.
    """

    def __init__(self) -> None:
        """
        Default initializer of the EqualMaxima class.
        """

        # Call the super initializer with the name.
        super().__init__(name="Equal_Maxima",
                         x_min=0.0, x_max=1.0)
    # _end_def_

    def func(self, x_pos: np.ndarray) -> np.ndarray:
        """
        This is 1D function. There are 5 global optima.

        :param x_pos: the current position(s) of the function.

        :return: the function value(s).
        """

        # Initialize function values to NaN.
        f_value = float("NaN")

        # Condition for the valid range.
        if (0.0 <= x_pos) & (x_pos <= 1.0):
            f_value = np.sin(5.0 * np.pi *x_pos)**6

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
        return self.rng.uniform(self._x_min, self._x_max, size=(n_pos, 1))
    # _end_def_

# _end_class_
