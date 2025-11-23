import numpy as np
from star_pso.benchmarks.test_function import TestFunction


class Vincent(TestFunction):
    """
    This function was originally proposed in:

    O. Shir and T. Back, “Niche radius adaptation in the cms-es niching algorithm”,
    in Parallel Problem-Solving from Nature - PPSN IX, 9th International Conference
    (LNCS 4193). Reykjavík, Iceland: Springer, 2006, pp. 142 – 151.
    """

    def __init__(self, n_dim: int = 2) -> None:
        """
        Default initializer of the Vincent D class.
        """

        # Ensure correct type.
        n_dim = int(n_dim)

        # Call the super initializer with the name.
        super().__init__(name=f"Vincent_{n_dim}D",
                         x_min=0.25, x_max=10.0)
        # Sanity check.
        if n_dim < 2:
            raise ValueError("Vincent D needs to be at least 2 dimensions.")
        # _end_if_

        # Assign the number of dimensions.
        self.n_dim = n_dim
    # _end_def_

    def func(self, x_pos: np.ndarray) -> np.ndarray:
        """
        This is a nD function with 6^n_dim global optimal values.

        :param x_pos: the current position(s) of the function.

        :return: the function value(s).
        """

        # Initialize function values to NaN.
        f_value = np.nan

        # Check the valid function range.
        if np.all((self.x_min <= x_pos) & (x_pos <= self.x_max)):
            # Compute the function value.
            f_value = np.sum(np.sin(10.0 * np.log(x_pos)), axis=0)

        # Return the ndarray.
        return f_value/self.n_dim
    # _end_def_

    def initial_random_positions(self, n_pos: int = 50) -> np.ndarray:
        """
        Generate an initial set of uniformly random sampled positions
        within the minimum / maximum bounds of the test problem.

        :param n_pos: the number of positions to generate.

        :return: a uniformly sampled set of random positions.
        """
        # Draw uniform random samples for the initial points.
        return self.rng.uniform(self._x_min, self._x_max, size=(n_pos, self.n_dim))
    # _end_def_

# _end_class_
