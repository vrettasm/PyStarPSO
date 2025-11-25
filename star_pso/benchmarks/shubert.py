import numpy as np
from star_pso.population.particle import Particle
from star_pso.benchmarks.test_function import TestFunction


class Shubert(TestFunction):
    """
    This function was originally proposed in:

    Z. Michalewicz, Genetic Algorithms + Data Structures = Evolution Programs.
    New York: Springer-Verlag, New York, 1996.
    """

    def __init__(self, n_dim: int = 2) -> None:
        """
        Default initializer of the Shubert D class.

        :param n_dim: Number of dimensions of the problem.

        :return: None.
        """
        # Call the super initializer with the name and the limits.
        super().__init__(name=f"Shubert_{n_dim}D", x_min=-10.0, x_max=+10.0)

        # Ensure correct type.
        n_dim = int(n_dim)

        # Sanity check.
        if n_dim < 2:
            raise ValueError("Shubert needs at least 2 dimensions.")
        # _end_if_

        # Assign the number of dimensions.
        self.n_dim = n_dim
    # _end_def_

    def func(self, x_pos: np.ndarray) -> np.ndarray:
        """
        This is a multidimensional function with 'n_dim * 3^n_dim'
        global optimal values.

        :param x_pos: the current position(s) of the function.

        :return: the function value(s).
        """
        # Initialize function value to NaN.
        f_value = np.nan

        # Check the valid function range.
        if np.all((self.x_min <= x_pos) & (x_pos <= self.x_max)):
            # Range 1 to 6.
            i = np.arange(1, 6)

            # Get the product of the sums.
            f_value = -np.prod(np.sum(i[:, np.newaxis] * np.cos((i[:, np.newaxis] + 1) * x_pos +
                                                                 i[:, np.newaxis]), axis=0))
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
        return self.rng.uniform(self._x_min, self._x_max, size=(n_pos, self.n_dim))
    # _end_def_

    def global_optima(self, population: list[Particle]) -> tuple[int, int]:
        """
        Calculates the global optimum found in the input population.

        :param population: the population to search the global optimum.

        :return: a tuple with the number of global optima found and the
        total number that exist.
        """
        # Sanity check.
        if self.n_dim > 3:
            raise ValueError(f"Unknown values for D = {self.n_dim}")
        # _end_if_

        # Calculate the total global optima along with
        # the f_opt for the given number of dimensions.
        if self.n_dim == 2:
            total_optima, f_opt = 18, 186.7309088
        else:
            total_optima, f_opt = 81, 2709.093505
        # _end_if_

        # Get the global optima particles.
        found_optima = self.global_optima_found(population, epsilon=1.0E-3,
                                                radius=0.5, f_opt=f_opt)
        # Find the number of optima.
        num_optima = len(found_optima)

        # Return the tuple (number of found, total number)
        return num_optima, total_optima
    # _end_def_

# _end_class_
