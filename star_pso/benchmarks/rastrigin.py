import numpy as np
from star_pso.population.particle import Particle
from star_pso.benchmarks.test_function import TestFunction


class Rastrigin(TestFunction):
    """
    This function was originally proposed in:

    A. Saha and K. Deb, “A bi-criterion approach to multimodal optimization:
    self-adaptive approach,” in Proceedings of the 8th international conference
    on Simulated evolution and learning, ser. SEAL-10. Berlin, Heidelberg:
    Springer-Verlag, 2010, pp. 95–104.
    """

    def __init__(self, n_dim: int = 2) -> None:
        """
        Default initializer of the D dimensional Rastrigin.

        :param n_dim: Number of dimensions of the problem.

        :return: None.
        """
        # Call the super initializer with the name and the limits.
        super().__init__(name=f"Rastrigin_{n_dim}D", x_min=0.0, x_max=1.0)

        # Ensure correct type.
        n_dim = int(n_dim)

        # Sanity check.
        if n_dim < 2:
            raise ValueError("Rastrigin needs at least 2 dimensions.")
        # _end_if_

        # Assign the number of dimensions.
        self.n_dim = n_dim

        # Placeholder for the total optimal values.
        self.total_optima = None
    # _end_def_

    def func(self, x_pos: np.ndarray) -> np.ndarray:
        """
        This is a multidimensional function with 'M' global optimal values.

        :param x_pos: the current position(s) of the function.

        :return: the function value(s).
        """
        # Initialize function value to NaN.
        f_value = np.nan

        # Check the valid function range.
        if np.all((self.x_min <= x_pos) & (x_pos <= self.x_max)):
            # Range 1 to D+1.
            k = np.arange(1, self.n_dim + 1)

            # Compute the total number of optimal values.
            self.total_optima = np.prod(k)

            # Get the sum.
            f_value = -np.sum(10.0 + 9.0 * np.cos(2.0 * np.pi * k * x_pos), axis=0)
        # _end_if_

        # Return the ndarray.
        return f_value
    # _end_def_

    def sample_random_positions(self, n_pos: int = 100) -> np.ndarray:
        """
        Generate an initial set of uniformly random sampled positions
        within the minimum / maximum bounds of the test problem.

        :param n_pos: the number of positions to generate.

        :return: a uniformly sampled set of random positions.
        """
        # Draw uniform random samples for the initial points.
        return self.rng.uniform(self._x_min, self._x_max, size=(n_pos, self.n_dim))
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
        if self.total_optima is None:
            raise ValueError(f"Unknown optimal values.")
        # _end_if_

        # Get the global optima particles.
        found_optima = self.search_global_optima(population, epsilon=epsilon,
                                                 radius=0.01, f_opt=-float(self.n_dim))
        # Find the number of optima.
        num_optima = len(found_optima)

        # Return the tuple (number of found, total number)
        return num_optima, self.total_optima
    # _end_def_

# _end_class_
