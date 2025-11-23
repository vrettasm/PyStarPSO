import numpy as np
from star_pso.population.particle import Particle
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

        # Call the super initializer with the name and the limits.
        super().__init__(name="Himmelblau", x_min=-6.0, x_max=6.0)
    # _end_def_

    def func(self, x_pos: np.ndarray) -> np.ndarray:
        """
        This is 2D function with is 4 global optima.

        :param x_pos: the current position(s) of the function.

        :return: the function value(s).
        """

        # Initialize function values to NaN.
        f_value = float("NaN")

        # Check the valid function range.
        if np.all((self.x_min <= x_pos) & (x_pos <= self.x_max)):
            # Separate the two variables.
            x, y = x_pos

            # Calculate the function value.
            f_value = 200.0 - (x**2 + y - 11)**2 - (x + y**2 - 7)**2

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

    def global_optima(self, population: list[Particle]) -> None:
        """
        Calculates the global optimum found in the input population.
        """
        # Get the global optima particles.
        found_optima = self.global_optima_found(population, epsilon=1.0E-3,
                                                radius=0.01, f_opt=200.0)
        # Display the number of global optima found.
        print(f"Found {len(found_optima)} out of 4 global optima.")
    # _end_def_

# _end_class_
