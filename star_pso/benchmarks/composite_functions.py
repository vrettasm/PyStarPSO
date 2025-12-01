import numpy as np
from star_pso.population.particle import Particle
from star_pso.benchmarks.test_function import TestFunction
from star_pso.utils.auxiliary import identify_global_optima

# Basic function: 1
def f_sphere(x_pos: np.ndarray) -> np.ndarray:
    """
    Computes the sphere function at x_pos.
    """
    return np.sum(x_pos ** 2)
# _end_def_

# Basic function: 2
def f_griewank(x_pos: np.ndarray) -> np.ndarray:
    """
    Computes the Griewank function at x_pos.
    """
    # Get the size of the vector.
    n_dim = x_pos.size

    # Compute the sqrt[i], {1, 2, ..., D}.
    sqrt_i = np.sqrt(np.arange(1, n_dim + 1))

    # Get the final value.
    return np.sum(x_pos ** 2) / 4000 - np.prod(np.cos(x_pos / sqrt_i)) + 1
# _end_def_

# Basic function: 3
def f_rastrigin(x_pos: np.ndarray,
                kappa: float = 10.0) -> np.ndarray:
    """
    Computes the Rastrigin function at x_pos,
    with default kappa parameter.
    """
    return np.sum(x_pos ** 2 - kappa * np.cos(2.0 * np.pi * x_pos) + kappa)
# _end_def_

# Basic function: 4
def f_weierstrass(x_pos: np.ndarray, k_max: int = 9,
                  alpha: float = 0.5, beta: int = 3) -> np.ndarray:
    """
    Computes the Weierstrass function at x_pos,
    with default k_max, alpha and beta parameters.
    """
    # Get the size of the vector.
    n_dim = x_pos.size

    # Precalculate k-index values.
    k = np.arange(0, k_max + 1)

    # Precalculate: alpha^k
    alpha_k = alpha ** k

    # Precalculate: beta^k
    beta_k = beta ** k

    # Vectorized double summation.
    sum_x = np.sum(alpha_k[:, np.newaxis] * np.cos(2 * np.pi * beta_k[:, np.newaxis] * (x_pos + 0.5)),
                   axis=0).sum()

    # Combine the final result with the last summation.
    return sum_x - n_dim * np.sum(alpha_k * np.cos(np.pi * beta_k))
# _end_def_


# Define a dictionary will all the basic functions.
BASIC_FUNCTIONS: dict = {"f_sphere": f_sphere,
                         "f_griewank": f_griewank,
                         "f_rastrigin": f_rastrigin,
                         "f_weierstrass": f_weierstrass}


class CompositeFunction(TestFunction):
    """
    Generates a Composite Function using a weighting average
    of basic functions, as defined in the 'basic_f' dict.
    """

    def __init__(self, n_dim: int = 2, n_func: int = 4) -> None:
        """
        Default initializer of the CompositeFunction class.

        :param n_dim: (int) number of dimensions of the problem.

        :param n_func: (int) is the number of basic functions we
        want to include.

        :return: None.
        """
        # Ensure correct type.
        n_dim = int(n_dim)

        # Sanity check.
        if n_dim < 2:
            raise ValueError(f"{self.__class__.__name__}: needs at least 2 dimensions.")
        # _end_if_

        # Call the super initializer.
        super().__init__(name=f"CF_{n_dim}D",
                         n_dim=n_dim, x_min=-5.0, x_max=5.0)

        # Ensure correct type.
        n_func = int(n_func)

        # Sanity check.
        if not (2 <= n_func <= 20):
            raise ValueError(f"{self.__class__.__name__}: Number of functions is too high. "
                             f"Choose a value in [2, 20].")
        # _end_if_

        # Assign the value.
        self.n_func = n_func

        # Extract the keys.
        key_list = self.rng.choice(list(BASIC_FUNCTIONS.keys()), size=self.n_func)

        # Create a new list with basic functions.
        self.basic_f = [BASIC_FUNCTIONS[key] for key in key_list]

        # Display the order of functions.
        print("The basic functions are:")
        for n, func in enumerate(self.basic_f):
            print(f"{n}: {func}")
    # _end_def_

    def func(self, x_pos: np.ndarray,
             i_bias: float = 0.0, f_bias: float = 0.0) -> float | np.ndarray:
        """
        Describes the general framework for the construction of
        multimodal composition functions with several global optima.

        :param x_pos: the current position(s) of the function.

        :param i_bias: function value bias for each basic function.

        :param f_bias: function value bias for the composition function.

        :return: the function value(s).
        """
        # Initialize function value to NaN.
        f_value = np.full_like(x_pos, np.nan, dtype=float)

        # Check the valid function range.
        if np.all((self.x_min <= x_pos) & (x_pos <= self.x_max)):
            # Get the number of basic functions.
            n_func = self.n_func

            # Square of sigma values.
            sigma_sq = np.arange(1, n_func + 1) ** 2

            # Compute the weights:
            weights = np.exp(-0.5 * np.sum(x_pos ** 2) / (self.n_dim * sigma_sq))

            # Normalize them.
            weights /= np.sum(weights)

            # Get total evaluation of the composite function.
            f_total = np.sum([wi * (cf(x_pos / n_func) + i_bias)
                              for wi, cf in zip(weights, self.basic_f)])
            # Add the bias at the end.
            f_value = f_total + f_bias
        # _end_if_

        # Return the ndarray.
        return f_value
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
        # Get the global optima particles.
        found_optima = identify_global_optima(population, epsilon=epsilon,
                                              radius=0.1, f_opt=0.0)
        # Find the number of optima.
        num_optima = len(found_optima)

        # Return the tuple (number of found, total number)
        return num_optima, len(self.basic_f)
    # _end_def_

# _end_class_
