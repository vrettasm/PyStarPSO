import numpy as np
from numba import njit
from numpy.typing import NDArray

from star_pso.population.particle import Particle
from star_pso.benchmarks.test_function import TestFunction
from star_pso.utils.auxiliary import (identify_global_optima,
                                      calculate_dynamic_radius)

# Basic function: 1
@njit(cache=True, fastmath=True)
def f_sphere(x_pos: NDArray) -> NDArray:
    """
    Computes the sphere function at x_pos.
    """
    return np.sum(x_pos ** 2, axis=-1)
# _end_def_

# Basic function: 2
@njit(cache=True, fastmath=True)
def f_griewank(x_pos: NDArray) -> NDArray:
    """
    Computes the Griewank function at x_pos.
    """
    # Ensure input is 2D array.
    x_pos_2d: NDArray = np.atleast_2d(x_pos)

    # Extract the shape variables.
    n_samples, n_dim = x_pos_2d.shape

    # Precompute the square root values.
    sqrt_i = np.sqrt(np.arange(1, n_dim + 1))

    # Pre-allocate output vector
    f_value = np.empty(n_samples, dtype=float)

    # Compute the function for each row (sample).
    for n in range(n_samples):

        # Extract row slice.
        row = x_pos_2d[n]

        # Sum all elements.
        sum_term = np.sum(row * row) / 4000.0

        # Compute the product of the elements.
        prod_term = np.prod(np.cos(row / sqrt_i))

        # Assign the new function value.
        f_value[n] = sum_term - prod_term + 1.0
    # _end_for_

    return f_value
# _end_def_

# Basic function: 3
@njit(cache=True, fastmath=True)
def f_rastrigin(x_pos: NDArray,
                kappa: float = 10.0) -> NDArray:
    """
    Computes the Rastrigin function at x_pos,
    with default kappa parameter.
    """
    return np.sum((x_pos * x_pos) -
                  kappa * np.cos(2.0 * np.pi * x_pos) + kappa,
                  axis=-1)
# _end_def_

# Basic function: 4
@njit(cache=True, fastmath=True)
def f_weierstrass(x_pos: NDArray, k_max: int = 9,
                  alpha: float = 0.5, beta: int = 3) -> NDArray:
    """
    Computes the Weierstrass function at x_pos, with default k_max,
    alpha and beta parameters.
    """
    # Ensure input is NDArray.
    x_pos: NDArray = np.atleast_2d(x_pos)

    # Get the shape of input.
    n_samples, n_dim = x_pos.shape

    # Initialize the constant.
    const_k: float = 0.0

    # Compute the k values.
    k_values: NDArray = np.arange(0, k_max + 1)

    # Compute the constant.
    for k in k_values:
        const_k += (alpha ** k) * np.cos(np.pi * (beta ** k))
    # _end_for_

    # Return array.
    f_value: NDArray = np.empty(n_samples, dtype=float)

    for n in range(n_samples):

        # Get the n-th sample.
        xn = x_pos[n]

        # Partial f_value.
        partial_f: float = 0.0

        # Go through all dimensions.
        for i in range(n_dim):

            xi = xn[i] + 0.5

            for k in k_values:
                partial_f += (alpha ** k) * np.cos(2.0 * np.pi * (beta ** k) * xi)
        # _end_for_

        f_value[n] = partial_f
    # _end_for_

    return f_value - (n_dim * const_k)
# _end_def_

# Basic function: 5
@njit(cache=True, fastmath=True)
def f_ackley(x_pos: NDArray, alpha: float = 20.0,
             beta: float = 0.2) -> NDArray:
    """
    Computes the Ackley function at x_pos,
    with default alpha and beta parameters.
    """
    # Ensure input is NDArray.
    x_pos: NDArray = np.atleast_2d(x_pos)

    # Get the shape of input.
    _, n_dim = x_pos.shape

    # Compute the first part of the equation.
    f_total: NDArray = -alpha * np.exp(-beta * np.sqrt(np.sum(x_pos * x_pos,
                                                              axis=1) / n_dim))
    # Update it with the second part.
    f_total -= np.exp(np.sum(np.cos(2.0 * np.pi * x_pos),
                             axis=1) / n_dim)
    # Return final result.
    return f_total + alpha + np.e
# _end_def_

# Basic function: 6
@njit(cache=True, fastmath=True)
def f_alpine(x_pos: NDArray) -> NDArray:
    """
    Computes the Alpine function at x_pos.
    """
    # Ensure input is at least 2D.
    x_pos: NDArray = np.atleast_2d(x_pos)

    # Return final result.
    return np.sum(np.abs(x_pos * np.sin(x_pos)) + 0.1 * x_pos, axis=1)
# _end_def_

# Auxiliary dictionary with the basis functions.
BASIS_FUNCTIONS: dict = {"f_ackley": f_ackley,
                         "f_alpine": f_alpine,
                         "f_sphere": f_sphere,
                         "f_griewank": f_griewank,
                         "f_rastrigin": f_rastrigin,
                         "f_weierstrass": f_weierstrass}

"""
Define a dictionary with all the basis functions:

    - Ackley
    - Alpine
    - Sphere
    - Griewank
    - Rastrigin
    - Weierstrass

NOTE: These functions are not vectorized!
"""

# Public interface.
__all__ = ["CompositeFunction", "BASIS_FUNCTIONS"]


class CompositeFunction(TestFunction):
    """
    Generates a Composite Function using a weighted average of basic
    functions, as defined in the 'BASIS_FUNCTIONS' dictionary.
    """

    def __init__(self, n_dim: int = 2, n_func: int | list = 4,
                 x_min: float = -5.0, x_max: float = 5.0) -> None:
        """
        Default initializer of the CompositeFunction class.

        :param n_dim: (int) number of dimensions of the problem.

        :param n_func: (int | list) is either the number of basic functions
                       that we want to include selected at random, or a list
                       with specific functions that we will include in the
                       given order.

        :param x_min: (float) the lower bound values of the search space.

        :param x_max: (float) the upper bound values of the search space.

        :return: None.
        """
        # Ensure correct type.
        n_dim: int = int(n_dim)

        # Sanity check.
        if n_dim < 2:
            raise ValueError(f"{self.__class__.__name__}: needs at least 2 dimensions.")
        # _end_if_

        # Call the super initializer.
        super().__init__(name=f"CF_{n_dim}D",
                         n_dim=n_dim, x_min=x_min, x_max=x_max)

        # Ensure correct type.
        if isinstance(n_func, int) and (2 <= n_func <= 20):
            # Create a new list by sampling randomly basic functions.
            self.basis_f: list = self.rng.choice(list(BASIS_FUNCTIONS.values()),
                                                 size=n_func, replace=True).tolist()
        elif isinstance(n_func, list):
            try:
                # Create a new list with the given basic functions.
                self.basis_f: list = [BASIS_FUNCTIONS[key] for key in n_func]
            except KeyError as ex:
                raise KeyError(f"Unknown basic function. "
                               f"Valid options are: {list(BASIS_FUNCTIONS.keys())}") from ex
        else:
            raise TypeError(f"{self.__class__.__name__}: "
                            f"'n_func' must either be an int in [2, 20], or a list.")
    # _end_def_

    @staticmethod
    def compute_weights(x_pos: NDArray, sigma: NDArray) -> NDArray:
        """
        Calculates a set of weights (one for each function).

        :param x_pos: (ndarray) the position that we are evaluating
                      the functions.

        :param sigma: (ndarray) are used to control each function's
                      range. A small value gives a narrow range for
                      the function.

        :return: a (ndarray) of normalized weights.
        """
        # Ensure the input is 2D array.
        x_pos = np.atleast_2d(x_pos)

        # Number of dimension.
        n_dim: NDArray = x_pos.shape[1]

        # Precompute the denominator array.
        denominator: NDArray = (2.0 * n_dim * sigma * sigma)

        # Compute the sum of squares.
        sum_sq: NDArray = np.sum(x_pos * x_pos, axis=1)[:, np.newaxis]

        # Broadcast division and exponential.
        weights: NDArray = np.exp(-sum_sq/denominator)

        # Track maximum values along the rows.
        i_max: int = np.argmax(weights, axis=1)

        row_indices: NDArray = np.arange(len(weights))
        w_max: NDArray = weights[row_indices, i_max]

        # Apply the transformation formula.
        w_max_power_10: NDArray = w_max ** 10
        weights *= (1.0 - w_max_power_10)[:, np.newaxis]

        # Restore the original max values
        # safely using advanced indexing.
        weights[row_indices, i_max] = w_max

        # Finally return the normalized values.
        return weights / weights.sum(axis=1, keepdims=True)
    # _end_def_

    def func(self, x_pos: NDArray,
             i_bias: float = 0.0, f_bias: float = 0.0) -> float | NDArray:
        """
        Describes the general framework for the construction of
        multimodal composition functions with several global optima.

        :param x_pos: the current position(s) of the function.

        :param i_bias: function value bias for each basic function.

        :param f_bias: function value bias for the composition function.

        :return: the function value(s).
        """
        # Initialize function value to NaN.
        f_value: NDArray = np.full_like(x_pos, np.nan, dtype=float)

        # Check the valid function range.
        if np.all((self.x_min <= x_pos) & (x_pos <= self.x_max)):
            # Get the number of basis functions.
            num_f: int = len(self.basis_f)

            # Sigma values for simplicity are set to one.
            sigma: NDArray = np.ones(num_f, dtype=float)

            # Calculate the weights of the functions.
            weights: NDArray = CompositeFunction.compute_weights(x_pos, sigma)

            # Initialize function value.
            f_total: float = 0.0

            # Get total evaluation of the composite function.
            f_total = np.sum([
                wi * (fi(x_pos / num_f) + i_bias)
                for wi, fi in zip(weights, self.basis_f)
            ])
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
        # Calculate the radius dynamically.
        radius: float = calculate_dynamic_radius(self.x_min, self.x_max)

        # Get the global optima particles.
        found_optima: list[Particle] = identify_global_optima(population,
                                                              f_opt=0.0,
                                                              epsilon=epsilon,
                                                              radius=radius)
        # Find the number of optima.
        num_optima: int = len(found_optima)

        # Return the tuple (number of found, total number)
        return num_optima, 1
    # _end_def_

    def __str__(self) -> str:
        """
        Returns a string representation of the CompositeFunction.
        """
        # Initialize the return string.
        cf_str: str = (f"{self.name}(x_min={self.x_min},"
                       f"x_max={self.x_max})\n")

        # Append all the basis functions.
        for n, func in enumerate(self.basis_f):
            cf_str += f"F{n} -> {func.__name__}\n"

        # Return the new string.
        return cf_str
    # _end_def_

# _end_class_
