import time

from enum import Enum
from typing import Union
from functools import wraps
from functools import cache

from collections import namedtuple

import numpy as np
from numpy.typing import ArrayLike

from numba import njit

# Make a type alias for the position's type.
ScalarOrArray = Union[int, float, ArrayLike]

# Declare a named tuple with the parameters
# we want to use in the velocity equations:
# 1) 'w': inertia weight
# 2) 'c1': cognitive coefficient
# 3) 'c2': social coefficient
# 4) 'fipso': fully informed PSO (optional).
VOptions = namedtuple("VOptions",
                      ["w", "c1", "c2", "fipso"], defaults=[False])

class BlockType(Enum):
    """
    Description:
        BlockType enumeration defines the types that a data block can take.
    """
    FLOAT, INTEGER, BINARY, CATEGORICAL = range(4)
# _end_class_

class SpecialMode(Enum):
    """
    Description:
        SpecialMode enumeration defines specific modes that the GenericPSO
        can accommodate. These are handled internally (i.e. NOT by the end
        user) to call the evaluate_function methods and perform operations
        directly related to the specific versions of PSO method.
    """
    NORMAL, CATEGORICAL, JACK_OF_ALL_TRADES = range(3)
# _end_class_

def check_parameters(options: dict) -> None:
    """
    Checks that the options dictionary has all the additional
    parameters to estimate the velocities of the optimization
    algorithm.
        1)  'w': inertia weight
        2) 'c1': cognitive coefficient
        3) 'c2': social coefficient

    :param options: dictionary to check for missing parameters.

    :return: None.
    """
    # Sanity check.
    for key in {"w", "c1", "c2"}:
        # Make sure the right keys exist.
        if key not in options:
            raise KeyError(f"Option '{key}' is missing.")
        # _end_if_
# _end_def_

def np_median_entropy(x_pos: np.ndarray,
                      normal: bool = False) -> float:
    """
    Calculate the median entropy value of the input array.
    It is assumed that the input 'x_pos', represents the 2D
    array of "objects", where each row represents a particle
    and the columns contain the probability vectors, one for
    each of the categorical variables. In essence x_pos is a
    3D array.

    :param x_pos: 2D numpy array where each column represents
    a different optimization (categorical) variable.

    :param normal: If enabled the entropy values will be
    normalized using the maximum entropy value depending
    on the set of possible outcomes for each categorical
    variable.

    :return: The median entropy value of the swarm positions.
    """
    # Get the input columns.
    n_cols = x_pos.shape[1]

    # Preallocate entropy array.
    entropy_x = np.zeros(n_cols)

    # Average all the positional
    # values on the first rows.
    x_mean = x_pos.mean(0)

    # Process along the columns of the x_pos.
    for j in range(n_cols):

        # Normalize values to account
        # for probabilities in [0, 1].
        xj = x_mean[j] / x_mean[j].sum()

        # Calculate the entropy value.
        entropy_x[j] = -np.sum(np.where(xj > 0.0,
                                        xj * np.log(xj), 0.0))
        # Compute maximum entropy value.
        log_k = np.log(len(x_pos[0, j]))

        # Check for normalization.
        if normal and log_k != 0.0:
            entropy_x[j] /= log_k
    # _end_for_

    # Return the median value.
    return np.median(entropy_x).item()
# _end_def_

@njit
def nb_median_kl_divergence(x_pos: np.ndarray,
                            normal: bool = False) -> float:
    """
    Calculate the 'median KL divergence' value of the input array.
    It is assumed that each row is a distribution. To ensure that
    each row sums to 1.0, we perform the normalization.

    :param x_pos: 2D array where each column.

    :param normal: If enabled the KL values will be normalized using
    the maximum KL divergence from the data.

    :return: The median KL divergence of the swarm positions.
    """
    # Accumulate the KL divergence values.
    total_kl = []

    # Ensure the input array is normalized.
    x_pos /= x_pos.sum(axis=1)

    # Pairwise calculation of KL.
    for i, p in enumerate(x_pos):
        total_kl.extend(kl_divergence_array(p, x_pos[i + 1:]))
    # _end_for_

    # Convert to numpy array.
    kl_dist = np.array(total_kl)

    # Find the maximum KL.
    kl_max = kl_dist.max()

    # Check for normalization.
    if normal and kl_max != 0.0:
        kl_dist /= kl_max
    # _end_if_

    # Return the median value.
    return np.median(kl_dist).item()
# _end_def_

@cache
def linear_rank_probabilities(p_size: int) -> np.array:
    """
    Calculate the rank probability distribution over the
    population size.  The function is cached so repeated
    calls with the  same input should not  recompute the
    same array since the population size of the swarm is
    not expected to change.

    NOTE: Probabilities are returned in descending order.

    :param p_size: (int) population size.

    :return: (array) rank probability distribution in
    descending order.
    """
    # Sanity check.
    if p_size <= 0:
        raise ValueError("'p_size' must be a positive number.")
    # _end_if_

    # Calculate the sum of all the ranked swarm particles.
    sum_ranked_values = float(0.5 * p_size * (p_size + 1))

    # Calculate the linear ranked probabilities of each
    # particle in the swarm using their ranking position.
    return np.arange(1, p_size + 1) / sum_ranked_values
# _end_def_

@njit
def kl_divergence_item(p: np.array, q: np.array) -> float:
    """
    Calculates the Kullback-Leibler divergence between
    two distributions. Note that KL divergence is not
    symmetric, thus KL(p, q) != KL(q, p).

    NOTE: Both distributions 'p' and 'q' should already
    be normalized to sum to one.

    This method is equivalent to entropy(p, q) from scipy.stats,
    only it's around 10x faster.

    :param p: (np.array) probability distribution.
    :param q: (np.array) probability distribution.

    :return: (float) Kullback-Leibler divergence.
    """
    return np.sum(np.where(p != 0.0,
                           np.where(q != 0.0,
                                    p * np.log(p / q), np.nan), 0.0)).item()
# _end_def_

@njit
def kl_divergence_array(p: np.array, q: np.array) -> np.array:
    """
    Calculates the Kullback-Leibler divergence between two distributions.
    Note that KL divergence is not symmetric, thus KL(p, q) != KL(q, p).

    NOTE: Both distributions 'p' and 'q' should already be normalized to
    sum to one.

    This method is equivalent to entropy(p, q, axis=1) from scipy.stats,
    only it's around 10x faster.

    Example:
    # Create a random array.
    x = np.random.rand(10, 4)

    # Normalize to sum to 1.0
    x /= np.sum(x, axis=1).reshape(-1, 1)

    # Entropy (from scipy.stats).
    entropy(x[0], x[1:], axis=1)
    > array([0.12405413, 0.79411391, 0.5340511 , 0.13075877, 0.77733431,
    >        0.04979758, 0.34470209, 0.83185617, 0.29883382])

    # This (numba optimized) method.
    kl_divergence_array(x[0], x[1:])
    > array([0.12405413, 0.79411391, 0.5340511 , 0.13075877, 0.77733431,
    >        0.04979758, 0.34470209, 0.83185617, 0.29883382])

    :param p: (np.array) probability distribution.
    :param q: (np.array) probability distribution.

    :return: (np.array) Kullback-Leibler divergence.
    """
    return np.sum(np.where(p != 0.0,
                           np.where(q != 0.0,
                                    p * np.log(p / q), np.nan), 0.0), axis=1)
# _end_def_

@njit
def nb_median_hamming_distance(x_pos: np.ndarray,
                               normal: bool = False) -> float:
    """
    Compute the median Hamming distance of the input array.
    It is assumed that the input 'x_pos', represents the 2D
    array of particle binary positions {0, 1}.

    :param x_pos: a 2D array with the particle positions.

    :param normal: whether to compute the normalized average
    Hamming distance. This will yield a value between 0 and 1.
    * A value of '0' indicates that all rows are identical.
    * A value of '1' indicates that all rows are completely
    different across all positions.

    The normalization provides a clearer understanding of the
    diversity of the binary strings in relation to their length.

    :return: the (normalized) median Hamming distance value.
    """
    # Get the columns size.
    n_cols = x_pos.shape[1]

    # Initialize the counter.
    total_diff = []

    # Count the non-identical positions.
    for i, p in enumerate(x_pos):
        total_diff.extend(np.count_nonzero(p != x_pos[i + 1:], axis=1))
    # _end_for_

    # Convert the list to an array of floats.
    x_diff = np.array([float(k) for k in total_diff])

    # Check for normalization.
    if normal:
        # In this case the number of columns represents
        # the  maximum  hamming  distance where all the
        # positions between two particles are different.
        x_diff /= n_cols
    # _end_if_

    # Return the median value.
    return np.median(x_diff).item()
# _end_def_

@njit
def nb_median_euclidean_distance(x_pos: np.ndarray,
                                 normal: bool = False) -> float:
    """
    Calculate the median Euclidean distance of swarm particles.

    :param x_pos: (np.ndarray) A 2D array of shape (n_particles,
    n_features) representing the positions of the swarm.

    :param normal: (bool) if "True", normalize the distances by
    their maximum distance.

    :return: the median Euclidean distance.
    """
    # Collect all the distances.
    total_dist = []

    # Scan the positions array.
    for i, p in enumerate(x_pos):
        total_dist.extend(np.sqrt(np.sum((p - x_pos[i + 1:]) ** 2, axis=1)))
    # _end_for_

    # Convert to array.
    x_dist = np.array(total_dist)

    # Find the maximum distance.
    d_max = x_dist.max()

    # Normalize the distances with d_max.
    if normal and d_max != 0.0:
        x_dist /= d_max
    # _end_if_

    # Return the median value.
    return np.median(x_dist).item()
# _end_def_

@njit
def nb_median_taxicab_distance(x_pos: np.ndarray,
                               normal: bool = False) -> float:
    """
    Calculate the median TaxiCab (Manhattan) distance of swarm particles.

    :param x_pos: (array) A 2D numpy array (n_particles, n_features)
    representing the positions of the swarm.

    :param normal: (bool) if "True", normalize the distances by their
    maximum distance.

    :return: the median taxicab distance.
    """
    # Collect all the distances.
    total_dist = []

    # Scan the positions array.
    for i, p in enumerate(x_pos):
        total_dist.extend(np.sum(np.abs(p - x_pos[i + 1:]), axis=1))
    # _end_for_

    # Convert to array.
    x_dist = np.array([float(k) for k in total_dist])

    # Find the maximum distance.
    d_max = x_dist.max()

    # Normalize the distances with d_max.
    if normal and d_max != 0.0:
        x_dist /= d_max
    # _end_if_

    # Return the median value.
    return np.median(x_dist).item()
# _end_def_

@njit
def nb_clip_array(x_new, lower_limit, upper_limit):
    """
    Local version of numba clip which limits the values of an array.
    Given an interval values outside the interval are clipped to the
    interval edges.

    :param x_new: array values to be clipped.

    :param lower_limit: lower limit.

    :param upper_limit: upper limit.

    :return: the clipped array values.
    """
    return np.minimum(np.maximum(x_new, lower_limit),
                      upper_limit)
# _end_def_

@njit
def nb_clip_item(x_new, lower_limit, upper_limit):
    """
    Local version of numba clip which limits the values of a scalar.
    Given an interval values outside the interval are clipped to the
    interval edges. The final value is returned with item().

    :param x_new: scalar value to be clipped.

    :param lower_limit: lower limit.

    :param upper_limit: upper limit.

    :return: the clipped item value.
    """
    return np.minimum(np.maximum(x_new, lower_limit),
                      upper_limit).item()
# _end_def_

def time_it(func):
    """
    Timing decorator function.

    :param func: the function we want to time.

    :return: the time wrapper method.
    """

    @wraps(func)
    def time_it_wrapper(*args, **kwargs):
        """
        Wrapper function that times the execution of the input function.

        :param args: function positional arguments.

        :param kwargs: function keywords arguments.

        :return: the output of the wrapper function.
        """
        # Initial time instant.
        time_t0 = time.perf_counter()

        # Run the function we want to time.
        result = func(*args, **kwargs)

        # Final time instant.
        time_tf = time.perf_counter()

        # Print final duration in seconds.
        print(f"{func.__name__ }: "
              f"elapsed time = {(time_tf - time_t0):.3f} seconds.")

        return result
    # _end_def_
    return time_it_wrapper
# _end_def_

def pareto_front(points: np.array) -> np.array:
    """
    Simple function that calculates the pareto (optimal)
    front points from a given input points list.

    :param points: array of points [(fx1, fx2, ..., fxn),
                                    (fy1, fy2, ..., fyn),
                                    ....................,
                                    (fk1, fk2, ..., fkn)]

    :return: Array of points that lie on the pareto front.
    """

    # Number of points.
    number_of_points = len(points)

    # Set all the flags initially to True.
    is_pareto = np.full(number_of_points, True, dtype=bool)

    # Scan all the points.
    for i, point_i in enumerate(points):

        # Compare against all others.
        for j, point_j in enumerate(points):

            # Do not compare point with itself!
            # If the condition is satisfied the
            # point_i does not lie on the front.
            if i != j and np.all(point_i >= point_j):
                # Change the flag value.
                is_pareto[i] = False

                # Break the internal loop.
                break
            # _end_if_
    # _end_for_
    return points[is_pareto]
# _end_def_

@cache
def get_spread_method() -> dict:
    """
    Create a dictionary with block types as keys and
    their corresponding spread estimation methods as
    values.

    :return: a cached dictionary with functions that
    correspond to the correct block types.
    """
    return {BlockType.FLOAT: nb_median_euclidean_distance,
            BlockType.BINARY: nb_median_hamming_distance,
            BlockType.INTEGER: nb_median_euclidean_distance,
            BlockType.CATEGORICAL: nb_median_kl_divergence}
# _end_def_
