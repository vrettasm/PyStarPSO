import time
import numpy as np
from enum import Enum
from functools import wraps


class BlockType(Enum):
    """
    Description:
        BlockType enumeration defines the types that a data block can take.
    """
    FLOAT, INTEGER, BINARY, CATEGORICAL = range(4)
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
            raise KeyError(f"Option '{key}' is missing. ")
        # _end_if_
# _end_def_

def my_clip(x_new, lower_limit, upper_limit):
    """
    Local version of numpy clip which limits
    the values in an a scalar (array). Given
    an interval, values outside the interval
    are clipped to the interval edges.

    :param x_new: scalar value to be clipped.

    :param lower_limit: lower limit.

    :param upper_limit: upper limit.

    :return: the clipped value.
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

def pareto_front(points: np.typing.ArrayLike) -> np.typing.ArrayLike:
    """
    Simple function that calculates the pareto (optimal)
    front points from a given input points list.

    :param points: array of points [(fx1, fx2, ..., fxn),
                                    (fy1, fy2, ..., fyn),
                                    ....................,
                                    (fk1, fk2, ..., fkn)]

    :return: An array of points that lie on the pareto front.
    """

    # Number of points.
    number_of_points = points.shape[0]

    # Set all the flags initially to True.
    is_pareto = np.full(number_of_points, True)

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

        # _end_internal_for_

    return points[is_pareto]
# _end_def_
