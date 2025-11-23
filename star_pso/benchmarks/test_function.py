import numpy as np
from collections import defaultdict
from numpy.random import default_rng, Generator

class TestFunction(object):
    """
    Description:
        All benchmark test functions should inherit from this class.
        Here is provided the interface for all the test problems.
    """

    # Make a random number generator.
    rng: Generator = default_rng()

    # Object variables.
    __slots__ = ("_name", "_x_min", "_x_max")

    def __init__(self, name: str,
                 x_min: float | np.ndarray,
                 x_max: float | np.ndarray) -> None:
        """
        Description:
        Default initializer of the "TestFunction" class.
        """

        # Sanity check.
        if np.any(x_min >= x_max):
            raise ValueError(f"{self.__class__.__name__}: "
                             f"x_min must be smaller than x_max.")
        # _end_if_

        # Assign the function name.
        self._name = name

        # Assign the minimum value(s).
        self._x_min = x_min

        # Assign the maximum value(s).
        self._x_max = x_max
    # _end_def_

    @property
    def name(self) -> str:
        """
        Accessor (getter) of the test function name.

        :return: string name of the test function.
        """
        return self._name
    # _end_def_

    @property
    def x_min(self) -> float | np.ndarray:
        """
        Accessor (getter) of the lower bounds of the test function.

        :return: numpy array with minimum values.
        """
        return self._x_min
    # _end_def_ù

    @property
    def x_max(self) -> float | np.ndarray:
        """
       Accessor (getter) of the upper bounds of the test function.

       :return: numpy array with maximum values.
       """
        return self._x_max
    # _end_def_

    @classmethod
    def set_seed(cls, new_seed=None) -> None:
        """
        Sets a new seed for the random number generator.

        :param new_seed: New seed value (default=None).

        :return: None.
        """
        # Re-initialize the class variable.
        cls.rng = default_rng(seed=new_seed)
    # _end_def_

    def func(self, x_pos: np.ndarray) -> np.ndarray:
        """
        This method will implement the objective function to be optimized.

        :return: None.
        """
        raise NotImplementedError(f"{self.__class__.__name__}: "
                                  f"You should implement this method!")
    # _end_def_

    def initial_random_positions(self, n_pos: int) -> np.ndarray:
        """
        This method will create an initial set of random positions
        that will be passed to the PSO algorithm to form the swarm.

        :param n_pos: Number of random positions to create.

        :return: None.
        """
        raise NotImplementedError(f"{self.__class__.__name__}: "
                                  f"You should implement this method!")
    # _end_def_

    @staticmethod
    def global_optima_found(x_pos: np.ndarray, modes: list | np.ndarray,
                            radius: float = 1.0) -> dict:
        """
        This method will check if the global optimal solution(s)
        are found in the x_pos.

        :param x_pos: numpy array with particle positions.

        :param modes: list of coordinates indicating the global modes.

        :param radius: radius of the distance.

        :return: a dictionary with the counts of the particles per mode.
        """
        # Create e dictionary to count the success.
        cppm = defaultdict(int)

        # Check sequentially all x_pos arrays.
        for px in x_pos:

            for vals in modes:
                # Make sure the modes are a numpy array too.
                vx = np.asarray(vals)

                # Check the distance from the mode, given a radius value.
                if np.sum((px - vx)**2, axis=0) <= radius ** 2:

                    # Increase counter by one.
                    cppm[tuple(vals)] += 1

                    # Exit to avoid double counting.
                    break
        # _end_for_

        # Return the dictionary.
        return cppm
    # _end_def_

# _end_class_
