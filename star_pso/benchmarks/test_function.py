import numpy as np
from math import fabs
from numpy.linalg import norm
from numpy.random import default_rng, Generator
from star_pso.population.particle import Particle

# Public interface.
__all__ = ["TestFunction", "search_global_optima"]

def search_global_optima(swarm_population: list[Particle], epsilon: float = 1.0e-5,
                         radius: float = 1.0e-1, f_opt: float | None = None) -> list:
    """
    This auxiliary method will search if the global optimal solution(s)
    are found in the swarm population.

    :param swarm_population: a list[Particle] of potential solutions.

    :param epsilon: accuracy level of the global optimal solution.

    :param radius: niche radius of the distance.

    :param f_opt: function value for the global optimal solution.

    :return: a list of best-fit individuals identified as solutions.
    """
    # Define a return list that will contain the
    # particles that are on the global solutions.
    optima_list = []

    # Check all the particles.
    for px in swarm_population:

        # Reset the exist flag.
        already_exists = False

        # Check if the fitness is near the global
        # optimal value (within error - epsilon).
        if fabs(f_opt - px.value) <= epsilon:

            # Check if the particle is already
            # in the optimal particles list.
            for k in optima_list:

                # Check if the two particles are close to each other.
                if norm(k.position - px.position) <= radius:
                    already_exists = True
                    break
            # Add the particle only if it doesn't
            # already exist in the list.
            if not already_exists:
                optima_list.append(px)
    # _end_for_

    # Return the list.
    return optima_list
# _end_def_


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

    def sample_random_positions(self, n_pos: int) -> np.ndarray:
        """
        This method will create an initial set of random positions
        that will be passed to the PSO algorithm to form the swarm.

        :param n_pos: Number of random positions to create.

        :return: None.
        """
        raise NotImplementedError(f"{self.__class__.__name__}: "
                                  f"You should implement this method!")
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
        raise NotImplementedError(f"{self.__class__.__name__}: "
                                  f"You should implement this method!")
    # _end_def_

# _end_class_
