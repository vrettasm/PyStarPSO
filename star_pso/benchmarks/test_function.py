import numpy as np
from scipy.stats import qmc
from numpy.random import default_rng, Generator
from star_pso.population.particle import Particle

# Public interface.
__all__ = ["TestFunction"]


class TestFunction(object):
    """
    Description:
        All benchmark test functions should inherit from this class.
        Here is provided the interface for all the test problems.
    """

    # Make a random number generator.
    rng: Generator = default_rng()

    # Object variables.
    __slots__ = ("_name", "_nDim", "_x_min", "_x_max", "_lhc")

    def __init__(self, name: str, n_dim: int,
                 x_min: float | np.ndarray,
                 x_max: float | np.ndarray) -> None:
        """
        Description:
        Default initializer of the "TestFunction" class.

        :param name: (str) the name of the function.

        :param n_dim: (int) the number of dimension of the input space.

        :param x_min: (float) the lower bound values of the search space.

        :param x_max: (float) the upper bound values of the search space.
        """
        # Sanity check.
        if np.any(x_min >= x_max):
            raise ValueError(f"{self.__class__.__name__}: "
                             f"x_min must be smaller than x_max.")
        # _end_if_

        # Assign the function name.
        self._name = name

        # Assign the function name.
        self._nDim = int(n_dim)

        # Assign the minimum value(s).
        self._x_min = x_min

        # Assign the maximum value(s).
        self._x_max = x_max

        # Construct a Latin Hyper Cube sampler.
        self._lhc = qmc.LatinHypercube(d=self._nDim, rng=TestFunction.rng,
                                       optimization="random-cd")
    # _end_def_

    @staticmethod
    def qmc_scale(*args, **kwargs) -> np.ndarray:
        """
        Simply provides access to the QMC scale method.

        :return: the output of the scale method.
        """
        return qmc.scale(*args, **kwargs)
    # _end_def_

    @property
    def n_dim(self) -> int:
        """
        Accessor (getter) of the test function dimensions.

        :return: (int) number of dimensions.
        """
        return self._nDim
    # _end_def_

    @property
    def lhc_sampler(self) -> qmc.QMCEngine:
        """
        Accessor (getter) of the Latin Hyper-cube.

        :return: string name of the test function.
        """
        return self._lhc
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

    def sample_random_positions(self, n_pos: int, method: str) -> np.ndarray:
        """
        This method will create an initial set of random positions that will
        be passed to the PSO algorithm to form the swarm.

        :param n_pos: (int) number of random positions to create.

        :param method: (str) method to use for sampling ("random", "latin-hc").

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
