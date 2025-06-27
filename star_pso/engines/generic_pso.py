from math import inf
from os import cpu_count
from copy import deepcopy
from collections import defaultdict

from typing import Callable
from joblib import (Parallel, delayed)

from numpy.typing import ArrayLike
from numpy import array as np_array
from numpy import empty as np_empty
from numpy.random import (default_rng, Generator)

from star_pso.auxiliary.swarm import Swarm
from star_pso.auxiliary.utilities import VOptions

# Public interface.
__all__ = ["GenericPSO"]


class GenericPSO(object):
    """
    Description:

        GenericPSO class models the interface of a specific particle swarm
        optimization model or engine. It provides the common variables and
        functionalities that all PSO models should share.
    """

    # Make a random number generator.
    rng: Generator = default_rng()

    # Set the maximum number of CPUs (at least one).
    MAX_CPUs: int = 1 if not cpu_count() else cpu_count()

    # Object variables.
    __slots__ = ("_swarm", "_velocities", "objective_func",
                 "_upper_bound", "_lower_bound", "_stats",
                 "_items", "n_cpus", "n_rows", "n_cols")

    def __init__(self,
                 initial_swarm: Swarm,
                 obj_func: Callable,
                 lower_bound: ArrayLike = None,
                 upper_bound: ArrayLike = None,
                 copy: bool = False,
                 n_cpus: int = None):
        """
        Default initializer of the GenericPSO class.

        :param initial_swarm: list of the initial population of particles.

        :param obj_func: callable objective function.

        :param lower_bound: lower search space bound.

        :param upper_bound: upper search space bound.

        :param copy: if True it will create a separate (deep) copy of the initial swarm.

        :param n_cpus: number of requested CPUs for the optimization process.
        """

        # Get the swarm population.
        self._swarm = deepcopy(initial_swarm) if copy else initial_swarm

        # Number of particles.
        self.n_rows = len(self._swarm)

        # Size (length) of particle.
        self.n_cols = len(self._swarm[0])

        # Make sure the fitness function is indeed callable.
        if not callable(obj_func):
            raise TypeError(f"{self.__class__.__name__}: Objective function is not callable.")
        else:
            # Get the objective function.
            self.objective_func = obj_func
        # _end_if_

        # Set the upper/lower bounds of the search space.
        self._lower_bound = np_array(lower_bound)
        self._upper_bound = np_array(upper_bound)

        # Get the number of requested CPUs.
        if n_cpus is None:

            # This is the default option.
            self.n_cpus = max(1, GenericPSO.MAX_CPUs-1)
        else:

            # Assign the  requested number, making sure we have
            # enough CPUs and the value entered has the correct
            # type.
            self.n_cpus = max(1, min(GenericPSO.MAX_CPUs-1, int(n_cpus)))
        # _end_if_

        # Dictionary with statistics.
        self._stats = defaultdict(list)

        # Place holder.
        self._items = None
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

    @property
    def stats(self) -> dict:
        """
        Accessor method that returns the 'stats' dictionary.

        :return: the dictionary with the statistics from the run.
        """
        return self._stats
    # _end_def_

    @property
    def items(self) -> list | tuple:
        """
        Accessor (getter) of the _items placeholder container.

        :return: _items (if any).
        """
        return self._items
    # _end_def_

    @property
    def swarm(self) -> Swarm:
        """
        Accessor of the swarm.

        :return: the reference the swarm.
        """
        return self._swarm
    # _end_def_

    def evaluate_function(self, parallel_mode: bool = False,
                          categorical_mode: bool = False,
                          jack_of_all_trades: bool = False,
                          backend: str = "threads") -> (list[float], bool):
        """
        Evaluate all the particles of the input list with the custom objective
        function. The parallel_mode is optional.

        :param parallel_mode: (bool) enables parallel computation of the objective
        function. Default is False (serial execution).

        :param categorical_mode: (bool) enables generation of position samples
        from probabilities.

        :param jack_of_all_trades: (bool) enables generation of position samples
        from data blocks as used by the JackOfAllTradesPSO class.

        :param backend: backend for the parallel Joblib ('threads' or 'processes').

        :return: the max function value and the found solution flag.
        """

        # Check if "Jack of All Trades" is enabled.
        if jack_of_all_trades:
            # Extract the positions in a list of lists.
            positions = self._swarm.positions_as_list()

            # Check if the swarm has categorical data blocks.
            if self.swarm.has_categorical:
                # Sample categorical variable.
                self._items["sample_random_values"](positions)
            # _end_if_
        else:
            # Extract the positions in a 2D numpy array.
            positions = self._swarm.positions_as_array()

            # Only True in CategoricalPSO.
            if categorical_mode:
                # Sample categorical variable.
                self._items["sample_random_values"](positions)
            # _end_if_
        # _end_if_

        # Get a local copy of the objective function.
        func = self.objective_func

        # Check the 'parallel_mode' flag.
        if parallel_mode:

            # Evaluate the particles in parallel mode.
            f_evaluation = Parallel(n_jobs=self.n_cpus, prefer=backend)(
                delayed(func)(x) for x in positions
            )
        else:

            # Evaluate all the particles in serial mode.
            f_evaluation = [func(x) for x in positions]
        # _end_if_

        # Flag to indicate if a solution has been found.
        found_solution = False

        # Initialize f_max.
        f_max = -inf

        # Initialize the best position.
        x_best = None

        # Stores the function values.
        fx_array = np_empty(self.n_rows, dtype=float)

        # Update all particles with their new objective function values.
        for n, (p, result) in enumerate(zip(self._swarm, f_evaluation)):
            # Extract the n-th function value.
            f_value = result[0]

            # Attach the function value to each particle.
            p.value = f_value

            # Update the found solution.
            found_solution |= result[1]

            # Update the statistics.
            fx_array[n] = f_value

            # Update f_max value.
            if f_value > f_max:
                f_max = f_value
                x_best = positions[n]
        # _end_for_

        # Store the function values as ndarray.
        self.stats["f_values"].append(fx_array)

        # Store the best (sampled) position.
        self.stats["x_best"].append(x_best)

        # Store the optimal f-value of this
        # iteration.
        self.stats["f_best"].append(f_max)

        # Update local best for consistent results.
        self.swarm.update_local_best()

        # Return the tuple.
        return f_max, found_solution
    # _end_def_

    def get_optimal_values(self) -> tuple:
        """
        Iterates through the stats to find the best recorded
        position form all the iterations.

        :return: a tuple with the optimal particle position,
                 its function value and the iteration it was
                 found.
        """

        # Get the maximum of f_best.
        f_opt = max(self.stats["f_best"])

        # Get the index of f_best.
        i_opt = self.stats["f_best"].index(f_opt)

        # Get the corresponding x_best.
        x_opt = self.stats["x_best"][i_opt]

        # Return the optimal particle position,
        # along with its function value and its
        # iteration.
        return i_opt, f_opt, x_opt
    # _end_def_

    def generate_random_positions(self) -> None:
        """
        Generate a population of particles with random positions.
        Each different class that inherits from here should know
        how to implement it.

        :return: None.
        """
        raise NotImplementedError(f"{self.__class__.__name__}: "
                                  f"You should implement this method!")
    # _end_def_

    def update_velocities(self, params: VOptions) -> None:
        """
        Performs the update on the velocity equations.

        :param params: VOptions tuple with the PSO options.

        :return: None.
        """
        raise NotImplementedError(f"{self.__class__.__name__}: "
                                  f"You should implement this method!")
    # _end_def_

    def update_positions(self) -> None:
        """
        Updates the positions of the particles in the swarm.

        :return: None.
        """
        raise NotImplementedError(f"{self.__class__.__name__}: "
                                  f"You should implement this method!")
    # _end_def_

    def run(self, *args, **kwargs):
        """
        Main method of the Generic PSO class that implements
        the optimization routine.
        """
        raise NotImplementedError(f"{self.__class__.__name__}: "
                                  f"You should implement this method!")
    # _end_def_

    def __call__(self, *args, **kwargs):
        """
        This method is only a wrapper of the "run" method.
        """
        return self.run(*args, **kwargs)
    # _end_def_

# _end_class_
