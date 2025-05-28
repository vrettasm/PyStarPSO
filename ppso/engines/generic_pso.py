import time
from os import cpu_count
from functools import wraps
from collections import defaultdict, deque

from typing import Callable
from joblib import (Parallel, delayed)

import numpy as np
from numpy.typing import ArrayLike
from numpy.random import default_rng, Generator

from ppso.auxiliary.swarm import Swarm

# Public interface.
__all__ = ["GenericPSO", "time_it"]

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


class GenericPSO(object):
    """
    Description:

        GenericPSO class models the interface of a specific particle swarm optimization
        model (or engine). It provides the common variables and functionalities that all
        PSO models should share.
    """

    # Make a random number generator.
    rng_PSO: Generator = default_rng()

    # Set the maximum number of CPUs (at least one).
    MAX_CPUs: int = 1 if not cpu_count() else cpu_count()

    # Object variables.
    __slots__ = ("_swarm", "objective_func", "_velocity_max", "_velocity_min",
                 "_upper_bound", "_lower_bound", "_stats", "_n_cpus")

    def __init__(self,
                 initial_swarm: Swarm,
                 obj_func: Callable,
                 lower_bound: ArrayLike,
                 upper_bound: ArrayLike,
                 v_min: ArrayLike,
                 v_max: ArrayLike,
                 n_cpus: int = None):
        """
        Default constructor of GenericPSO object.

        :param initial_swarm: list of the initial population of (randomized) particles.

        :param obj_func: callable objective function.

        :param lower_bound: lower search space bound.

        :param upper_bound: upper search space bound.

        :param v_min: minimum velocity vector.

        :param v_max: maximum velocity vector.

        :param n_cpus: number of requested CPUs for the optimization process.
        """

        # Get the swarm population.
        self._swarm = initial_swarm

        # Make sure the fitness function is indeed callable.
        if not callable(obj_func):
            raise TypeError(f"{self.__class__.__name__}: Objective function is not callable.")
        else:
            # Get the objective function.
            self.objective_func = obj_func
        # _end_if_

        # Set the upper/lower bounds of the search space.
        # Ensure the bounds vectors are arrays.
        self._lower_bound = np.array(lower_bound)
        self._upper_bound = np.array(upper_bound)

        # Get the number of requested CPUs.
        if n_cpus is None:

            # This is the default option.
            self._n_cpus = GenericPSO.MAX_CPUs
        else:

            # Assign the  requested number, making sure we have
            # enough CPUs and the value entered has the correct
            # type.
            self._n_cpus = max(1, min(GenericPSO.MAX_CPUs, int(n_cpus)))
        # _end_if_

        # Get the length of the particles' position.
        particle_size = len(initial_swarm[0])

        # Check the lengths of the velocity vectors.
        if any(len(vec) != particle_size for vec in [v_max, v_min]):
            raise RuntimeError(f"{self.__class__.__name__}: "
                               f"Velocity vectors should have the same length.")
        # _end_if_

        # Ensure the velocity vectors are consistent.
        if any(np.asarray(v_min) > np.asarray(v_max)):
            raise ValueError(f"{self.__class__.__name__}: "
                             f"Velocity bounds should be v_min < v_max.")
        # _end_if_

        # Ensure the velocity vectors are arrays.
        self._velocity_max = np.array(v_max)
        self._velocity_min = np.array(v_min)

        # Dictionary with statistics.
        self._stats = defaultdict(deque)
    # _end_def_

    @classmethod
    def set_seed(cls, new_seed=None) -> None:
        """
        Sets a new seed for the random number generator.

        :param new_seed: New seed value (default=None).

        :return: None.
        """
        # Re-initialize the class variable.
        cls.rng_PSO = default_rng(seed=new_seed)
    # _end_def_

    @property
    def velocity_max(self) -> ArrayLike:
        """
        Accessor method that returns the max velocity array.

        :return: ArrayLike of maximum velocity.
        """
        return self._velocity_max
    # _end_def_

    @property
    def velocity_min(self) -> ArrayLike:
        """
        Accessor method that returns the min velocity array.

        :return: ArrayLike of minimum velocity.
        """
        return self._velocity_min
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
    def n_cpus(self) -> int:
        """
        Accessor method that returns the number of CPUs.

        :return: the n_cpus.
        """
        return self._n_cpus
    # _end_def_

    @property
    def swarm(self) -> Swarm:
        """
        Accessor of the population list of the swarm.

        :return: the list of particles (the swarm).
        """
        return self._swarm
    # _end_def_

    def generate_random_positions(self,
                                  x_min: ArrayLike = None,
                                  x_max: ArrayLike = None) -> None:
        """
        Generate the population of particles positions by sampling
        uniformly random numbers within the [x_min, x_max] bounds.

        :param x_min: the minimum allowed values for the positions.

        :param x_max: the maximum allowed values for the positions.

        :return: None.
        """

        # If 'x_min' is absent use the default lower_bound.
        x_min = self._lower_bound if x_min is None else np.asarray(x_min)

        # If 'x_max' is absent use the default upper_bound.
        x_max = self._upper_bound if x_max is None else np.asarray(x_max)

        # Get the size of the particle.
        particle_size = self._swarm[0].size

        # Generate p ~ U(x_min, x_max).
        for p in self._swarm:
            p.position = GenericPSO.rng_PSO.uniform(x_min, x_max,
                                                    size=particle_size)
    # _end_def_

    def evaluate_function(self, parallel_mode: bool = False,
                          backend: str = "threading") -> (list[float], bool):
        """
        Evaluate all the particles of the input list with the custom objective function.
        The parallel_mode is optional.

        :param parallel_mode: (bool) Enables parallel computation of the objective function.

        :param backend: Backend for the parallel Joblib framework.

        :return: a list with the function values and the found solution flag.
        """

        # Get a local copy of the objective function.
        func = self.objective_func

        # Extract the positions of the swarm in numpy array.
        positions = self._swarm.positions()

        # Check the 'parallel_mode' flag.
        if parallel_mode:

            # Evaluate the particles in parallel mode.
            evaluation_i = Parallel(n_jobs=self._n_cpus, backend=backend)(
                delayed(func)(x) for x in positions
            )
        else:

            # Evaluate all the particles in serial mode.
            evaluation_i = [func(x) for x in positions]
        # _end_if_

        # Preallocate the function values list.
        function_values = len(evaluation_i) * [None]

        # Flag to indicate if a solution has been found.
        found_solution = False

        # Update all particles with their objective function values and check
        # if a solution has been found.
        for n, (p, result) in enumerate(zip(self._swarm.population,
                                            evaluation_i)):
            # Attach the fitness to each chromosome.
            p.value = result[0]

            # Collect the function values in a separate list.
            function_values[n] = result[0]

            # Update the "found solution".
            found_solution |= result[1]
        # _end_for_

        # Return the function values.
        return function_values, found_solution
    # _end_def_

    def run(self, *args, **kwargs):
        """
        Main method of the Generic PSO class,
        that implements the optimization routine.
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
