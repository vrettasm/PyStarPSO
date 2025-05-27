import time
from math import isnan
from os import cpu_count
from functools import wraps
from operator import attrgetter
from collections import defaultdict, deque

from typing import Callable
from joblib import (Parallel, delayed)

import numpy as np
from numpy.random import default_rng, Generator

from ppso.auxiliary.particle import Particle

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

    def __init__(self, initial_swarm: list[Particle], obj_func: Callable,
                 lower_bound: np.typing.ArrayLike, upper_bound: np.typing.ArrayLike,
                 v_max: np.typing.ArrayLike, v_min: np.typing.ArrayLike,
                 n_cpus: int = None):
        """
        Default constructor of GenericPSO object.

        :param initial_swarm: list of the initial population of (randomized) particles.

        :param obj_func: callable objective function.

        :param v_max: maximum velocity vector.

        :param v_min: minimum velocity vector.

        :param n_cpus: number of requested CPUs for the optimization process.
        """

        # Copy the reference of the population.
        self._swarm = initial_swarm.copy()

        # Make sure the fitness function is indeed callable.
        if not callable(obj_func):
            raise TypeError(f"{self.__class__.__name__}: Objective function is not callable.")
        else:
            # Get the objective function.
            self.objective_func = obj_func
        # _end_if_

        # Set the upper/lower bounds of the search space.
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

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
        particle_size = len(initial_swarm[0].position)

        # Check the lengths of the velocity vectors.
        if any(len(vec) != particle_size for vec in [v_max, v_min]):
            raise RuntimeError(f"{self.__class__.__name__}: "
                               f"Velocity vectors should have the same length.")
        # _end_if_

        # Ensure the velocity vectors are consistent.
        if any(np.array(v_min) > np.array(v_max)):
            raise ValueError(f"{self.__class__.__name__}: "
                             f"Velocity bounds should be v_min < v_max.")
        # _end_if_

        # Copy the (references) of the velocity vectors.
        self._velocity_max = v_max
        self._velocity_min = v_min

        # Dictionary with statistics.
        self._stats = defaultdict(deque)
    # _end_def_

    def generate_random_positions(self,
                                  x_min: np.typing.ArrayLike,
                                  x_max: np.typing.ArrayLike,
                                  check_bounds: bool = False) -> None:
        """
        Generate a uniformly random population of particle positions
        within the [x_min, x_max] bounds.

        :param x_min: ArrayLike lower bounds.

        :param x_max: ArrayLike higher bounds.

        :param check_bounds: (bool) If true it will check the bounds
        before random generation.

        :return: None.
        """

        if check_bounds and any(np.array(x_min) > np.array(x_max)):
            raise ValueError(f"{self.__class__.__name__}: "
                             f"Bounds should be x_min < x_max.")
        # _end_if_

        # Get the size of the particle.
        p_size = self.swarm[0].size

        # Generate p ~ U(x_min, x_max).
        for p in self.swarm:
            p.position = GenericPSO.rng_PSO.uniform(x_min, x_max, size=p_size)
        # _end_for_

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
    def velocity_max(self) -> np.typing.ArrayLike:
        """
        Accessor method that returns the max velocity array.

        :return: ArrayLike of maximum velocity.
        """
        return self._velocity_max
    # _end_def_

    @property
    def velocity_min(self) -> np.typing.ArrayLike:
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
    def swarm(self) -> list[Particle]:
        """
        Accessor of the population list of the swarm.

        :return: the list (of particles) of the swarm.
        """
        return self._swarm
    # _end_def_

    def best_particle(self) -> Particle:
        """
        Auxiliary method that returns the particle with the
        highest function value. Safeguard with ignoring NaNs.

        :return: Return the particle with the highest value.
        """
        return max([p for p in self.swarm if not isnan(p.value)],
                   key=attrgetter("value"), default=None)
    # _end_def_

    def best_n(self, n: int = 1) -> list[Particle]:
        """
        Auxiliary method that returns the best 'n' particles
        with the highest objective function value.

        :param n: the number of the best chromosomes.

        :return: Return the 'n' chromosomes with the highest fitness.
        """

        # Make sure 'n' is positive integer.
        if not isinstance(n, int) or n <= 0:
            raise ValueError(f"{self.__class__.__name__}: "
                             f"Input must be a positive integer.")
        # _end_if_

        # Ensure the number of return particles do not exceed
        # the size of the swarm.
        if n > len(self.swarm):
            raise RuntimeError(f"{self.__class__.__name__}: "
                               f"Best {n} exceeds swarm size.")
        # _end_if_

        # Sort the swarm in descending order.
        sorted_swarm = sorted([p for p in self.swarm if not isnan(p.value)],
                              key=attrgetter("value"), reverse=True)

        # Return the best 'n' particles.
        return sorted_swarm[0:n]
    # _end_def_

    def swarm_function_values(self) -> list[float]:
        """
        Get the function values of all the swarm.

        :return: A list with all the objectives values.
        """
        return [p.value for p in self.swarm]
    # _end_def_

    def individual_value(self, index: int) -> float:
        """
        Get the fitness value of an individual member of the population.

        :param index: Position of the individual in the population.

        :return: The fitness value (float).
        """
        return self.swarm[index].value
    # _end_def_

    def swarm_positions(self) -> list:
        """
        Get the particle positions of all the swarm.

        :return: A list with all the positions.
        """
        return [p.position for p in self.swarm]
    # _end_def_

    def evaluate_function(self, input_swarm: list[Particle], parallel_mode: bool = False,
                          backend: str = "threading") -> (list[float], bool):
        """
        Evaluate all the particles of the input list with the custom objective function.
        The parallel_mode is optional.

        :param input_swarm: (list) The population of particle that we want to evaluate
        their function values.

        :param parallel_mode: (bool) Enables parallel computation of the objective function.

        :param backend: Backend for the parallel Joblib framework.

        :return: a list with the function values and the found solution flag.
        """

        # Get a local copy of the objective function.
        func = self.objective_func

        # Generator expression that yields the positions.
        particle_positions = (p.position for p in self.swarm)

        # Check the 'parallel_mode' flag.
        if parallel_mode:

            # Evaluate the particles in parallel mode.
            iteration_i = Parallel(n_jobs=self._n_cpus, backend=backend)(
                delayed(func)(p) for p in particle_positions
            )
        else:

            # Evaluate all the particles in serial mode.
            iteration_i = [func(p) for p in particle_positions]
        # _end_if_

        # Preallocate the function values list.
        function_values = len(iteration_i) * [None]

        # Flag to indicate if a solution has been found.
        found_solution = False

        # Update all particles with their objective function values and check
        # if a solution has been found.
        for n, (p, output) in enumerate(zip(input_swarm, iteration_i)):
            # Attach the fitness to each chromosome.
            p.value = output[0]

            # Collect the function values in a separate list.
            function_values[n] = output[0]

            # Update the "found solution".
            found_solution |= output[1]
        # _end_for_

        # Return the function_values values.
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
