from math import isnan
from copy import deepcopy
from operator import attrgetter

import numpy as np
from numpy.typing import ArrayLike

from ppso.auxiliary.particle import Particle

# Public interface.
__all__ = ["Swarm"]


class Swarm(object):
    """
    Description:
        TBD
    """

    def __init__(self, initial_population: list[Particle]):

        # Get the swarm population.
        self._population = deepcopy(initial_population)
    # _end_def_

    @property
    def global_best_index(self) -> int:
        """
        Accessor of the global best index.

        :return: the index (int) of the best.
        """
        return self._population.index(self.best_particle())
    # _end_def_

    @property
    def population(self) -> list[Particle]:
        """
        Accessor of the population list of the swarm.

        :return: the list (of particles) of the swarm.
        """
        return self._population
    # _end_def_

    def best_particle(self) -> Particle:
        """
        Auxiliary method that returns the particle with the
        highest function value. Safeguard with ignoring NaNs.

        :return: Return the particle with the highest value.
        """
        return max([p for p in self._population if not isnan(p.value)],
                   key=attrgetter("value"), default=None)
    # _end_def_

    def best_n(self, n: int = 1) -> list[Particle]:
        """
        Auxiliary method that returns the best 'n' particles
        with the highest objective function value.

        :param n: the number of the best chromosomes.

        :return: Return a list with the 'n' top particles.
        """

        # Make sure 'n' is positive integer.
        if not isinstance(n, int) or n <= 0:
            raise ValueError(f"{self.__class__.__name__}: "
                             f"Input must be a positive integer.")
        # _end_if_

        # Ensure the number of return particles do not exceed
        # the size of the swarm.
        if n > len(self._population):
            raise RuntimeError(f"{self.__class__.__name__}: "
                               f"Best {n} exceeds swarm size.")
        # _end_if_

        # Sort the swarm in descending order.
        sorted_swarm = sorted([p for p in self._population if not isnan(p.value)],
                              key=attrgetter("value"), reverse=True)

        # Return the best 'n' particles.
        return sorted_swarm[0:n]
    # _end_def_

    def function_values(self) -> ArrayLike:
        """
        Get the objectives function values of all the swarm.

        :return: A numpy array (vector) with all the values.
        """
        return np.asarray([p.value for p in self._population])
    # _end_def_

    def value_at(self, index: int) -> float:
        """
        Get the function value of an individual particle of the swarm.

        :param index: Position of the individual in the population.

        :return: The function value (float).
        """
        return self._population[index].value
    # _end_def_

    def position_at(self, index: int) -> ArrayLike:
        """
        Get the position vector of an individual particle of the swarm.

        :param index: Position of the individual in the population.

        :return: The position vector (array).
        """
        return self._population[index].position
    # _end_def_

    def velocity_at(self, index: int) -> ArrayLike:
        """
        Get the velocity vector of an individual particle of the swarm.

        :param index: Position of the individual in the population.

        :return: The velocity vector (array).
        """
        return self._population[index].velocity
    # _end_def_

    def positions(self) -> ArrayLike:
        """
        Get the particle positions of all the swarm.

        :return: A numpy array with all the positions.
        """
        return np.asarray([p.position for p in self._population])
    # _end_def_

    def velocities(self) -> ArrayLike:
        """
        Get the particle velocities of all the swarm.

        :return: A numpy array with all the velocities.
        """
        return np.asarray([p.velocity for p in self._population])
    # _end_def_

# _end_class_
