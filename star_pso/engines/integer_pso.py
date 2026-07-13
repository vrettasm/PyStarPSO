"""
Description:

    This implements an Integer variant of the original PSO algorithm that
    operates similarly to the StandardPSO, but rounds the positions to the
    nearest integer.

Author:
    Michail D. Vrettas, PhD

Email:
    michail.vrettas@gmail.com

Metadata:
    License: GPL-3
"""

from numpy import rint
from numpy.typing import (NDArray, ArrayLike)

from star_pso.engines.generic_pso import GenericPSO
from star_pso.utils.auxiliary import (nb_clip_inplace,
                                      nb_median_taxicab_distance)

# Public interface.
__all__ = ["IntegerPSO"]


class IntegerPSO(GenericPSO):
    """
    Description:

    This implements an Integer variant of the original PSO algorithm that operates
    similarly to the StandardPSO, but rounds the positions to the nearest integer.
    """

    def __init__(self, x_min: ArrayLike, x_max: ArrayLike, **kwargs) -> None:
        """
        Default initializer of the IntegerPSO class.

        :param x_min: lower search space bound.

        :param x_max: upper search space bound.
        """

        # Call the super initializer with the input parameters.
        super().__init__(lower_bound=x_min, upper_bound=x_max, **kwargs)

        # Generate initial particle velocities.
        self.generate_random_velocities()
    # _end_def_

    def update_positions(self) -> None:
        """
        Updates the positions of the particles in the swarm.

        :return: None.
        """
        # Round the new positions and convert them to type int.
        new_positions: NDArray = rint(self.swarm.positions_as_array() +
                                      self._velocities).astype(int)

        # Ensure the particle stays within bounds.
        nb_clip_inplace(new_positions, self.lower_bound, self.upper_bound)

        # Update all particle positions.
        self.swarm.set_positions(new_positions)
    # _end_def_

    def generate_random_velocities(self) -> None:
        """
        Generate the population of velocities by sampling uniformly
        random numbers within predefined bounds that depend on the
        search space range. In this case we fix that to +/-10 % of
        the total range.

        :return: None.
        """
        # Calculate the search space range per dimension.
        space_range: NDArray = self.upper_bound - self.lower_bound

        # Generate initial particle velocities scaled by the search
        # space range. E.g.: initial velocity is bounded by +/- 10%
        # of the total range.
        self._velocities: NDArray = GenericPSO.rng.uniform(
            low=-0.1 * space_range, high=0.1 * space_range,
            size=(self.n_rows, self.n_cols)
        )
    # _end_def_

    def generate_random_positions(self) -> None:
        """
        Generate the population of particles positions by sampling
        uniformly random integer numbers within the [x_min, x_max]
        bounds.

        :return: None.
        """
        # Generate uniform INTEGER positions Int(x_min, x_max).
        integer_positions = GenericPSO.rng.integers(
            low=self.lower_bound, high=self.upper_bound,
            endpoint=True, size=(self.n_rows, self.n_cols),
            dtype=int
        )

        # Assign the new positions in the swarm.
        self.swarm.set_positions(integer_positions)
    # _end_def_

    def reset_all(self) -> None:
        """
        Resets the particle positions, velocities
        and clear all the statistics dictionary.

        :return: None.
        """
        # Reset particle velocities.
        self.generate_random_velocities()

        # Generate random integer positions.
        self.generate_random_positions()

        # Clear all the internal bookkeeping.
        self.clear_all()
    # _end_def_

    def calculate_spread(self) -> float:
        """
        Calculates a spread measure for the particle positions
        using the normalized median Taxi-Cab distance.

        A value close to '0' indicates the swarm is converging
        to a single value. On the contrary a value close to '1'
        indicates the swarm is still spread around the search space.

        :return: an estimated measure (float) for the spread of
                 the particles.
        """
        # Extract the positions in a 2D numpy array.
        positions = self.swarm.positions_as_array()

        # Normalized median Taxi-Cab/Manhattan distance.
        return nb_median_taxicab_distance(positions, normal=True)
    # _end_def_

# _end_class_
