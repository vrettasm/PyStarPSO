from numpy.typing import (NDArray, ArrayLike)

from star_pso.engines.generic_pso import GenericPSO
from star_pso.utils.auxiliary import (nb_clip_inplace,
                                      nb_median_euclidean_distance)
# Public interface.
__all__ = ["StandardPSO"]


class StandardPSO(GenericPSO):
    """
    Description:

    This implements a basic variant of the original PSO algorithm as
    described in:

    - Shi, Y. and Eberhart, R. (1998). "A modified particle swarm optimizer".
      In Proceedings of the IEEE World Congress on Computational Intelligence,
      Anchorage, AK, USA, 4–9 May 1998; pp. 69–73.
    """

    def __init__(self, x_min: ArrayLike, x_max: ArrayLike, **kwargs) -> None:
        """
        Default initializer of the StandardPSO class.

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
        # Initialize the new positions.
        new_positions: NDArray = self.swarm.positions_as_array()

        # Add the new velocities to the positions.
        new_positions += self._velocities

        # Ensure the particle stays within bounds.
        nb_clip_inplace(new_positions, self.lower_bound, self.upper_bound)

        # Update all particle positions.
        self.swarm.set_positions(new_positions)
    # _end_def_

    def generate_random_positions(self) -> None:
        """
        Generate the population of particles positions by sampling
        uniformly random numbers within the [x_min, x_max] bounds.

        :return: None.
        """
        # Generate uniform FLOAT positions U(x_min, x_max).
        uniform_positions: NDArray = GenericPSO.rng.uniform(
            low=self.lower_bound, high=self.upper_bound,
            size=(self.n_rows, self.n_cols)
        )
        # Assign the new positions in the swarm.
        self.swarm.set_positions(uniform_positions)
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
            size=(self.n_rows, self.n_cols),
        )
    # _end_def_

    def reset_all(self) -> None:
        """
        Resets the particle positions, velocities and the statistics dictionary.

        :return: None.
        """
        # Reset particle velocities.
        self.generate_random_velocities()

        # Generate random uniform positions.
        self.generate_random_positions()

        # Clear all the internal bookkeeping.
        self.clear_all()
    # _end_def_

    def calculate_spread(self) -> float:
        """
        Calculates a spread measure for the particle positions
        using the normalized median Euclidean distance.

        A value close to '0' indicates the swarm is converging
        to a single value. On the contrary a value close to '1'
        indicates the swarm is still  spread around the search
        space.

        :return: an estimated measure (float) for the spread of
                 the particles.
        """
        # Extract the positions in a 2D numpy array.
        positions = self.swarm.positions_as_array()

        # Normalized median Euclidean distance.
        return nb_median_euclidean_distance(positions, normal=True)
    # _end_def_

# _end_class_
