from numpy import clip as np_clip
from numpy.typing import ArrayLike

from star_pso.engines.generic_pso import GenericPSO
from star_pso.auxiliary.utilities import nb_median_euclidean_distance

# Public interface.
__all__ = ["StandardPSO"]


class StandardPSO(GenericPSO):
    """
    Description:

    This implements a basic variant of the original PSO algorithm as
    described in:

    Kennedy, J. and Eberhart, R. (1995). "Particle Swarm Optimization".
    Proceedings of IEEE International Conference on Neural Networks.
    Vol. IV. pp. 1942–1948. doi:10.1109/ICNN.1995.488968.
    """

    def __init__(self, x_min: ArrayLike, x_max: ArrayLike, **kwargs):
        """
        Default initializer of the StandardPSO class.

        :param x_min: lower search space bound.

        :param x_max: upper search space bound.
        """

        # Call the super initializer with the input parameters.
        super().__init__(lower_bound=x_min, upper_bound=x_max, **kwargs)

        # Generate initial particle velocities.
        self._velocities = GenericPSO.rng.uniform(-1.0, +1.0,
                                                  size=(self.n_rows, self.n_cols))
    # _end_def_

    def update_positions(self) -> None:
        """
        Updates the positions of the particles in the swarm.

        :return: None.
        """
        # Add the new velocities to the positions.
        new_positions = self.swarm.positions_as_array() + self._velocities

        # Ensure the particle stays within bounds.
        np_clip(new_positions, self._lower_bound, self._upper_bound,
                out=new_positions)

        # Update all particle positions.
        for particle, x_new in zip(self._swarm.population,
                                   new_positions):
            particle.position = x_new
    # _end_def_

    def generate_random_positions(self) -> None:
        """
        Generate the population of particles positions by sampling
        uniformly random numbers within the [x_min, x_max] bounds.

        :return: None.
        """
        # Generate uniform FLOAT positions U(x_min, x_max).
        uniform_positions = GenericPSO.rng.uniform(self._lower_bound,
                                                   self._upper_bound,
                                                   size=(self.n_rows, self.n_cols))
        # Assign the new positions in the swarm.
        for p, x_new in zip(self._swarm, uniform_positions):
            p.position = x_new
    # _end_def_

    def reset_all(self) -> None:
        """
        Resets the particle positions, velocities and the statistics dictionary.

        :return: None.
        """
        # Reset particle velocities.
        self._velocities = GenericPSO.rng.uniform(-1.0, +1.0,
                                                  size=(self.n_rows, self.n_cols))
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
        indicates the swarm is still spread around the search space.

        :return: an estimated measure (float) for the spread of the
        particles.
        """
        # Extract the positions in a 2D numpy array.
        positions = self.swarm.positions_as_array()

        # Normalized median Euclidean distance.
        return nb_median_euclidean_distance(positions, normal=True)
    # _end_def_

# _end_class_
