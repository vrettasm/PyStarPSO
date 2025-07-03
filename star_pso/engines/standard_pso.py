from numpy import mean as np_mean
from numpy import clip as np_clip
from numpy.typing import ArrayLike

from star_pso.auxiliary.utilities import VOptions
from star_pso.engines.generic_pso import GenericPSO
from star_pso.auxiliary.utilities import nb_average_euclidean_distance

# Public interface.
__all__ = ["StandardPSO"]


class StandardPSO(GenericPSO):
    """
    Description:

    This implements a basic variant of the original PSO algorithm as described in:

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

    def update_velocities(self, params: VOptions) -> None:
        """
        Performs the update on the velocity equations.

        :param params: VOptions tuple with the PSO options.

        :return: None.
        """
        # Get the shape of the velocity array.
        arr_shape = (self.n_rows, self.n_cols)

        # Pre-sample the cognitive coefficients.
        cogntv = GenericPSO.rng.uniform(0, params.c1, size=arr_shape)

        # Pre-sample the social coefficients.
        social = GenericPSO.rng.uniform(0, params.c2, size=arr_shape)

        # Get the GLOBAL best particle position.
        if params.global_avg:
            # In the fully informed case we take the average of all the best positions.
            g_best = np_mean([p.best_position for p in self.swarm.population], axis=0)
        else:
            g_best = self.swarm.best_particle().position
        # _end_if_

        # Inertia weight parameter.
        w = params.w

        for i, (c1, c2) in enumerate(zip(cogntv, social)):
            # Get the current position of i-th the particle.
            x_i = self.swarm[i].position

            # Update the new velocity.
            self._velocities[i] = w * self._velocities[i] +\
                c1 * (self.swarm[i].best_position - x_i) +\
                c2 * (g_best - x_i)
        # _end_for_
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
        Calculates a spread measure for the particle positions using
        the (normalized) average Euclidean distance from the swarm
        centroid position.

        A value close to '0' indicates the swarm is converging to a
        single value. On the contrary, a value close to '1' indicates
        the swarm is still wide spread around the search space.

        :return: an estimated measure (float) for the spread
        of the particles.
        """
        # Extract the positions in a 2D numpy array.
        positions = self.swarm.positions_as_array()

        # Normalized average Euclidean distance.
        return nb_average_euclidean_distance(positions, normal=True)
    # _end_def_

# _end_class_
