from numpy import log, tile, where
from numpy.typing import ArrayLike

from star_pso.utils import VOptions
from star_pso.engines.generic_pso import GenericPSO
from star_pso.utils.auxiliary import (nb_clip_inplace,
                                      nb_median_euclidean_distance)

# Public interface.
__all__ = ["QuantumPSO"]


class QuantumPSO(GenericPSO):
    """
    Description:

    This class implements a variant of the quantum particle swarm optimization
    as described in:

    - M. Xi, J. Sun, W. Xu (2008), "An improved quantum-behaved particle swarm optimization
      algorithm with weighted mean best position", Applied Mathematics and Computation vol.
      205 pp: 751–759, doi: 10.1016/j.amc.2008.05.135.
    """

    def __init__(self, x_min: ArrayLike, x_max: ArrayLike, **kwargs) -> None:
        """
        Default initializer of the QuantumPSO class.

        :param x_min: lower search space bound.

        :param x_max: upper search space bound.
        """
        # Call the super initializer with the input parameters.
        super().__init__(lower_bound=x_min, upper_bound=x_max, **kwargs)

        # DISABLE the adaptation of model parameters
        # since this class does not support the same
        # velocity equations.
        self.disable_parameters_update()

        # Generate initial particle "velocities".
        self._velocities = GenericPSO.rng.uniform(self.lower_bound,
                                                  self.upper_bound,
                                                  size=(self.n_rows,
                                                        self.n_cols))
    # _end_def_

    def update_velocities(self, params: VOptions) -> None:
        """
        Performs the update on the "velocity" equations.

        :param params: VOptions tuple with the PSO options.

        :return: None.
        """
        # Hardcode the contraction / expansion coefficient.
        beta_coefficient = 0.5

        # Get the shape of the velocity array.
        arr_shape = (self.n_rows, self.n_cols)

        # Pre-sample the 'phi' parameters.
        param_phi = GenericPSO.rng.random(size=arr_shape)

        # Pre-sample the 'u' parameters.
        param_u = GenericPSO.rng.random(size=arr_shape)

        # Get the (Global / Local / FIPSO) best positions.
        m_best = self.get_local_best_positions(params.mode.lower())

        # Extract the current positions.
        x_current = self.swarm.positions_as_array()

        # Extract the best (historical) positions.
        p_best = self.swarm.best_positions_as_array()

        # Extract the global best position.
        g_best = self.swarm.best_particle().position

        # Construct the 'p_best' (in-place).
        p_best *= param_phi

        # Element-wise multiplication using broadcasting.
        p_best += (1.0 - param_phi) * g_best

        # Ensure there are no zero values that would raise
        # an error below when computing log(param_u). This
        # is very unlikely but it could happen.
        param_u[param_u == 0.0] = QuantumPSO.NUMPY_EPS

        # Compute the offset.
        p_offset = beta_coefficient * (m_best - x_current) * log(param_u)

        # Select the directions at random.
        direction = where(self.rng.random(arr_shape) < 0.5, -1.0, 1.0)

        # Perform all operations in place.
        p_best += direction * p_offset

        # Ensure we stay within limits.
        nb_clip_inplace(p_best,
                        self.lower_bound,
                        self.upper_bound)

        # Assign the new "velocities" vectors.
        self._velocities = p_best
    # _end_def_

    def update_positions(self) -> None:
        """
        Updates the positions of the particles in the swarm.

        :return: None.
        """
        # Update all particle positions.
        self.swarm.set_positions(self._velocities)
    # _end_def_

    def generate_random_positions(self) -> None:
        """
        Generate the population of particles positions by
        sampling uniform random numbers within the limits.

        :return: None.
        """
        # Generate uniform FLOAT positions U(x_min, x_max).
        self._velocities = GenericPSO.rng.uniform(self.lower_bound,
                                                  self.upper_bound,
                                                  size=(self.n_rows,
                                                        self.n_cols))
        # Assign the new positions.
        self.swarm.set_positions(self._velocities)
    # _end_def_

    def reset_all(self) -> None:
        """
        Resets the particle positions, velocities
        and clear all the statistics dictionary.

        :return: None.
        """
        # Generate random the positions.
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
