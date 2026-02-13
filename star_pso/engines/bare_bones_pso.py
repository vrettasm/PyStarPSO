from numpy import newaxis
from numpy.typing import ArrayLike
from numpy.linalg import norm as np_norm

from star_pso.utils import VOptions
from star_pso.engines.generic_pso import GenericPSO
from star_pso.utils.auxiliary import (nb_clip_inplace,
                                      nb_median_euclidean_distance)
# Public interface.
__all__ = ["BareBonesPSO"]


class BareBonesPSO(GenericPSO):
    """
    Description:

    This class implements a variant of the bare-bones particle swarm optimization
    as described in:

    - J. Kennedy, "Bare-bones particle swarms", Proceedings of the 2003 IEEE Swarm
      Intelligence Symposium. SIS'03 (Cat. No.03EX706), Indianapolis, IN, USA, 2003,
      pp. 80-87, doi: 10.1109/SIS.2003.1202251.
    """

    def __init__(self, x_min: ArrayLike, x_max: ArrayLike, **kwargs) -> None:
        """
        Default initializer of the BareBonesPSO class.

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
        By definition Bare-Bones doesn't have "velocities",
        but to keep the API  consistent we  use this method
        to sample the Gaussian updates, and subsequently we
        simply assign the values to the positions.

        :param params: VOptions tuple with the PSO options.

        :return: None.
        """
        # Get the (Global / Local / FIPSO) best positions.
        g_best = self.get_local_best_positions(params.mode.lower())

        # Extract the best (historical) positions.
        p_best = self.swarm.best_positions_as_array()

        # Compute the means: 'm_array'.
        # This produces an: (n_rows, n_cols) array.
        m_array = 0.5 * (p_best + g_best)

        # Compute the standard deviations: 's_array'.
        # This produces an: (n_rows,) array.
        s_array = np_norm(p_best - g_best, axis=1)

        # Generate Gaussian values with mean and standard deviation.
        v_samples = self.rng.normal(m_array, s_array[:, newaxis],
                                    size=(self.n_rows, self.n_cols))

        # Ensure we stay within limits.
        nb_clip_inplace(v_samples,
                        self.lower_bound,
                        self.upper_bound)

        # Assign the new "velocities" vectors.
        self._velocities = v_samples
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
