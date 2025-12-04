import numpy as np
from numba import njit

from numpy import zeros_like
from numpy import exp as np_exp

from star_pso.utils import VOptions
from star_pso.engines.generic_pso import GenericPSO
from star_pso.utils.auxiliary import nb_median_hamming_distance

# Public interface.
__all__ = ["BinaryPSO"]

# Define a local auxiliary function.
@njit
def clip_inplace(x, x_min, x_max) -> None:
    """
    Local auxiliary function that is used to clip the values of
    input array 'x' to [x_min, x_max] range, and put the output
    inplace.

    :param x: the numpy array we want to clip its values.

    :param x_min: the minimum (lower bound).

    :param x_max: the maximum (upper bound).
    """
    np.clip(x, x_min, x_max, out=x)
# _end_def_


class BinaryPSO(GenericPSO):
    """
    Description:

    This class implements the discrete binary particle swarm optimization variant
    as described in:

    Kennedy, J., and R. C. Eberhart 1997. “A Discrete Binary Version of the Particle
    Swarm Algorithm.” IEEE International conference on systems, man, and cybernetics,
    1997. Computational cybernetics and simulation, Vol. 5, Orlando, FL, October 12–15,
    pp: 4104–4108.
    """

    def __init__(self, v_min: ArrayLike, v_max: ArrayLike, **kwargs) -> None:
        """
        Default initializer of the BinaryPSO class.

        :param v_min: lower velocity bound.

        :param v_max: upper velocity bound.
        """

        # Call the super initializer with the input parameters.
        super().__init__(lower_bound=v_min, upper_bound=v_max, **kwargs)

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
        # Call the method of the parent class.
        super().update_velocities(params)

        # Clip velocities in [v_min, v_max].
        clip_inplace(self._velocities,
                     self.lower_bound,
                     self.upper_bound)
    # _end_def_

    def update_positions(self) -> None:
        """
        Updates the positions of the particles in the swarm.

        :return: None.
        """
        # Generate random vectors in U(0, 1).
        r_uniform = GenericPSO.rng.uniform(0, 1,
                                           size=(self.n_rows, self.n_cols))
        # Create a matrix with zeros.
        new_positions = zeros_like(r_uniform, dtype=int)

        # Compute the logistic values.
        s_arr = 1.0 / (1.0 + np_exp(-self._velocities))

        # Where the logistic function values are
        # higher than the random value set to 1.
        new_positions[s_arr > r_uniform] = 1

        # Update all particle positions.
        for particle, x_new, in zip(self._swarm.population,
                                    new_positions):
            particle.position = x_new
    # _end_def_

    def generate_random_positions(self) -> None:
        """
        Generate the population of particles positions by
        sampling discrete binary random numbers within the
        {0, 1} set.

        :return: None.
        """
        # Generate random BINARY positions Bin(0, 1).
        binary_positions = GenericPSO.rng.integers(0, 1, endpoint=True,
                                                   size=(self.n_rows, self.n_cols))
        # Assign the new positions in the swarm.
        for p, x_new in zip(self._swarm, binary_positions):
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
        # Generate random binary positions.
        self.generate_random_positions()

        # Clear all the internal bookkeeping.
        self.clear_all()
    # _end_def_

    def calculate_spread(self) -> float:
        """
        Calculates a spread measure for the particle positions
        using the normalized median Hamming distance.

        A value close to '0' indicates the swarm is converging
        to a single value. On the contrary a value close to '1'
        indicates the swarm is still spread around the search
        space.

        :return: an estimated measure (float) for the spread of
        the particles.
        """
        # Extract the positions in a 2D numpy array.
        positions = self.swarm.positions_as_array()

        # Normalized median Hamming distance.
        return nb_median_hamming_distance(positions, normal=True)
    # _end_def_

# _end_class_
