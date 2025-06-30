from numpy import zeros_like
from numpy import exp as np_exp
from numpy import mean as np_mean
from numpy import clip as np_clip
from numpy.typing import ArrayLike

from star_pso.auxiliary.utilities import VOptions
from star_pso.engines.generic_pso import GenericPSO

# Public interface.
__all__ = ["BinaryPSO"]


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

    def __init__(self, v_min: ArrayLike, v_max: ArrayLike, **kwargs):
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

        # We clip the velocities in [V_min, V_max].
        np_clip(self._velocities, self._lower_bound, self._upper_bound,
                out=self._velocities)
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

# _end_class_
