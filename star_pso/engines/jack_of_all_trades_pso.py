from numpy import sum as np_sum
from numpy import array as np_array
from numpy import empty as np_empty
from numpy import arange as np_arange
from numpy import average as np_average
from numpy import isscalar as np_isscalar
from numpy import subtract as np_subtract

from star_pso.engines.generic_pso import GenericPSO
from star_pso.auxiliary.utilities import (VOptions, BlockType, SpecialMode,
                                          linear_rank_probabilities)
# Public interface.
__all__ = ["JackOfAllTradesPSO"]


class JackOfAllTradesPSO(GenericPSO):
    """
    Description:

        JackOfAllTradesPSO class  is an implementation  of the  PSO algorithm that
        can deal with mixed types  of optimization variables.  The supported types
        are: i) float (continuous), ii) integer (discrete), iii) binary (discrete)
        and iv) categorical (discrete).

        The fundamental building block of the algorithm is the 'DataBlock' which
        encapsulates the data and the functionality of each variable type.
    """

    def __init__(self, permutation_mode: bool = False, **kwargs):
        """
        Default initializer of the JackOfAllTradesPSO class.

        :param permutation_mode: (bool) if True it will sample
        permutations of the valid sets.
        """

        # Call the super initializer.
        super().__init__(**kwargs)

        # First we declare the velocities to be
        # an [n_rows x n_cols] array of objects.
        self._velocities = np_empty(shape=(self.n_rows, self.n_cols),
                                    dtype=object)

        # Call the random velocity generator.
        self.generate_uniform_velocities()

        # Assign the correct local sample method
        # according to the permutation mode flag.
        if permutation_mode:
            self._items = {"sample_random_values":
                           self.sample_permutation_values}
        else:
            self._items = {"sample_random_values":
                           self.sample_categorical_values}
        # _end_if_

        # Set the special mode to Jack-Of-All-Trades.
        self._special_mode = SpecialMode.JACK_OF_ALL_TRADES
    # _end_def_

    def generate_uniform_velocities(self) -> None:
        """
        Generates random uniform velocities for the data blocks.

        :return: None.
        """

        # Here we generate the random velocities.
        for i, particle in enumerate(self.swarm.population):
            for j, blk in enumerate(particle.container):
                # If the block is CATEGORICAL we
                # will use it's valid set length.
                n_vars = len(blk.valid_set) if blk.valid_set else 1

                # Generate the velocities randomly.
                self._velocities[i, j] = JackOfAllTradesPSO.rng.uniform(-1.0, +1.0,
                                                                        size=n_vars)
        # _end_for_
    # _end_def_

    def generate_random_positions(self) -> None:
        """
        Generate random positions for the population
        of particles, by calling the btype dependent
        reset methods of each data block.

        :return: None.
        """
        # Go through the whole swarm population.
        for particle in self.swarm.population:
            particle.reset_position()
    # _end_def_

    def sample_categorical_values(self, positions: list[list]) -> None:
        """
        Samples the actual position based on particles probabilities and
        valid sets for each data block.

        :param positions: the container with the lists of probabilities
        (one list for each position).

        :return: None.
        """

        # Check all particles in the swarm.
        for i, particle in enumerate(self.swarm.population):

            # Check all data blocks in the particle.
            for j, blk in enumerate(particle.container):

                # If the data block is categorical.
                if blk.btype == BlockType.CATEGORICAL:

                    # Replace the probabilities with an actual sample.
                    # WARNING: 'shuffle' option MUST be set to False!
                    positions[i][j] = JackOfAllTradesPSO.rng.choice(blk.valid_set,
                                                                    shuffle=False,
                                                                    p=positions[i][j])
            # _end_for_
    # _end_def_

    def sample_permutation_values(self, positions: list[list]) -> None:
        """
        Samples a permutation from a given set of variables.
        It is used in problems like the 'Traveling Salesman'.

        It is assumed that all data blocks are CATEGORICAL
        and that they have the same valid set of values.

        :param positions: the container with the lists of probabilities
        (one list for each position).

        :return: None.
        """

        # Create a range of values.
        random_index = np_arange(self.n_cols, dtype=int)

        # Shuffle in place. This is used to avoid introducing
        # biasing by using always the same order of blocks to
        # select first their categorical sample value.
        JackOfAllTradesPSO.rng.shuffle(random_index)

        # Check all particles in the swarm.
        for i, particle in enumerate(self.swarm.population):

            # Auxiliary set.
            exclude_idx = set()

            # Check all data blocks in the particle,
            # using the randomized index.
            for j in random_index:
                # Get the j-th data block.
                blk = particle[j]

                # Extract the probability values.
                xj = positions[i][j]

                # Sort in reverse order from high to low.
                for k in xj.argsort()[::-1]:
                    # Continue until we find the first
                    # unused element of the valid set.
                    if k not in exclude_idx:
                        # Assign the element in the right position.
                        positions[i][j] = blk.valid_set[k]

                        # Update the set() with
                        # the excluded indexes.
                        exclude_idx.add(k)

                        # Break the internal loop.
                        break
        # _end_for_
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
        cogntv = JackOfAllTradesPSO.rng.uniform(0, params.c1, size=arr_shape)

        # Pre-sample the social coefficients.
        social = JackOfAllTradesPSO.rng.uniform(0, params.c2, size=arr_shape)

        # Get the GLOBAL best particle position.
        if params.global_avg:
            # Compile a list with best positions, along with
            # their best values.
            best_positions = [(p.best_position, p.best_value)
                              for p in self.swarm.population]

            # Sort the list in ascending order
            # using their best function value.
            best_positions.sort(key=lambda item: item[1])

            # Extract only the best positions and convert to numpy array.
            g_best = np_array([item[0] for item in best_positions],
                              dtype=object)

            # Compute the weighted average according to their ranking.
            g_best = np_average(g_best, axis=0,
                                weights=linear_rank_probabilities(self.swarm.size))

            # Finally normalize them to
            # account for probabilities.
            for i in range(self.n_cols):
                # Avoid errors with scalar values.
                if not np_isscalar(g_best[i]):
                    g_best[i] /= np_sum(g_best[i], dtype=float)
            # _end_for_
        else:
            g_best = self.swarm.best_particle().position
        # _end_if_

        # Inertia weight parameter.
        w = params.w

        for i, (c1, c2) in enumerate(zip(cogntv, social)):
            # Get the (old) position of the i-th particle (as list).
            x_old = self.swarm[i].position

            # Get the local best position.
            l_best = self.swarm[i].best_position

            # Update all velocity values.
            for j, (xk, vk) in enumerate(zip(x_old, self._velocities[i])):
                # Apply the update equations.
                self._velocities[i][j] = (w * vk +
                                          c1[j] * np_subtract(l_best[j], xk) +
                                          c2[j] * np_subtract(g_best[j], xk))
        # _end_for_
    # _end_def_

    def update_positions(self) -> None:
        """
        Updates the positions of the particles in the swarm.

        :return: None.
        """

        # Evaluates all the particles.
        for particle, velocity in zip(self.swarm.population,
                                      self._velocities):
            # This calls internally the update method
            # for each data block.
            particle.position = velocity
    # _end_def_

    def reset_all(self) -> None:
        """
        Resets the particle positions, velocities and the statistics dictionary.

        :return: None.
        """
        # Randomize particle velocities.
        self.generate_uniform_velocities()

        # Randomize particle positions.
        self.generate_random_positions()

        # Clear all the internal bookkeeping.
        self.clear_all()
    # _end_def_

# _end_class_
