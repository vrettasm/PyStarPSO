from math import isclose

from numpy import sum as np_sum
from numpy import mean as np_mean
from numpy import array as np_array
from numpy import empty as np_empty
from numpy import arange as np_arange
from numpy import isscalar as np_isscalar
from numpy import subtract as np_subtract

from star_pso.engines.generic_pso import GenericPSO
from star_pso.auxiliary.utilities import (time_it,
                                          BlockType,
                                          check_parameters)
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
    # _end_def_

    def generate_uniform_velocities(self) -> None:
        """
        Generates random uniform velocities for the data blocks.

        :return: None.
        """

        # Here we generate the random velocities.
        for i, particle in enumerate(self.swarm.population):
            for j, blk in enumerate(particle):
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
            for j, blk in enumerate(particle):

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

        :return: None.
        """

        # Create a range of values.
        random_index = np_arange(self.n_cols)

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

    def update_velocities(self, options: dict) -> None:
        """
        Performs the update on the velocity equations.

        :param options: dictionary with the basic parameters:
              i)  'w': inertia weight
             ii) 'c1': cognitive coefficient
            iii) 'c2': social coefficient

        :return: None.
        """
        # Inertia weight parameter.
        w = options["w"]

        # Cognitive coefficient.
        c1 = options["c1"]

        # Social coefficient.
        c2 = options["c2"]

        # Global average parameter (OPTIONAL).
        g_avg = options.get("global_avg", False)

        # Get the shape of the velocity array.
        arr_shape = (self.n_rows, self.n_cols)

        # Pre-sample the coefficients.
        cogntv = JackOfAllTradesPSO.rng.uniform(0, c1, size=arr_shape)
        social = JackOfAllTradesPSO.rng.uniform(0, c2, size=arr_shape)

        # Get the GLOBAL best particle position.
        if g_avg:
            # Initialize an array with the best particle positions.
            g_best = np_array([particle.best_position
                               for particle in self.swarm.population],
                              dtype=object)

            # Get the mean value along the zero-axis.
            g_best = np_mean(g_best, axis=0)

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

        for i, (param_c, param_s) in enumerate(zip(cogntv, social)):
            # Get the (old) position of the i-th particle (as list).
            x_old = self.swarm[i].position

            # Get the local best position.
            l_best = self.swarm[i].best_position

            # Update all velocity values.
            for j, (xk, vk) in enumerate(zip(x_old, self._velocities[i])):
                # Apply the update equations.
                self._velocities[i][j] = (w * vk +
                                          param_c[j] * np_subtract(l_best[j], xk) +
                                          param_s[j] * np_subtract(g_best[j], xk))
        # _end_for_
    # _end_def_

    def update_positions(self, options: dict) -> None:
        """
        Updates the positions of the particles in the swarm.

        :param options: dictionary with options for the update
        equations, i.e. ('w', 'c1', 'c2', 'fipso').

        :return: None.
        """
        # Get the new updated velocities.
        self.update_velocities(options)

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

        # Clear the statistics.
        self.stats.clear()
    # _end_def_

    @time_it
    def run(self, max_it: int = 100, f_tol: float = None, options: dict = None,
            parallel: bool = False, reset_swarm: bool = False, verbose: bool = False) -> None:
        """
        Main method of the JackOfAllTradesPSO class, that implements the optimization
        routine.

        :param max_it: (int) maximum number of iterations in the optimization loop.

        :param f_tol: (float) tolerance in the difference between the optimal function
        value of two consecutive iterations. It is used to determine the convergence of
        the swarm. If this value is None (default) the algorithm will terminate using
        the max_it value.

        :param options: dictionary with the update equations options ('w': inertia weight,
        'c1': cognitive coefficient, 'c2': social coefficient).

        :param parallel: (bool) flag that enables parallel computation of the objective function.

        :param reset_swarm: (bool) if True it will reset the positions of the swarm to uniformly
        random respecting the boundaries of each space dimension.

        :param verbose: (bool) if True it will display periodically information about the current
        optimal function values.

        :return: None.
        """

        # Check if resetting the swarm is requested.
        if reset_swarm:
            self.reset_all()
        # _end_if_

        if options is None:
            # Default values of the simplified version.
            options = {"w": 0.5, "c1": 0.65, "c2": 0.65}
        else:
            # Ensure all the parameters are here.
            check_parameters(options)
        # _end_if_

        # Local variable to display information on the screen.
        # To avoid cluttering the screen we print info only 10
        # times regardless of the total number of iterations.
        its_time_to_print = (max_it // 10)

        # Get the function values 'before' optimisation.
        f_opt, _ = self.evaluate_function(parallel,
                                          jack_of_all_trades=True)
        # Display an information message.
        print(f"Initial f_optimal = {f_opt:.4f}")

        # Repeat for 'max_it' times.
        for i in range(max_it):

            # Update the positions in the swarm.
            self.update_positions(options)

            # Calculate the new function values.
            f_new, found_solution = self.evaluate_function(parallel,
                                                           jack_of_all_trades=True)
            # Check if we want to print output.
            if verbose and (i % its_time_to_print) == 0:
                # Display an information message.
                print(f"Iteration: {i + 1:>5} -> f_optimal = {f_new:.4f}")
            # _end_if_

            # Check for termination.
            if found_solution:
                # Update optimal function.
                f_opt = f_new

                # Display a warning message.
                print(f"{self.__class__.__name__} finished in {i + 1} iterations.")

                # Exit from the loop.
                break
            # _end_if_

            # Check for convergence.
            if f_tol and isclose(f_new, f_opt, rel_tol=f_tol):
                # Update optimal function.
                f_opt = f_new

                # Display a warning message.
                print(f"{self.__class__.__name__} converged in {i + 1} iterations.")

                # Exit from the loop.
                break
            # _end_if_

            # Update optimal function for next iteration.
            f_opt = f_new
        # _end_for_

        # Display an information message.
        print(f"Final f_optimal = {f_opt:.4f}")
    # _end_def_

# _end_class_
