from math import isclose
from functools import cached_property

import numpy as np
from numpy import sum as np_sum
from numpy import clip as np_clip
from numpy import subtract as np_subtract

from star_pso.engines.generic_pso import GenericPSO
from star_pso.auxiliary.utilities import (time_it,
                                          check_parameters)
# Public interface.
__all__ = ["CategoricalPSO"]


class CategoricalPSO(GenericPSO):
    """
    Description:

    This implements a simplified variant of the original ICPSO algorithm as described in:

    Strasser, S., Goodman, R., Sheppard, J., et al. "A new discrete particle swarm optimization
    algorithm", Proceedings of 2016 Genetic and Evolutionary Computation Conference (GECCO) 16,
    ACM Press, Denver, Colorado, USA, pp. 53-60.
    """

    def __init__(self, variable_sets: list, permutation_mode: bool = False,
                 **kwargs):
        """
        Default initializer of the CategoricalPSO class.

        :param variable_sets: this is list with the sets
        (one for each optimization variable).

        :param permutation_mode: (bool) if True it will sample
        permutations of the valid sets.

        :param kwargs: these are the default parameters for the
        GenericPSO.
        """

        # First call the super initializer.
        super().__init__(**kwargs)

        # Local copy of the variable sets.
        self._valid_sets = variable_sets

        # First we declare the velocities to be
        # an [n_rows x n_cols] array of objects.
        self._velocities = np.empty(shape=(self.n_rows, self.n_cols),
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

    @cached_property
    def size_of_sets(self) -> list[int]:
        """
        Compile a list with the sizes of the valid sets.
        To avoid recomputing the list over and over again
        we decorate it with @cached_property.

        :return: a list with the sizes of the valid sets.
        """
        return [len(k) for k in self._valid_sets]
    # _end_def_

    def generate_uniform_velocities(self) -> None:
        """
        Generates random uniform velocities for the
        categorical variable positions.

        :return: None.
        """
        # Get the length of each set.
        size_L = self.size_of_sets

        # Here we generate the random velocities
        # in a short uniform range, according to
        # the size of the variable set.
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                self._velocities[i, j] = GenericPSO.rng.uniform(-0.1, +0.1,
                                                                size=size_L[j])
        # _end_for_
    # _end_def_

    def generate_random_positions(self) -> None:
        """
        Generate the population of particles positions setting
        their probabilities to 1/L (for all possible L states).

        :return: None.
        """

        # Get the length of each set.
        size_L = self.size_of_sets

        # Reset the probabilities to uniform values.
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                # Get the length of the j-th set.
                size_j = size_L[j]

                # Set the variables uniformly.
                self.swarm[i][j] = np.ones(size_j)/size_j
        # _end_for_
    # _end_def_

    def sample_categorical_values(self, positions) -> None:
        """
        Samples an actual categorical position based on
        particle's probabilities and valid set for each
        particle position in the swarm.

        :return: None.
        """

        # Local copy of the valid sets.
        local_sets = self._valid_sets

        # Loop over all particle positions.
        for i, x_pos in enumerate(positions):

            # Each position is sampled according to its
            # particle probabilities and its valid set.
            for j, (set_j, probs_j) in enumerate(zip(local_sets, x_pos)):

                # Sample an item according to its probabilities.
                # WARNING: shuffle option MUST be set to False!
                x_pos[j] = GenericPSO.rng.choice(set_j,
                                                 p=probs_j,
                                                 shuffle=False)
        # _end_for_
    # _end_def_

    def sample_permutation_values(self, positions) -> None:
        """
        Samples a permutation from a given set of variables.
        It is used in problems like the 'Traveling Salesman'.

        It is assumed that all particle variables have the
        same valid set of values.

        :return: None.
        """

        # Local copy of the valid sets.
        local_sets = self._valid_sets

        # Create a range of values.
        random_index = np.arange(self.n_cols)

        # Shuffle in place. This is used to avoid introducing
        # biasing by using always the same order of blocks to
        # select first their categorical sample value.
        CategoricalPSO.rng.shuffle(random_index)

        # Loop over all particle positions.
        for i, x_pos in enumerate(positions):

            # Auxiliary set.
            exclude_idx = set()

            # Scan the particle using the random index.
            for j in random_index:
                # Get the j-th element.
                xj = x_pos[j]

                # Reference of the j-th set.
                set_j = local_sets[j]

                # Sort in reverse order from high to low.
                for k in xj.argsort()[::-1]:
                    # Continue until we find the first
                    # unused element of the valid set.
                    if k not in exclude_idx:
                        # Assign the element in
                        # the position[i, j].
                        x_pos[j] = set_j[k]

                        # Update the set() with
                        # the excluded indexes.
                        exclude_idx.add(k)

                        # Break the internal loop.
                        break
                    # _end_if_
            # _end_for_
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
        R1 = GenericPSO.rng.uniform(0, c1, size=arr_shape)
        R2 = GenericPSO.rng.uniform(0, c2, size=arr_shape)

        # Get the GLOBAL best particle position.
        if g_avg:
            # Initialize a vector (of vectors).
            g_best = np.array([particle.best_position
                              for particle in self.swarm.population],
                              dtype=object)

            # Get the mean value along the zero-axis.
            g_best = np.mean(g_best, axis=0)

            # Finally normalize them to
            # account for probabilities.
            for i in range(self.n_cols):
                g_best[i] /= np_sum(g_best[i], dtype=float)
            # _end_for_
        else:
            g_best = self.swarm.best_particle().position
        # _end_if_

        for i, (r1, r2) in enumerate(zip(R1, R2)):

            # Get the current position of i-th the particle.
            x_i = self.swarm[i].position

            # Get the Best local position.
            l_best = self.swarm[i].best_position

            # Update all velocities.
            for j, (xk, vk) in enumerate(zip(x_i, self._velocities[i])):

                # Apply the update equations.
                vk = (w * vk +
                      r1[j] * np_subtract(l_best[j], xk) +
                      r2[j] * np_subtract(g_best[j], xk))

                # Ensure the velocities are within limits.
                np_clip(vk, -0.5, +0.5, out=vk)
            # _end_for_
        # _end_for_
    # _end_def_

    def update_positions(self, options: dict) -> None:
        """
        Updates the positions of the particles in the swarm.

        :param options: dictionary with options for the update
        equations.

        :return: None.
        """

        # Update the velocity equations.
        self.update_velocities(options)

        # Update all particle positions.
        for particle, v_upd in zip(self._swarm.population,
                                   self._velocities):

            # Process each position separately.
            for k, (x_j, v_j) in enumerate(zip(particle.position, v_upd)):

                # Update j-th position.
                x_j += v_j

                # Ensure they stay within limits.
                np_clip(x_j, 0.0, 1.0, out=x_j)

                # Ensure there will be at least one
                # element with positive probability.
                if all(np.isclose(x_j, 0.0)):
                    x_j[GenericPSO.rng.integers(len(x_j))] = 1.0
                # _end_if_

                # Normalize (to account for probabilities).
                x_j /= np_sum(x_j, dtype=float)
        # _end_for_
    # _end_def_

    def reset_all(self) -> None:
        """
        Resets the particle positions, velocities and the statistics dictionary.

        :return: None.
        """
        # Reset particle velocities.
        self.generate_uniform_velocities()

        # Reset particle positions.
        self.generate_random_positions()

        # Clear the statistics.
        self.stats.clear()
    # _end_def_

    @time_it
    def run(self, max_it: int = 100, f_tol: float = None, options: dict = None,
            parallel: bool = False, reset_swarm: bool = False, verbose: bool = False) -> None:
        """
        Main method of the CategoricalPSO class, that implements the optimization routine.

        :param max_it: (int) maximum number of iterations in the optimization loop.

        :param f_tol: (float) tolerance in the difference between the optimal function value
        of two consecutive iterations. It is used to determine the convergence of the swarm.
        If this value is None (default) the algorithm will terminate using the max_it value.

        :param options: dictionary with the update equations options ('w': inertia weight,
        'c1': cognitive coefficient, 'c2': social coefficient).

        :param parallel: (bool) flag that enables parallel computation of the objective function.

        :param reset_swarm: (bool) if True it will reset the positions of the swarm to uniformly
        random respecting the boundaries of each space dimension.

        :param verbose: (bool) if True it will display periodically information about the current
        optimal function values.

        :return: None.
        """

        # Check if resetting the swarm is required.
        if reset_swarm:
            self.reset_all()
        # _end_if_

        if options is None:
            # Default values of the simplified version.
            options = {"w": 0.5, "c1": 0.1, "c2": 0.1}
        else:
            # Ensure all the parameters are here.
            check_parameters(options)
        # _end_if_

        # Get the function values before optimisation.
        f_opt, _ = self.evaluate_function(parallel,
                                          categorical_mode=True)
        # Display an information message.
        print(f"Initial f_optimal = {f_opt:.4f}")

        # Local variable to display information on the screen.
        # To avoid cluttering the screen we print info only 10
        # times regardless of the total number of iterations.
        its_time_to_print = (max_it // 10)

        # Repeat for 'max_it' times.
        for i in range(max_it):

            # Update the positions in the swarm.
            self.update_positions(options)

            # Calculate the new function values.
            f_new, found_solution = self.evaluate_function(parallel,
                                                           categorical_mode=True)
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
