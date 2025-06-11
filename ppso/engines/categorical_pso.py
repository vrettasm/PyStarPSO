from math import isclose

import numpy as np
from numpy import sum as np_sum
from numpy import clip as np_clip
from numpy import subtract as np_subtract

from ppso.auxiliary.utilities import time_it
from ppso.engines.generic_pso import GenericPSO

# Public interface.
__all__ = ["CategoricalPSO"]


class CategoricalPSO(GenericPSO):
    """
    Description:

    This implements a simplified variant of the original ICPSO algorithm as described in:

    Strasser, S., Goodman, R., Sheppard, J., et al. A new discrete particle swarm optimization
    algorithm", Proceedings of 2016 Genetic and Evolutionary Computation Conference (GECCO) 16,
    ACM Press, Denver, Colorado, USA, pp. 53-60.

    """

    # Object variables (specific for the CategoricalPSO).
    __slots__ = ("_velocities",)

    def __init__(self, variable_sets: list, **kwargs):
        """
        Default initializer of the CategoricalPSO class.

        :param variable_sets: this is list with the sets
        (one for each optimization variable).

        :param kwargs: these are the default parameters
        for the GenericPSO.
        """

        # First call the super initializer.
        super().__init__(**kwargs)

        # Local copy of the variable sets.
        self._items = {"sets": variable_sets}

        # First we declare the velocities to be
        # an [n_rows x n_cols] array of objects.
        self._velocities = np.empty(shape=(self.n_rows, self.n_cols),
                                    dtype=object)

        # Call the random velocity generator.
        self.generate_uniform_velocities()
    # _end_def_

    def generate_uniform_velocities(self) -> None:
        """
        Generates random uniform velocities for the
        categorical variable positions.

        :return: None.
        """
        # Local copy of the variable sets.
        v_sets = self._items["sets"]

        # Here we generate the random velocities
        # in a short uniform range, according to
        # the size of the variable set.
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                self._velocities[i, j] = GenericPSO.rng_PSO.uniform(-0.1, +0.1,
                                                                    size=len(v_sets[j]))
        # _end_for_
    # _end_def_

    def generate_categorical_positions(self) -> None:
        """
        Generates random uniform positions
        for the categorical variables.

        :return: None.
        """
        # Local copy of the valid sets.
        v_sets = self._items["sets"]

        # Reset the positions to uniform values.
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                # Get the length of the j-th set.
                size_j = len(v_sets[j])

                # Set the variables uniformly.
                self.swarm.position_at(i)[j] = np.ones(size_j)/size_j
        # _end_for_
    # _end_def_

    def update_velocities(self, options: dict) -> None:
        """
        Performs the update on the velocity equations.

        :param options: Dictionary with the basic PSO options:
              i)  'w': inertia weight
             ii) 'c1': cognitive coefficient
            iii) 'c2': social coefficient

        :return: None.
        """
        # Inertia weight parameter.
        w = options.get("w")

        # Cognitive coefficient.
        c1 = options.get("c1")

        # Social coefficient.
        c2 = options.get("c2")

        # Fully informed PSO option.
        fipso = options.get("fipso", False)

        # Get the shape of the velocity array.
        arr_shape = (self.n_rows, self.n_cols)

        # Pre-sample the coefficients.
        R1 = GenericPSO.rng_PSO.uniform(0, c1, size=arr_shape)
        R2 = GenericPSO.rng_PSO.uniform(0, c2, size=arr_shape)

        # Get the GLOBAL best particle position.
        if fipso:
            # Initialize a vector (of vectors).
            g_best = np.array([np.zeros(len(k)) for k in self._items["sets"]],
                              dtype=object)

            # Accumulate (per positional variable)
            # the best positions in the swarm.
            for particle in self.swarm.population:
                for i, pos in enumerate(particle.best_position):
                    g_best[i] += pos
            # _end_for_

            # Finally process the accumulated vectors.
            for i in range(self.n_cols):
                # Get the averaged values.
                g_best[i] /= self.n_rows

                # Normalize them to account for probabilities.
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

            # Update all positions.
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

        N.B. This method is very slow O(n_rows*n_cols).

        :param options: dictionary with options for the update equations.

        :return: None.
        """

        # Update the velocity equations.
        self.update_velocities(options)

        # Local reference of the population.
        population = self._swarm.population

        # Update all particle positions.
        for particle, v_upd in zip(population, self._velocities):

            # Process each position separately.
            for x_j, v_j in zip(particle.position, v_upd):

                # Update j-th position.
                x_j += v_j

                # Ensure they stay within limits.
                np_clip(x_j, 0.0, 1.0, out=x_j)

                # Ensure there will be at least one
                # element with positive probability.
                if np.allclose(x_j, 0.0):
                    x_j[GenericPSO.rng_PSO.integers(len(x_j))] = 1.0
                # _end_if_

                # Normalize (to account for probabilities).
                x_j /= np_sum(x_j, dtype=float)
            # _end_for_
        # _end_for_
    # _end_def_

    @time_it
    def run(self, max_it: int = 100, f_tol: float = None, options: dict = None,
            parallel: bool = False, reset_swarm: bool = False, verbose: bool = False) -> None:
        """
        Main method of the StandardPSO class, that implements the optimization routine.

        :param max_it: (int) maximum number of iterations in the optimization loop.

        :param f_tol: (float) tolerance in the difference between the optimal function value
        of two consecutive iterations. It is used to determine the convergence of the swarm.
        If this value is None (default) the algorithm will terminate using the max_it value.

        :param options: dictionary with the update equations options ('w': inertia weight,
        'c1': cognitive coefficient, 'c2': social coefficient).

        :param parallel: (bool) Flag that enables parallel computation of the objective function.

        :param reset_swarm: if true it will reset the positions of the swarm to uniformly random
        respecting the boundaries of each space dimension.

        :param verbose: (bool) if 'True' it will display periodically information about the
        current optimal function values.

        :return: None.
        """

        # Check if resetting the swarm is required.
        if reset_swarm:
            # Reset particle velocities.
            self.generate_uniform_velocities()

            # Reset particle positions.
            self.generate_categorical_positions()

            # Clear the statistics.
            self.stats.clear()
        # _end_if_

        # If options is not given, set the
        # parameters of the original paper.
        if options is None:
            # Default values of the simplified version.
            options = {"w": 0.5, "c1": 0.1, "c2": 0.1}
        else:
            # Make sure the right keys exist.
            for key in {"w", "c1", "c2"}:
                if key not in options:
                    raise ValueError(f"{self.__class__.__name__}: "
                                     f"Option '{key}' is missing. ")
            # _end_for_
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
