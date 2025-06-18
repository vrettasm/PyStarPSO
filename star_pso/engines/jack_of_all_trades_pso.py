from math import (inf, isclose)

from os import cpu_count
from copy import deepcopy
from collections import defaultdict

from typing import Callable
from joblib import (Parallel, delayed)

import numpy as np
from numpy import sum as np_sum
from numpy import empty as np_empty
from numpy import isscalar as np_isscalar
from numpy import subtract as np_subtract
from numpy.random import (default_rng, Generator)

from star_pso.auxiliary.swarm import Swarm
from star_pso.auxiliary.utilities import (time_it,
                                          BlockType,
                                          check_parameters)
# Public interface.
__all__ = ["JackOfAllTradesPSO"]


class JackOfAllTradesPSO(object):
    """
    Description:

        JackOfAllTradesPSO class TBD.
    """

    # Make a random number generator.
    rng: Generator = default_rng()

    # Set the maximum number of CPUs (at least one).
    MAX_CPUs: int = 1 if not cpu_count() else cpu_count()

    # Object variables.
    __slots__ = ("_swarm", "objective_func", "_stats", "n_cpus",
                 "n_rows", "n_cols", "_velocities")

    def __init__(self,
                 initial_swarm: Swarm,
                 obj_func: Callable,
                 copy: bool = False,
                 n_cpus: int = None):
        """
        Default initializer of the JackOfAllTradesPSO class.

        :param initial_swarm: list of the initial population of particles.

        :param obj_func: callable objective function.

        :param copy: if true it will create a separate (deep) copy of the initial swarm.

        :param n_cpus: number of requested CPUs for the optimization process.
        """

        # Get the swarm population.
        self._swarm = deepcopy(initial_swarm) if copy else initial_swarm

        # Number of particles.
        self.n_rows = len(self._swarm)

        # Size (length) of particle.
        self.n_cols = len(self._swarm[0])

        # Make sure the fitness function is indeed callable.
        if not callable(obj_func):
            raise TypeError(f"{self.__class__.__name__}: Objective function is not callable.")
        else:
            # Get the objective function.
            self.objective_func = obj_func
        # _end_if_

        # Get the number of requested CPUs.
        if n_cpus is None:

            # This is the default option.
            self.n_cpus = max(1, JackOfAllTradesPSO.MAX_CPUs-1)
        else:

            # Assign the  requested number, making sure we have
            # enough CPUs and the value entered has the correct
            # type.
            self.n_cpus = max(1, min(JackOfAllTradesPSO.MAX_CPUs-1, int(n_cpus)))
        # _end_if_

        # Dictionary with statistics.
        self._stats = defaultdict(list)

        # First we declare the velocities to be
        # an [n_rows x n_cols] array of objects.
        self._velocities = np_empty(shape=(self.n_rows, self.n_cols),
                                    dtype=object)

        # Call the random velocity generator.
        self.generate_uniform_velocities()
    # _end_def_

    def generate_uniform_velocities(self) -> None:
        """
        Generates random uniform velocities
        for the data blocks.

        :return: None.
        """

        # Here we generate the random velocities.
        for i, particle in enumerate(self.swarm.population):
            for j, blk in enumerate(particle):
                # If the block is CATEGORICAL we
                # will use it's valid set length.
                n_vars = len(blk.valid_set) if blk.valid_set else 1

                # Generate the velocities randomly.
                self._velocities[i][j] = JackOfAllTradesPSO.rng.uniform(-1.0, +1.0,
                                                                        size=n_vars)
        # _end_for_
    # _end_def_

    @classmethod
    def set_seed(cls, new_seed=None) -> None:
        """
        Sets a new seed for the random number generator.

        :param new_seed: New seed value (default=None).

        :return: None.
        """
        # Re-initialize the class variable.
        cls.rng = default_rng(seed=new_seed)
    # _end_def_

    @property
    def stats(self) -> dict:
        """
        Accessor method that returns the 'stats' dictionary.

        :return: the dictionary with the statistics from the run.
        """
        return self._stats
    # _end_def_

    @property
    def swarm(self) -> Swarm:
        """
        Accessor of the swarm.

        :return: the reference the swarm.
        """
        return self._swarm
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

    def sample_categorical_values(self, positions) -> None:
        """
        Samples the actual position based on particles
        probabilities and valid sets for each data block.

        :param positions: the container with the lists
        of probabilities (one for each position).

        :return: None
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

    def evaluate_function(self, parallel=None) -> (float, bool):
        """
        Evaluate all the particles of the swarm with the custom
        objective function. The parallel_mode is optional.

        :return: the max function value and the found solution flag.
        """

        # Get a local copy of the objective function.
        func = self.objective_func

        # Extract the positions of the swarm in list.
        positions = self._swarm.positions_as_list()

        # Check if the swarm has categorical data blocks.
        if self.swarm.has_categorical:
            # Sample categorical variable.
            self.sample_categorical_values(positions)
        # _end_if_

        # Evaluates the particles in parallel mode.
        if parallel:
            # Evaluates the particles in parallel mode.
            f_evaluation = parallel(delayed(func)(x) for x in positions)
        else:
            # Evaluates the particles in serial mode.
            f_evaluation = [func(x) for x in positions]
        # _end_if_

        # Flag to indicate if a solution has been found.
        found_solution = False

        # Initialize f_max.
        f_max = -inf

        # Initialize the best position.
        x_best = None

        # Stores the function values.
        fx_array = np_empty(self.n_rows, dtype=float)

        # Update all particles with their new objective function values.
        for n, (p, result) in enumerate(zip(self._swarm, f_evaluation)):
            # Extract the n-th function value.
            f_value = result[0]

            # Attach the function value to each particle.
            p.value = f_value

            # Update the found solution.
            found_solution |= result[1]

            # Update the statistics.
            fx_array[n] = f_value

            # Update f_max value.
            if f_value > f_max:
                f_max = f_value
                x_best = positions[n]
        # _end_for_

        # Store the function values as ndarray.
        self.stats["f_values"].append(fx_array)

        # Store the best (sampled) position.
        self.stats["x_best"].append(x_best)

        # Update local best for consistent results.
        self.swarm.update_local_best()

        # Return the tuple.
        return f_max, found_solution
    # _end_def_

    def update_velocities(self, options: dict) -> None:
        """
        Performs the update on the velocity equations
        according to the original PSO paper by:
        "Kennedy, J. and Eberhart, R. (1995)".

        :param options: Dictionary with the basic options:
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
            g_best = np.array([particle.best_position
                               for particle in self.swarm.population],
                              dtype=object)

            # Get the mean value along the zero-axis.
            g_best = np.mean(g_best, axis=0)

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

    @time_it
    def run(self, max_it: int = 100, f_tol: float = None, options: dict = None,
            reset_swarm: bool = False, verbose: bool = False) -> None:
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

        :param reset_swarm: if true it will reset the positions of the swarm to uniformly
        random respecting the boundaries of each space dimension.

        :param verbose: (bool) if 'True' it will display periodically information about
        the current optimal function values.

        :return: None.
        """

        # Check if resetting the swarm is requested.
        if reset_swarm:
            # Randomize particle velocities.
            self.generate_uniform_velocities()

            # Randomize particle positions.
            self.generate_random_positions()

            # Clear the statistics.
            self.stats.clear()
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

        # Reuse the pool of workers for the whole optimization.
        with Parallel(n_jobs=self.n_cpus, prefer="threads") as parallel:

            # Get the function values 'before' optimisation.
            f_opt, _ = self.evaluate_function(parallel)

            # Display an information message.
            print(f"Initial f_optimal = {f_opt:.4f}")

            # Repeat for 'max_it' times.
            for i in range(max_it):

                # Update the positions in the swarm.
                self.update_positions(options)

                # Calculate the new function values.
                f_new, found_solution = self.evaluate_function(parallel)

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
        # _end_with_

        # Display an information message.
        print(f"Final f_optimal = {f_opt:.4f}")
    # _end_def_

    def __call__(self, *args, **kwargs):
        """
        This method is only a wrapper of the "run" method.
        """
        return self.run(*args, **kwargs)
    # _end_def_

# _end_class_
