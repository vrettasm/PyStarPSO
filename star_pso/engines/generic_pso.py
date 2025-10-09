from os import cpu_count
from copy import deepcopy
from math import inf, isclose
from operator import attrgetter
from collections import defaultdict

from typing import Callable
from joblib import (Parallel, delayed)

import numpy as np
from numpy.typing import ArrayLike
from numpy import array as np_array
from numpy import empty as np_empty
from numpy.random import (default_rng, Generator)

from star_pso.auxiliary.swarm import Swarm, SwarmParticle
from star_pso.auxiliary.utilities import (time_it, VOptions, nb_clip_item,
                                          SpecialMode, check_parameters,
                                          linear_rank_probabilities)
# Public interface.
__all__ = ["GenericPSO"]


class GenericPSO(object):
    """
    Description:

        GenericPSO class models the interface of a specific particle swarm
        optimization model or engine. It provides the common variables and
        functionalities that all PSO models should share.
    """

    # Make a random number generator.
    rng: Generator = default_rng()

    # Set the maximum number of CPUs (at least one).
    MAX_CPUs: int = 1 if not cpu_count() else cpu_count()

    # Object variables.
    __slots__ = ("_swarm", "_velocities", "objective_func", "_upper_bound", "_lower_bound", "_stats",
                 "_items", "_f_eval", "n_cpus", "n_rows", "n_cols", "w_min", "w_max", "_special_mode")

    def __init__(self,
                 initial_swarm: Swarm,
                 obj_func: Callable,
                 lower_bound: ArrayLike = None,
                 upper_bound: ArrayLike = None,
                 copy: bool = False,
                 n_cpus: int = None):
        """
        Default initializer of the GenericPSO class.

        :param initial_swarm: list of the initial population of particles.

        :param obj_func: callable objective function.

        :param lower_bound: lower search space bound.

        :param upper_bound: upper search space bound.

        :param copy: if True it will create a separate (deep) copy of the initial swarm.

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

        # Set the upper/lower bounds of the search space.
        self._lower_bound = np_array(lower_bound)
        self._upper_bound = np_array(upper_bound)

        # Get the number of requested CPUs.
        if n_cpus is None:

            # This is the default option.
            self.n_cpus = max(1, GenericPSO.MAX_CPUs-1)
        else:

            # Assign the  requested number, making sure we have
            # enough CPUs and the value entered has the correct
            # type.
            self.n_cpus = max(1, min(GenericPSO.MAX_CPUs-1, int(n_cpus)))
        # _end_if_

        # Dictionary with statistics.
        self._stats = defaultdict(list)

        # Place holder.
        self._items = None

        # Set the function evaluation to zero.
        self._f_eval = 0

        # Set the min/max values for the inertia parameters.
        self.w_min = 0.0
        self.w_max = 1.0

        # Set the special mode to Normal.
        self._special_mode = SpecialMode.NORMAL
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
    def f_eval(self) -> int:
        """
        Accessor method that returns the value of the f_eval.

        :return: (int) the counted number of function evaluations.
        """
        return self._f_eval
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
    def items(self) -> list | tuple:
        """
        Accessor (getter) of the _items placeholder container.

        :return: _items (if any).
        """
        return self._items
    # _end_def_

    @property
    def swarm(self) -> Swarm:
        """
        Accessor of the swarm.

        :return: the reference the swarm.
        """
        return self._swarm
    # _end_def_

    def _get_typed_positions(self) -> list | np.ndarray:
        """
        Extracts the positions from the swarm and returns them
        in their correct type according to the setting of the algorithm.

        :return: the particle positions either as list or ndarray.
        """
        # Check if "Jack of All Trades" is enabled.
        if self._special_mode == SpecialMode.JACK_OF_ALL_TRADES:

            # Extract the positions in a list of lists.
            positions = self._swarm.positions_as_list()

            # Check if the swarm has categorical data blocks.
            if self.swarm.has_categorical:
                # Sample categorical variable.
                self._items["sample_random_values"](positions)
            # _end_if_
        else:
            # Extract the positions in a 2D numpy array.
            positions = self._swarm.positions_as_array()

            # Only True in CategoricalPSO.
            if self._special_mode == SpecialMode.CATEGORICAL:
                # Sample categorical variable.
                self._items["sample_random_values"](positions)
            # _end_if_
        # _end_if_

        return positions
    # _end_def_

    def evaluate_function(self, parallel_mode: bool = False,
                          backend: str = "threads") -> tuple[list[float], bool]:
        """
        Evaluate all the particles of the input list with the custom objective
        function. The parallel_mode is optional.

        :param parallel_mode: (bool) enables parallel computation of the objective
        function. Default is False (serial execution).

        :param backend: backend for the parallel Joblib ('threads' or 'processes').

        :return: the max function value and the found solution flag.
        """

        # Extract the correct type positions.
        positions = self._get_typed_positions()

        # Get a local copy of the objective function.
        func = self.objective_func

        # Check the 'parallel_mode' flag.
        if parallel_mode:

            # Evaluate the particles in parallel mode.
            f_evaluation = Parallel(n_jobs=self.n_cpus, prefer=backend)(
                delayed(func)(x) for x in positions
            )
        else:

            # Evaluate all the particles in serial mode.
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
        for n, (p, result) in enumerate(zip(self._swarm.population, f_evaluation)):
            # Extract the n-th function value.
            f_value = result["f_value"]

            # Attach the function value to each particle.
            p.value = f_value

            # Update the found solution.
            found_solution |= result["solution_is_found"]

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

        # Store the optimal f-value of this
        # iteration.
        self.stats["f_best"].append(f_max)

        # Update the counter of function evaluations.
        self._f_eval += self.swarm.size

        # Update local best for consistent results.
        self.swarm.update_local_best()

        # Return the tuple.
        return f_max, found_solution
    # _end_def_

    def clear_all(self) -> None:
        """
        Clears the stats dictionary and the f_eval counter.

        :return: None.
        """
        self.stats.clear()
        self._f_eval = 0
    # _end_def_

    def get_optimal_values(self) -> tuple:
        """
        Iterates through the stats to find the best recorded
        position from all the iterations.

        :return: a tuple with the optimal particle position,
                 its function value and the iteration it was
                 found.
        """

        # Get the maximum of f_best.
        f_opt = max(self.stats["f_best"])

        # Get the index of f_best.
        i_opt = self.stats["f_best"].index(f_opt)

        # Get the corresponding x_best.
        x_opt = self.stats["x_best"][i_opt]

        # Return the optimal particle position,
        # along with its function value and its
        # iteration.
        return i_opt, f_opt, x_opt
    # _end_def_

    def reset_all(self) -> None:
        """
        Resets the particle positions, velocities
        and the  statistics dictionary. Since the
        various implementations vary  this method
        should be implemented separately.

        :return: None.
        """
        raise NotImplementedError(f"{self.__class__.__name__}: "
                                  f"You should implement this method!")
    # _end_def_

    def generate_random_positions(self) -> None:
        """
        Generate a population of particles with random positions.
        Each different class that inherits from here should know
        how to implement it.

        :return: None.
        """
        raise NotImplementedError(f"{self.__class__.__name__}: "
                                  f"You should implement this method!")
    # _end_def_

    @staticmethod
    def fully_informed_global_best(population: list[SwarmParticle]) -> np.ndarray:
        """
        Uses the whole input population and computes a weighted average position
        according to the linear ranking of the particles. Those with a higher
        function value also have a bigger weight in the calculation.

        :param population: a list of particles which we want to consider in the
        calculation of the global best position.

        :return: the gBest (as numpy array).
        """
        # Extract only their positions and convert to numpy array.
        all_positions = np_array([item.position for item in sorted(population,
                                                                   key=attrgetter("value"))])
        # Compute the probabilities.
        p_weights, p_weights_sum = linear_rank_probabilities(len(population))

        # Take a "weighted average" from all the positions of the swarm.
        g_best = np.multiply(all_positions,
                             p_weights[:, np.newaxis]).sum(axis=0) / p_weights_sum

        # Return the global best position.
        return g_best
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

        # Get the global best.
        if params.fipso:

            # Compute a weighted average from all the positions of the swarm,
            # according to their linear ranking.
            g_best = GenericPSO.fully_informed_global_best(self.swarm.population)
        else:

            # Get the swarm's best particle position.
            g_best = self.swarm.best_particle().position
        # _end_if_

        # Inertia weight parameter.
        w = params.w0

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
        raise NotImplementedError(f"{self.__class__.__name__}: "
                                  f"You should implement this method!")
    # _end_def_

    def calculate_spread(self) -> float:
        """
        Calculates a measure of how spread are the particle
        positions, according to the specific type of PSO.

        :return: None.
        """
        raise NotImplementedError(f"{self.__class__.__name__}: "
                                  f"You should implement this method!")
    # _end_def_

    def set_inertia_min_max(self,
                            new_min: float = None,
                            new_max: float = None) -> None:
        """
        Sets the inertia minimum and maximum values. Calling
        the function without new values it will set the w_min
        and w_max to their default values (0 and 1).

        :param new_min: Minimum value for the inertia.
        :param new_max: Maximum value for the inertia.

        :return: None.
        """
        # If new_min is not provided,
        # set it to 0.0 (default value).
        if new_min is None:
            new_min = 0.0
        # _end_if_

        # If new_max is not provided,
        # set it to 1.0 (default value).
        if new_max is None:
            new_max = 1.0
        # _end_if_

        # Make sure new_min is less than new_max.
        if new_min >= new_max:
            raise ValueError(f"{self.__class__.__name__}: "
                             f"New minimum value {new_min} is larger than new maximum value {new_max}.")
        # _end_if_

        # Ensure the final values are in [0, 1].
        self.w_min = nb_clip_item(new_min, 0.0, 1.0)
        self.w_max = nb_clip_item(new_max, 0.0, 1.0)
    # _end_def_

    def adapt_velocity_parameters(self, options: dict) -> None:
        """
        Provides a very basic adapt mechanism for the PSO update
        velocity parameters. It can be used as a placeholder for
        more advanced techniques.

        :param options: (dict) contains the previous estimates of
        the PSO parameters.

        :return: None.
        """
        # For the moment we hardcode the min/max
        # values of the c1 and c2 parameters.
        c_min, c_max = 0.1, 2.5

        # Get an estimate of the particles' spread,
        # ensuring its range in [0, 1].
        spread_t = nb_clip_item(self.calculate_spread(),
                                0.0, 1.0)

        # Compute the new inertia weight parameter.
        wt = self.w_max - (self.w_max - self.w_min) * spread_t

        # Get the previous values of the parameters.
        w0 = options["w0"]
        c1 = options["c1"]
        c2 = options["c2"]

        # To reduce "noise effects" we allow the update only if the
        # new inertia parameter "wt" is different then ~5% from the
        # previous one "w0".
        if abs(wt - w0) > 0.05:
            
            # Update the cognitive and social parameters.
            if wt > w0:
                # If the inertia weight has increased,
                # then decrease the c1/c2 coefficients.
                c1 *= 0.9
                c2 *= 0.9
            else:
                # If the inertia weight has decreased,
                # then increase the c1/c2 coefficients.
                c1 *= 1.1
                c2 *= 1.1
            # _end_if_

            # Ensure the updated c1 / c2 values
            # stay within their bounds.
            c1 = nb_clip_item(c1, c_min, c_max)
            c2 = nb_clip_item(c2, c_min, c_max)

            # Update the dictionary.
            options["w0"] = wt
            options["c1"] = c1
            options["c2"] = c2

            # Store the updated parameters.
            self.stats["inertia_w"].append(wt)
            self.stats["cogntv_c1"].append(c1)
            self.stats["social_c2"].append(c2)
        # _end_if_
    # _end_def_

    @time_it
    def run(self, max_it: int = 1000, options: dict = None, parallel: bool = False,
            reset_swarm: bool = False, f_tol: float = None, f_max_eval: int = None,
            adapt_params: bool = False, verbose: bool = False) -> None:
        """
        Main method of the GenericPSO class that implements the optimization routine.

        :param max_it: (int) maximum number of iterations in the optimization loop.

        :param f_tol: (float) tolerance in the difference between the optimal function
        value of two consecutive iterations. It is used to determine the convergence of
        the swarm. If this value is None (default) the algorithm will terminate using
        the max_it value.

        :param options: dictionary with update equations options ('w': inertia weight,
        'c1': cognitive coefficient, 'c2': social coefficient).

        :param parallel: (bool) flag that enables parallel computation of the objective
        function.

        :param reset_swarm: (bool) if True it will reset the positions of the swarm to
        uniformly random respecting the boundaries of each space dimension.

        :param f_max_eval: (int) it sets an upper limit of function evaluations. If the
        number is exceeded the algorithm stops.

        :param adapt_params: (bool) If set to "True" it will allow the inertia, cognitive
        and social parameters to adapt according to the convergence of the swarm population
        to a single solution. Default is set to "False".

        :param verbose: (bool) if True it will display periodically information about the
        current optimal function values.

        :return: None.
        """
        # Check if resetting the swarm is requested.
        if reset_swarm:
            self.reset_all()
        # _end_if_

        if options is None:
            # Default values of the simplified version.
            options = {"w0": 0.5, "c1": 2.0, "c2": 2.0}
        else:
            # Ensure all the parameters are here.
            check_parameters(options)
        # _end_if_

        # Convert options dict to VOptions.
        params = VOptions(**options)

        # Get the function values before optimisation.
        f_opt, _ = self.evaluate_function(parallel)

        # Display an information message.
        print(f"Initial f_optimal = {f_opt:.4f}")

        # Local variable to display information on the screen.
        # To avoid cluttering the screen we print info only 10
        # times regardless of the total number of iterations.
        its_time_to_print = (max_it // 10)

        # Repeat for 'max_it' times.
        for i in range(max_it):

            # First update the velocity equations.
            self.update_velocities(params)

            # Then update the positions in the swarm.
            self.update_positions()

            # Calculate the new function values.
            f_new, found_solution = self.evaluate_function(parallel)

            # Check if we want to print output.
            if verbose and (i % its_time_to_print) == 0:
                # Display an information message.
                print(f"Iteration: {i + 1:>5} -> f_optimal = {f_new:.4f}")
            # _end_if_

            # Check for the maximum function evaluations.
            if f_max_eval and self._f_eval >= f_max_eval:
                # Update optimal function.
                f_opt = f_new

                # Display an information message.
                print(f"{self.__class__.__name__} "
                      "Reached the maximum number of function evaluations.")

                # Exit from the loop.
                break
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
            if f_tol and isclose(f_new, f_opt, rel_tol=1.0e-5, abs_tol=f_tol):
                # Update optimal function.
                f_opt = f_new

                # Display a warning message.
                print(f"{self.__class__.__name__} converged in {i + 1} iterations.")

                # Exit from the loop.
                break
            # _end_if_

            # Check for adapting the parameters.
            if adapt_params and (i % 5) == 0:
                # Get a copy of the previous parameters.
                dict_options = params._asdict()

                # Update the parameters.
                self.adapt_velocity_parameters(dict_options)

                # Convert  the new  parameters to
                # VOptions for the next iteration.
                params = VOptions(**dict_options)
            # _end_if_

            # Update optimal function for next iteration.
            f_opt = f_new
        # _end_for_

        # Display an information message.
        print(f"Final f_optimal = {f_opt:.4f}")
    # _end_def_

    def __call__(self, *args, **kwargs) -> None:
        """
        Wrapper of the run() method.
        """
        return self.run(*args, **kwargs)
    # _end_def_

# _end_class_
