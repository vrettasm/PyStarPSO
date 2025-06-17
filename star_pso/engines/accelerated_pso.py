from math import isclose

import numpy as np
from numpy.typing import ArrayLike

from star_pso.auxiliary.utilities import time_it
from star_pso.engines.generic_pso import GenericPSO

# Public interface.
__all__ = ["AcceleratedPSO"]


class AcceleratedPSO(GenericPSO):
    """
    Description:

    This class implements the 'Accelerated Particle Swarm Optimization' variant
    as described in:

    Yang, X. S., Deb, S., and Fong, S., (2011), Accelerated Particle Swarm Optimization
    and Support Vector Machine for Business Optimization and Applications, in: Networked
    Digital Technologies (NDT2011), Communications in Computer and Information Science,
    Vol. 136, Springer, pp. 53-66.

    """

    def __init__(self, x_min: ArrayLike, x_max: ArrayLike, **kwargs):
        """
        Default initializer of the AcceleratedPSO class.

        :param x_min: lower search space bound.

        :param x_max: upper search space bound.
        """

        # Call the super initializer with the input parameters.
        super().__init__(lower_bound=x_min, upper_bound=x_max, **kwargs)
    # _end_def_

    def update_positions(self, options: dict) -> None:
        """
        Updates the positions of the particles in the swarm.

        :param options: dictionary with options for the update
        equations, i.e. ('alpha', 'beta', 'fipso').

        :return: None.
        """
        # Get the 'alpha' parameter.
        c_alpha = options.get("alpha")

        # Get the 'beta' parameter.
        c_beta = options.get("beta")

        # Fully informed PSO option.
        fipso = options.get("fipso", False)

        # Get the GLOBAL best particle position.
        if fipso:
            # In the fully informed case we take the average of all the best positions.
            g_best = np.mean([p.best_position for p in self.swarm.population], axis=0)
        else:
            g_best = self.swarm.best_particle().position
        # _end_if_

        # Temporary 'velocity-like' parameters.
        tmp_velocities = (c_beta*g_best + GenericPSO.rng.normal(0, c_alpha,
                                                                size=(self.n_rows, self.n_cols)))
        # Compute the complement of beta.
        c_param = (1.0 - c_beta)

        # Update all particle positions.
        for particle, velocity in zip(self._swarm.population, tmp_velocities):
            # Ensure the particle stays within bounds.
            np.clip(c_param*particle.position + velocity,
                    self._lower_bound, self._upper_bound, out=particle.position)
    # _end_def_

    @time_it
    def run(self, max_it: int = 100, f_tol: float = None, options: dict = None,
            parallel: bool = False, reset_swarm: bool = False, verbose: bool = False) -> None:
        """
        Main method of the AcceleratedPSO class, that implements the optimization routine.

        :param max_it: (int) maximum number of iterations in the optimization loop.

        :param f_tol: (float) tolerance in the difference between the optimal function value
        of two consecutive iterations. It is used to determine the convergence of the swarm.
        If this value is None (default) the algorithm will terminate using the max_it value.

        :param options: dictionary with the update equations options ('alpha': 0.1xL ~ 0.5xL,
        'beta': 0.1 ~ 0.7), where L is the typical length of the problem at hand.

        :param parallel:(bool) Flag that enables parallel computation of the objective function.

        :param reset_swarm: if true it will reset the positions of the swarm to uniformly random
        respecting the boundaries of each space dimension.

        :param verbose: (bool) if 'True' it will display periodically information about the
        current optimal function values.

        :return: None.
        """

        # Check if resetting the swarm is required.
        if reset_swarm:
            # Generate random positions.
            self.generate_uniform_positions()

            # Clear the statistics.
            self.stats.clear()
        # _end_if_

        # If options is not given, set the
        # parameters of the original paper.
        if options is None:
            # Default values of the simpler version.
            options = {"alpha": 0.5, "beta": 0.5}
        else:
            # Sanity check.
            for key in {"alpha", "beta"}:
                # Make sure the right keys exist.
                if key not in options:
                    raise ValueError(f"{self.__class__.__name__}: "
                                     f"Option '{key}' is missing. ")
            # _end_for_
        # _end_if_

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
        # _end_for_

        # Display an information message.
        print(f"Final f_optimal = {f_opt:.4f}")
    # _end_def_

# _end_class_
