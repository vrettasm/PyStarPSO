from math import isclose

import numpy as np

from ppso.auxiliary.utilities import time_it
from ppso.engines.generic_pso import GenericPSO

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

    def __init__(self, **kwargs):
        """
        Default constructor of StandardPSO object.
        """

        # Call the super constructor with the input parameters.
        super().__init__(**kwargs)

        # Number of particles.
        self.n_row = len(self.swarm.population)

        # Size (length) of particle.
        self.n_col = len(self.swarm.population[0])
    # _end_def_

    def update_positions(self, options: dict) -> None:
        """
        Updates the positions of the particles in the swarm.

        :param options: dictionary with options for the update equations.

        :return: None.
        """

        # Get the shape of the velocity array.
        arr_shape = (self.n_row, self.n_col)

        # Get the 'alpha' parameter.
        c_alpha = options.get("alpha")

        # Get the 'beta' parameter.
        c_beta = options.get("beta")

        # Get the GLOBAL best particle position.
        global_best_position = self.swarm.best_particle().position

        # Temporary 'velocity-like' parameters.
        tmp_velocities = (c_beta*global_best_position +
                          GenericPSO.rng_PSO.normal(0, c_alpha,
                                                    size=arr_shape))
        # Update all particle positions.
        for particle, velocity in zip(self._swarm.population, tmp_velocities):
            # Ensure the particle stays within bounds.
            particle.position = np.clip((1.0 - c_beta)*particle.position + velocity,
                                        self._lower_bound, self._upper_bound)
        # _end_for_
    # _end_def_

    @time_it
    def run(self, max_it: int = 100, f_tol: float = 1.0e-8, options: dict = None,
            parallel: bool = False, reset_swarm: bool = False, verbose: bool = False) -> None:
        """
        Main method of the StandardPSO class, that implements the optimization routine.

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
            self.generate_uniform_positions()
        # _end_if_

        # If options is not given set the
        # parameters of the original paper.
        if options is None:
            # Default values of the simpler version.
            options = {"alpha": 0.5, "beta": 0.5}
        else:
            # Make sure the right keys exist.
            for key in {"alpha", "beta"}:
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
