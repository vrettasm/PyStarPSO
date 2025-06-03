from math import isclose

import numpy as np
from numpy.typing import ArrayLike

from ppso.auxiliary.utilities import time_it
from ppso.engines.generic_pso import GenericPSO

# Public interface.
__all__ = ["StandardPSO"]


class StandardPSO(GenericPSO):
    """
    Description:

        TBD

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

        # Generate initial particle velocities.
        self._velocities = GenericPSO.rng_PSO.uniform(-1.0, +1.0,
                                                      size=(self.n_row, self.n_col))
    # _end_def_

    def update_velocities(self, w: float = 0.5, c1: float = 1.5, c2: float = 1.5) -> ArrayLike:

        # Pre-sample the coefficients.
        R1 = GenericPSO.rng_PSO.uniform(0, c1, size=(self.n_row, self.n_col))
        R2 = GenericPSO.rng_PSO.uniform(0, c2, size=(self.n_row, self.n_col))

        # Get the global best particle position.
        g_best = self.swarm.best_particle().position

        for i, (r1, r2) in enumerate(zip(R1, R2)):
            # Get the current position of i-th the particle.
            x_i = self.swarm[i].position

            # Update the new velocity.
            self._velocities[i] = w * self._velocities[i] +\
                                  r1 * (self.swarm[i].best_position - x_i) +\
                                  r2 * (g_best - x_i)
        # _end_for_

    # _end_def_

    def update_positions(self, new_velocities: ArrayLike) -> None:
        """
        Updates the positions of the particles in the swarm.

        :param new_velocities: array-like object with the new
        velocities that will update the particle positions.

        :return: None.
        """

        # Update all particles positions.
        for particle, velocity in zip(self._swarm.population,
                                      new_velocities):
            # Ensure the particle stays within bounds.
            particle.position = np.clip(particle.position + velocity,
                                        self._lower_bound, self._upper_bound)

    # _end_def_

    @time_it
    def run(self, max_it: int = 1000, f_tol: float = None, parallel: bool = False,
            reset_swarm: bool = False, verbose: bool = False) -> None:

        # Check if resetting the swarm is required.
        if reset_swarm:
            self.generate_random_positions()
        # _end_if_

        # Get the function values before optimisation.
        fun_values_0, _ = self.evaluate_function(parallel)

        # Get the first optimal function value.
        f_opt = max(fun_values_0)

        # Display an information message.
        print(f"Initial f_optimal = {f_opt:.4f}")

        # Local variable to display information on the screen.
        # To avoid cluttering the screen we print info only 10
        # times regardless of the total number of iterations.
        its_time_to_print = (max_it // 10)

        # Repeat for 'max_it' times.
        for i in range(max_it):

            # Perform the update of velocity.
            self.update_velocities()

            # Update the positions in the swarm.
            self.update_positions(self._velocities)

            # Calculate the new function values.
            fun_values_i, found_solution = self.evaluate_function(parallel)

            # Compute the optimal function value.
            f_new = max(fun_values_i)

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
