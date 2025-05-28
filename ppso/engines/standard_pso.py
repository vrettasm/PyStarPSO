from math import isclose

from numpy.typing import ArrayLike

from ppso.engines.generic_pso import GenericPSO, time_it

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
        local_N = len(self.swarm.population)

        # Size of particle.
        local_D = len(self.swarm.population[0])

        # Generate initial particle velocities.
        self._velocities = GenericPSO.rng_PSO.uniform(-1.0, +1.0,
                                                      size=(local_N, local_D))
    # _end_def_

    def update_velocities(self, w: float = 0.5, c1: float = 1.5, c2: float = 1.5) -> ArrayLike:

        # Get the size of the velocities matrix.
        size_N, size_D = self._velocities.shape

        # Pre-sample the coefficients.
        R1 = GenericPSO.rng_PSO.uniform(0, 1, size=(size_N, size_D))
        R2 = GenericPSO.rng_PSO.uniform(0, 1, size=(size_N, size_D))

        # Get the Global best particle.
        g_best = self.swarm.best_particle().position

        for i, (r1, r2) in enumerate(zip(R1, R2)):
            # Get the current position of the particle.
            position_x = self.swarm[i].position

            # Update the new velocity.
            self._velocities[i] = (w * self._velocities[i] +
                                   c1 * r1 * (self.swarm[i].best_position - position_x) +
                                   c2 * r2 * (g_best - position_x))
        # _end_for_

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
