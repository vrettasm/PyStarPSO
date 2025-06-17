from math import isclose

import numpy as np
from numpy.typing import ArrayLike

from star_pso.auxiliary.utilities import time_it
from star_pso.engines.generic_pso import GenericPSO

# Public interface.
__all__ = ["BinaryPSO"]


class BinaryPSO(GenericPSO):
    """
    Description:

    This class implements the discrete binary particle swarm optimization variant
    as described in:

    Kennedy, J., and R. C. Eberhart 1997. “A Discrete Binary Version of the Particle
    Swarm Algorithm.” IEEE International conference on systems, man, and cybernetics,
    1997. Computational cybernetics and simulation, Vol. 5, Orlando, FL, October 12–15,
    pp: 4104–4108.

    """

    # Object variables (specific for the BinaryPSO).
    __slots__ = ("_velocities",)

    def __init__(self, v_min: ArrayLike, v_max: ArrayLike, **kwargs):
        """
        Default initializer of the BinaryPSO class.

        :param v_min: lower velocity bound.

        :param v_max: upper velocity bound.
        """

        # Call the super initializer with the input parameters.
        super().__init__(lower_bound=v_min, upper_bound=v_max, **kwargs)

        # Generate initial particle velocities.
        self._velocities = GenericPSO.rng.uniform(-1.0, +1.0,
                                                  size=(self.n_rows, self.n_cols))
    # _end_def_

    def update_velocities(self, options: dict) -> None:
        """
        Performs the update on the velocity equations according to the
        original PSO paper by "Kennedy, J. and Eberhart, R. (1995)".

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
        R1 = GenericPSO.rng.uniform(0, c1, size=arr_shape)
        R2 = GenericPSO.rng.uniform(0, c2, size=arr_shape)

        # Get the GLOBAL best particle position.
        if fipso:
            # In the fully informed case we take the average of all the best positions.
            g_best = np.mean([p.best_position for p in self.swarm.population], axis=0).round()
        else:
            g_best = self.swarm.best_particle().position
        # _end_if_

        for i, (r1, r2) in enumerate(zip(R1, R2)):
            # Get the current position of i-th the particle.
            x_i = self.swarm[i].position

            # Update the new velocity.
            self._velocities[i] = w * self._velocities[i] +\
                r1 * (self.swarm[i].best_position - x_i) +\
                r2 * (g_best - x_i)
        # _end_for_

        # We clip the velocities in [V_min, V_max].
        np.clip(self._velocities, self._lower_bound, self._upper_bound,
                out=self._velocities)
    # _end_def_

    def update_positions(self, options: dict) -> None:
        """
        Updates the positions of the particles in the swarm.

        :param options: dictionary with options for the update
        equations, i.e. ('w', 'c1', 'c2', 'fipso').

        :return: None.
        """

        # Update the velocity equations.
        self.update_velocities(options)

        # Generate random vectors in U(0, 1).
        r_uniform = GenericPSO.rng.uniform(0, 1,
                                           size=(self.n_rows, self.n_cols))
        # Create a matrix with zeros.
        new_positions = np.zeros_like(r_uniform, dtype=int)

        # Compute the logistic values.
        s_arr = 1.0 / (1.0 + np.exp(-self._velocities))

        # Where the logistic function values are
        # higher than the random value set to 1.
        new_positions[s_arr > r_uniform] = 1

        # Update all particle positions.
        for particle, x_new, in zip(self._swarm.population,
                                    new_positions):
            particle.position = x_new
    # _end_def_

    @time_it
    def run(self, max_it: int = 100, f_tol: float = None, options: dict = None,
            parallel: bool = False, reset_swarm: bool = False, verbose: bool = False) -> None:
        """
        Main method of the BinaryPSO class, that implements the optimization routine.

        :param max_it: (int) maximum number of iterations in the optimization loop.

        :param f_tol: (float) tolerance in the difference between the optimal function value
        of two consecutive iterations. It is used to determine the convergence of the swarm.
        If this value is None (default) the algorithm will terminate using the max_it value.

        :param options: dictionary with the update equations options ('w': inertia weight,
        'c1': cognitive coefficient, 'c2': social coefficient).

        :param parallel:(bool) Flag that enables parallel computation of the objective function.

        :param reset_swarm: if true it will reset the positions of the swarm to uniformly random
        respecting the boundaries of each space dimension.

        :param verbose: (bool) if 'True' it will display periodically information about the
        current optimal function values.

        :return: None.
        """

        # Check if resetting the swarm is required.
        if reset_swarm:
            # Reset particle velocities.
            self._velocities = GenericPSO.rng.uniform(-1.0, +1.0,
                                                      size=(self.n_rows, self.n_cols))
            # Generate random positions.
            self.generate_binary_positions()

            # Clear the statistics.
            self.stats.clear()
        # _end_if_

        # If options is not given, set the
        # parameters of the original paper.
        if options is None:
            # Default values of the simplified version.
            options = {"w": 1.0, "c1": 2.0, "c2": 2.0}
        else:
            # Sanity check.
            for key in {"w", "c1", "c2"}:
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
