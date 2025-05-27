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
    # _end_def_

    @time_it
    def run(self, iterations: int = 1000, f_tol: float = None,
            parallel: bool = False, reset_swarm: bool = False,
            verbose: bool = False) -> None:

        # Check if resetting the swarm is required.
        if reset_swarm:
            self.generate_random_positions(self.velocity_min,
                                           self.velocity_max)
        # _end_if_

        # Get the function values before optimisation.
        fun_values_0, _ = self.evaluate_function(self.swarm, parallel)
    # _end_def_

# _end_def_
