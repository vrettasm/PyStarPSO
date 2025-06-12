from typing import Any
from numbers import Number

import numpy as np
from numpy.random import default_rng, Generator

class DataBlock(object):
    """
    Description:

       This is the main class that encodes the data of a single particle variable.
       The class encapsulates not only the data (position and velocity), but also
       the way that this data can be updated using a specific type dependent function.
    """

    # Make a random number generator.
    rng: Generator = default_rng()

    # Object variables.
    __slots__ = ("_position", "_velocity", "_lower_bound", "_upper_bound", "_kind")

    def __init__(self, position: Any,
                 lower_bound: Number,
                 upper_bound: Number,
                 kind: str):
        # ...
        self._position = position

        # ...
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

        # ...
        self._kind = kind
    # _end_def_

    def upd_float(self, v_new):
        """

        :param v_new:

        :return:
        """
        # Ensure the position stays within bounds.
        np.clip(self._position + v_new,
                self._lower_bound, self._upper_bound, out=self._position)
    # _end_def_

    def upd_integer(self, v_new):
        """

        :param v_new:

        :return:
        """

        # Round the new position and convert it to type int.
        new_position = np.rint(self._position + v_new).astype(int)

        # Ensure the position stays within bounds.
        np.clip(new_position, self._lower_bound, self._upper_bound,
                out=self._position)
    # _end_def_

    def upd_binary(self, v_new):
        """

        :param v_new:

        :return:
        """

        # Draw a random value in U(0, 1).
        r_uniform = DataBlock.rng.uniform()

        # Compute the logistic value.
        threshold = 1.0 / (1.0 + np.exp(-v_new))

        # Assign the binary value.
        self._position = 1 if threshold > r_uniform else 0
    # _end_def_

    def upd_categorical(self, v_new):
        """

        :param v_new:

        :return:
        """

        # Update the position.
        x_new = self._position + v_new

        # Ensure the vector stays within limits.
        np.clip(x_new, 0.0, 1.0, out=x_new)

        # Ensure there will be at least one
        # element with positive probability.
        if np.allclose(x_new, 0.0):
            x_new[DataBlock.rng.integers(len(x_new))] = 1.0
        # _end_if_

        # Normalize (to account for probabilities).
        self._position = x_new / np.sum(x_new, dtype=float)
    # _end_def_

# _end_class_
