from typing import Any
from numbers import Number
from functools import cache

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

    @staticmethod
    def upd_float(**kwargs):
        # Extract the required values for the update.
        x_pos, v_new, lower_bound, upper_bound = kwargs

        # Ensure the position stays within bounds.
        return np.clip(x_pos + v_new, lower_bound, upper_bound)
    # _end_def_

    @staticmethod
    def upd_integer(**kwargs):
        # Extract the required values for the update.
        x_pos, v_new, lower_bound, upper_bound = kwargs

        # Round the new position and convert it to type int.
        new_position = np.rint(x_pos + v_new).astype(int)

        # Ensure the position stays within bounds.
        return np.clip(new_position, lower_bound, upper_bound)
    # _end_def_

    @classmethod
    def upd_binary(cls, **kwargs):
        # Extract the required value for the update.
        v_new = kwargs["v_new"]

        # Draw a random value in U(0, 1).
        r_uniform = cls.rng.uniform()

        # Compute the logistic function value.
        threshold = 1.0 / (1.0 + np.exp(-v_new))

        # Assign the binary value.
        return 1 if threshold > r_uniform else 0
    # _end_def_

    @classmethod
    def upd_categorical(cls, **kwargs):
        # Extract the required values for the update.
        x_pos = kwargs["x_new"]
        v_new = kwargs["v_new"]

        # Ensure the vector stays within limits.
        x_new = np.clip(x_pos + v_new, 0.0, 1.0)

        # Ensure there will be at least one
        # element with positive probability.
        if np.allclose(x_new, 0.0):
            x_new[DataBlock.rng.integers(len(x_new))] = 1.0
        # _end_if_

        # Normalize (to account for probabilities).
        return x_new / np.sum(x_new, dtype=float)
    # _end_def_

    @classmethod
    @cache
    def get_method_dict(cls):
        """
        Initialize a dictionary with method names as keys
        and method references as values.

        :return:
        """
        return {"float": DataBlock.upd_float,
                "binary": DataBlock.upd_binary,
                "integer": DataBlock.upd_integer,
                "categorical": DataBlock.upd_categorical
                }
    # _end_def_

    def _call_update(self, **kwargs):
        # Call the method based on the name provided
        method_dict = DataBlock.get_method_dict()

        # Return the outcome of the correct method.
        return method_dict[self._kind](**kwargs)
    # _end_def_

    def upd_position(self, *kwargs):
        self._position = DataBlock._call_update(*kwargs)
    # _end_def_

# _end_class_
