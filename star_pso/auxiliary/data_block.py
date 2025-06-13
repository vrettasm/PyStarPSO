from copy import copy
from typing import Any
from numbers import Number
from functools import cache

from numpy import exp as np_exp
from numpy import sum as np_sum
from numpy import clip as np_clip
from numpy import rint as np_rint
from numpy import ones as np_ones
from numpy import allclose as np_allclose

from numpy.typing import ArrayLike
from numpy.random import default_rng, Generator

from star_pso.auxiliary.utilities import BlockType


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
    __slots__ = ("_position", "_best_position", "_velocity",
                 "_lower_bound", "_upper_bound", "_btype")

    def __init__(self,
                 position: Any,
                 btype: BlockType,
                 lower_bound: Number = None,
                 upper_bound: Number = None):
        # ...
        self._position = copy(position)
        self._best_position = copy(position)

        # ...
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

        # ...
        self._btype = btype
    # _end_def_

    @staticmethod
    def upd_float(**kwargs) -> float:
        """
        It is used to update the positions of continuous 'float' data blocks.

        :param kwargs: contains the parameters for the updates equations.

        :return: a new (float) position.
        """
        # Extract the required values for the update.
        x_pos = kwargs["x_pos"]
        v_new = kwargs["v_new"]

        lower_bound = kwargs["lower_bound"]
        upper_bound = kwargs["upper_bound"]

        # Ensure the position stays within bounds.
        return np_clip(x_pos + v_new, lower_bound, upper_bound)
    # _end_def_

    @staticmethod
    def upd_integer(**kwargs) -> int:
        """
        It is used to update the positions of discrete 'int' data blocks.

        :param kwargs: contains the parameters for the updates equations.

        :return: a new (int) position.
        """
        # Extract the required values for the update.
        x_pos = kwargs["x_pos"]
        v_new = kwargs["v_new"]

        lower_bound = kwargs["lower_bound"]
        upper_bound = kwargs["upper_bound"]

        # Round the new position and convert it to type int.
        new_position = np_rint(x_pos + v_new).astype(int)

        # Ensure the position stays within bounds.
        return np_clip(new_position, lower_bound, upper_bound)
    # _end_def_

    @classmethod
    def upd_binary(cls, **kwargs) -> int:
        """
        It is used to update the positions of discrete 'binary' data blocks.

        :param kwargs: contains the parameters for the updates equations.

        :return: a new (binary) position.
        """
        # Extract the required value for the update.
        v_new = kwargs["v_new"]

        # Draw a random value in U(0, 1).
        r_uniform = cls.rng.uniform()

        # Compute the sigmoid function value.
        threshold = 1.0 / (1.0 + np_exp(-v_new))

        # Assign the binary value.
        return 1 if threshold > r_uniform else 0
    # _end_def_

    @classmethod
    def upd_categorical(cls, **kwargs) -> ArrayLike:
        # Extract the required values for the update.
        x_pos = kwargs["x_pos"]
        v_new = kwargs["v_new"]

        # Ensure the vector stays within limits.
        x_new = np_clip(x_pos + v_new, 0.0, 1.0)

        # Ensure there will be at least one
        # element with positive probability.
        if np_allclose(x_new, 0.0):
            x_new[DataBlock.rng.integers(len(x_new))] = 1.0
        # _end_if_

        # Normalize (to account for probabilities).
        return x_new / np_sum(x_new, dtype=float)
    # _end_def_

    @classmethod
    @cache
    def get_update_method(cls):
        """
        Create a dictionary with method names as keys
        and their corresponding update methods as values.

        :return: a (cached) dictionary with functions
        that correspond to the correct block types.
        """
        return {BlockType.FLOAT: DataBlock.upd_float,
                BlockType.BINARY: DataBlock.upd_binary,
                BlockType.INTEGER: DataBlock.upd_integer,
                BlockType.CATEGORICAL: DataBlock.upd_categorical
                }
    # _end_def_

    def new_position(self, **kwargs) -> None:
        """
        This method provides the public interface of the new position
        calculation for all types of data blocks.

        :param kwargs: the input parameters to the update methods.

        :return: None.
        """
        # Call the method based on the name provided
        method_dict = DataBlock.get_update_method()

        # Assign the function value to the new position.
        self._position = method_dict[self._btype](**kwargs)
    # _end_def_

    @classmethod
    def init_float(cls, **kwargs) -> float:
        """
        It is used to initialize randomly the positions
        of a continuous 'float' data blocks.

        :param kwargs: contains the required parameters.

        :return: a new random (float) position.
        """
        lower_bound = kwargs["lower_bound"]
        upper_bound = kwargs["upper_bound"]

        # Ensure the random position stays within bounds.
        return DataBlock.rng.uniform(lower_bound,
                                     upper_bound)
    # _end_def_

    @classmethod
    def init_integer(cls, **kwargs) -> int:
        """
        It is used to initialize randomly the positions
        of a discrete 'int' data blocks.

        :param kwargs: contains the required parameters.

        :return: a new random (integer) position.
        """
        lower_bound = kwargs["lower_bound"]
        upper_bound = kwargs["upper_bound"]

        # Ensure the random position stays within bounds.
        return DataBlock.rng.integers(lower_bound,
                                      upper_bound,
                                      endpoint=True)
    # _end_def_

    @classmethod
    def init_binary(cls, **kwargs) -> int:
        """
        It is used to initialize randomly the positions
        of a discrete 'binary' data blocks.

        :param kwargs: contains the required parameters.

        :return: a new random (integer) position.
        """
        # Get the number of variables.
        num_var = kwargs["num_var"]

        # Ensure the random position stays within bounds.
        return DataBlock.rng.integers(0, 1, size=num_var, endpoint=True)
    # _end_def_

    @classmethod
    def init_categorical(cls, **kwargs) -> int:
        """
        It is used to initialize randomly the positions
        of a discrete 'categorical' data blocks.

        :param kwargs: contains the required parameters.

        :return: a new random (integer) position.
        """
        # Get the number of variables.
        num_var = kwargs["num_var"]

        # Set the variables uniformly.
        return np_ones(num_var)/num_var
    # _end_def_

    @classmethod
    @cache
    def get_init_method(cls):
        """
        Create a dictionary with method names as keys
        and their corresponding initialization methods
        as values.

        :return: a (cached) dictionary with functions
        that correspond to the correct block types.
        """
        return {BlockType.FLOAT: DataBlock.init_float,
                BlockType.BINARY: DataBlock.init_binary,
                BlockType.INTEGER: DataBlock.init_integer,
                BlockType.CATEGORICAL: DataBlock.init_categorical
                }
    # _end_def_

    def reset_position(self) -> None:
        """
        This method provides the public interface of the
        reset of new positions for all types of data blocks.

        :return: None.
        """
        # Call the method based on the name provided
        method_dict = DataBlock.get_init_method()

        # Assign the function value to the new position.
        self._position = method_dict[self._btype](lower_bound=self._lower_bound,
                                                  upper_bound=self._upper_bound,
                                                  num_var=len(self._position))
    # _end_def_

    @property
    def position(self):
        return self._position
    # _end_def_

    @property
    def best_position(self):
        return self._best_position
    # _end_def_

    @best_position.setter
    def best_position(self, new_value):
        self._best_position = copy(new_value)
    # _end_def_

    @property
    def block_type(self):
        return self._btype
    # _end_def_

    def __deepcopy__(self, memo):
        """
        This custom method overrides the default deepcopy method.

        :param memo: Dictionary of objects already copied during
        the current copying pass.

        :return: a new identical "clone" of the self object.
        """

        # Create a new instance.
        new_object = DataBlock.__new__(DataBlock)

        # Don't copy self reference.
        memo[id(self)] = new_object

        # Shallow copy the position vector.
        setattr(new_object, "_position", copy(self._position))

        # Shallow copy the best position vector.
        setattr(new_object, "_best_position", copy(self._best_position))

        # Simple copy the lower/upper bounds (float).
        setattr(new_object, "_lower_bound", self._lower_bound)
        setattr(new_object, "_upper_bound", self._upper_bound)

        # Simple copy the block type value (enum).
        setattr(new_object, "_btype", self._btype)

        # Return an identical particle.
        return new_object
    # _end_def_

# _end_class_
