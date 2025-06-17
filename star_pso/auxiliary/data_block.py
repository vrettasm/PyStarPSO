from typing import Any
from numbers import Number
from functools import cache
from copy import copy, deepcopy

from numpy import exp as np_exp
from numpy import sum as np_sum
from numpy import clip as np_clip
from numpy import rint as np_rint
from numpy import ones as np_ones
from numpy import isclose as np_isclose
from numpy import isscalar as np_isscalar

from numpy.typing import ArrayLike
from numpy.random import (default_rng, Generator)

from star_pso.auxiliary.utilities import (BlockType, my_clip)


class DataBlock(object):
    """
    Description:

       This is the main class that encodes the data of a single particle variable.
       The class encapsulates not only the data (position and best position), but
       also the way that this data can be updated using a specific type dependent
       functions.
    """

    # Make a random number generator (Class variable).
    rng: Generator = default_rng()

    # Object variables.
    __slots__ = ("_btype", "_valid_set", "_position", "_best_position",
                 "_lower_bound", "_upper_bound")

    def __init__(self,
                 position: Any,
                 btype: BlockType,
                 valid_set: list | tuple = None,
                 lower_bound: Number = None,
                 upper_bound: Number = None):
        """
        Default initializer for the DataBlock class.

        :param btype: the type of the data block (e.g. FLOAT, INTEGER, etc.).

        :param position: initial position (i.e. initial value in the block).

        :param valid_set: the set of values in the case of CATEGORICAL type.

        :param lower_bound: the lower bound in the search space.

        :param upper_bound: the upper bound in the search space.
        """

        # Assign the data block type.
        if isinstance(btype, BlockType):
            self._btype = btype
        else:
            raise TypeError(f"{self.__class__.__name__}: Unknown Block Type.")
        # _end_if_

        # Copy the initial position.
        self._position = copy(position)
        self._best_position = copy(position)

        # Get the lower and upper bounds.
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

        # Get the valid set (categorical variables).
        self._valid_set = valid_set
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
    def valid_set(self) -> list | tuple:
        """
        Accessor (getter) method for the valid set.

        :return: The set of valid values of a CATEGORICAL block.
        """
        return self._valid_set
    # _end_def_

    @staticmethod
    def upd_float(**kwargs) -> float:
        """
        It is used to update the positions of continuous
        'float' data blocks.

        :param kwargs: contains the parameters for the
        update equation.

        :return: a new (float) position.
        """
        # Extract the required values for the update.
        x_old = kwargs["x_old"]
        v_new = kwargs["v_new"]

        # Ensure the new position stays within bounds.
        return my_clip(x_old + v_new,
                       kwargs["lower_bound"],
                       kwargs["upper_bound"])
    # _end_def_

    @staticmethod
    def upd_integer(**kwargs) -> int:
        """
        It is used to update the positions of discrete
        'int' data blocks.

        :param kwargs: contains the parameters for the
        update equations.

        :return: a new (int) position.
        """
        # Extract the required values for the update.
        x_old = kwargs["x_old"]
        v_new = kwargs["v_new"]

        # Round the new position and convert it to type int.
        x_new = np_rint(x_old + v_new).astype(int)

        # Ensure the new position stays within bounds.
        return my_clip(x_new,
                       kwargs["lower_bound"],
                       kwargs["upper_bound"])
    # _end_def_

    @classmethod
    def upd_binary(cls, **kwargs) -> int:
        """
        It is used to update the positions of discrete
        'binary' data blocks.

        :param kwargs: contains the parameters for the
        update equations.

        :return: a new (binary) position.
        """
        # Extract the required value for the update.
        v_new = kwargs["v_new"]

        # Draw a random value in (0, 1).
        random_01 = cls.rng.random()

        # Compute the sigmoid function value.
        threshold = 1.0 / (1.0 + np_exp(-v_new))

        # Assign the binary value.
        return 1 if threshold > random_01 else 0
    # _end_def_

    @classmethod
    def upd_categorical(cls, **kwargs) -> ArrayLike:
        """
        It is used to update the positions of discrete
        'categorical' data blocks.

        :param kwargs: contains the parameters for the
        update equations.

        :return: a new array like with probabilities.
        """
        # Extract the required values for the update.
        x_old = kwargs["x_old"]
        v_new = kwargs["v_new"]

        # Ensure the velocities are within limits.
        np_clip(v_new, -0.5, +0.5, out=v_new)

        # Ensure the vector stays within limits.
        x_new = np_clip(x_old + v_new, 0.0, 1.0)

        # Ensure there will be at least one
        # element with positive probability.
        if all(np_isclose(x_new, 0.0)):
            x_new[cls.rng.integers(len(x_new))] = 1.0
        # _end_if_

        # Normalize (to account for probabilities).
        return x_new / np_sum(x_new, dtype=float)
    # _end_def_

    @staticmethod
    @cache
    def get_update_method() -> dict:
        """
        Create a dictionary with method names as keys
        and their corresponding update methods as values.

        :return: a (cached) dictionary with functions
        that correspond to the correct block types.
        """
        return {BlockType.FLOAT: DataBlock.upd_float,
                BlockType.BINARY: DataBlock.upd_binary,
                BlockType.INTEGER: DataBlock.upd_integer,
                BlockType.CATEGORICAL: DataBlock.upd_categorical}
    # _end_def_

    @classmethod
    def init_float(cls, **kwargs) -> float:
        """
        It is used to initialize randomly the positions
        of a continuous 'float' data blocks.

        :param kwargs: contains the required parameters.

        :return: a new random (float) position.
        """
        # Ensure the random position stays within bounds.
        return cls.rng.uniform(kwargs["lower_bound"],
                               kwargs["upper_bound"])
    # _end_def_

    @classmethod
    def init_integer(cls, **kwargs) -> int:
        """
        It is used to initialize randomly the positions
        of a discrete 'int' data blocks.

        :param kwargs: contains the required parameters.

        :return: a new random (integer) position.
        """
        # Ensure the random position stays within bounds.
        return cls.rng.integers(kwargs["lower_bound"],
                                kwargs["upper_bound"],
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
        # Ensure the random position stays within bounds.
        return cls.rng.integers(low=0, high=1,
                                size=kwargs["n_vars"],
                                endpoint=True)
    # _end_def_

    @staticmethod
    def init_categorical(**kwargs) -> int:
        """
        It is used to initialize randomly the positions
        of a discrete 'categorical' data blocks.

        :param kwargs: contains the required parameters.

        :return: a new random (integer) position.
        """
        # Get the number of variables.
        n_vars = kwargs["n_vars"]

        # Set the variables uniformly.
        return np_ones(n_vars)/n_vars
    # _end_def_

    @staticmethod
    @cache
    def get_init_method() -> dict:
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
                BlockType.CATEGORICAL: DataBlock.init_categorical}
    # _end_def_

    def reset_position(self) -> None:
        """
        This method provides the public interface of the
        reset of new positions for all types of data blocks.

        :return: None.
        """
        # Call the method based on the name provided.
        method_dict = DataBlock.get_init_method()

        # Differentiate between scalar and vector data block.
        n_vars = 1 if np_isscalar(self._position) else len(self._position)

        # Assign the function value to the new position.
        self._position = method_dict[self._btype](n_vars=n_vars,
                                                  lower_bound=self._lower_bound,
                                                  upper_bound=self._upper_bound)
    # _end_def_

    @property
    def position(self) -> Any:
        """
        Accessor (getter) of the data block's position.

        :return: the position value of the data block.
        """
        return self._position
    # _end_def_

    @position.setter
    def position(self, v_new: Number) -> None:
        """
        This method provides the public interface for setting
        the new position calculation for all types of data blocks.

        :param v_new: new velocity value.

        :return: None.
        """
        # Call the method based on the name provided.
        method_dict = DataBlock.get_update_method()

        # Assign the function value to the new position.
        self._position = method_dict[self._btype](v_new=v_new,
                                                  x_old=self._position,
                                                  lower_bound=self._lower_bound,
                                                  upper_bound=self._upper_bound)
    # _end_def_

    @property
    def best_position(self) -> Any:
        """
        Accessor (getter) of the data block's
        best position.

        :return: the best recorded position
        value of the data block.
        """
        return self._best_position
    # _end_def_

    @best_position.setter
    def best_position(self, new_value) -> None:
        """
        This method provides the public interface for
        setting the new best position of data block.

        :param new_value: new best position value.

        :return: None.
        """
        self._best_position = copy(new_value)
    # _end_def_

    @property
    def btype(self) -> BlockType:
        """
        Accessor (getter) of the data block's type.

        :return: the data block type.
        """
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
        setattr(new_object, "_position", deepcopy(self._position))

        # Shallow copy the best position vector.
        setattr(new_object, "_best_position", deepcopy(self._best_position))

        # Simple copy the lower/upper bounds (float).
        setattr(new_object, "_lower_bound", self._lower_bound)
        setattr(new_object, "_upper_bound", self._upper_bound)

        # Simple copy the block type value (enum).
        setattr(new_object, "_btype", self._btype)

        # Return an identical particle.
        return new_object
    # _end_def_

# _end_class_
