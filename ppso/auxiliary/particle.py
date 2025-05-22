import numpy as np
from copy import deepcopy

class Particle(object):
    """
    Description:
        Models the particle in the swarm of the PSO.
    """

    # Object variables.
    __slots__ = ("_position", "_velocity", "_best_position", "_best_value")

    def __init__(self,
                 initial_position: np.typing.ArrayLike,
                 initial_velocity: np.typing.ArrayLike) -> None:
        """
        Initialize the Particle object.

        :param initial_position: (ArrayLike) Initial position of the particle.

        :param initial_velocity: (ArrayLike) Initial velocity of the particle.
        """

        # Set the particle position to an initial vector.
        self._position = initial_position

        # Set the particle velocity to an initial vector.
        self._velocity = initial_velocity

        # Initially the best position is the initial vector.
        self._best_position = initial_position

        # Initially the best (function) value is set to Inf.
        self._best_value = float("inf")
    # _end_def_

    @property
    def position(self) -> np.typing.ArrayLike:
        return self._position
    # _end_def_

    @position.setter
    def position(self, new_vector: np.typing.ArrayLike) -> None:
        """
        Updates the position in the particle object.

        :param new_vector: (ArrayLike) New position vector.

        :return: None.
        """
        self._position = new_vector
    # _end_def_

    @property
    def velocity(self) -> np.typing.ArrayLike:
        return self._velocity
    # _end_def_

    @velocity.setter
    def velocity(self, new_vector: np.typing.ArrayLike) -> None:
        """
        Updates the velocity in the particle object.

        :param new_vector: (ArrayLike) New position vector.

        :return: None.
        """
        self._velocity = new_vector
    # _end_def_

    @property
    def best_position(self) -> np.typing.ArrayLike:
        return self._best_position
    # _end_def_

    @best_position.setter
    def best_position(self, new_vector: np.typing.ArrayLike) -> None:
        """
        Updates the best position (so far) in the particle object.

        :param new_vector: (ArrayLike) New best position vector.

        :return: None.
        """
        self._best_position = new_vector
    # _end_def_

    @property
    def best_value(self) -> float:
        return self._best_value
    # _end_def_

    @best_value.setter
    def best_value(self, new_value: float) -> float:
        """
        Updates the best function value in the particle object.

        :param new_value: (float) New best function value.

        :return: None.
        """
        self._best_value = new_value
    # _end_def_

    def __deepcopy__(self, memo):
        """
        This custom method overrides the default deepcopy method
        and is used when we call the "clone" method of the class.

        :param memo: Dictionary of objects already copied during
        the current copying pass.

        :return: a new identical "clone" of the self object.
        """

        # Create a new instance.
        new_object = Particle.__new__(Particle)

        # Don't copy self reference.
        memo[id(self)] = new_object

        # Deep copy the position vector.
        setattr(new_object, "_position", deepcopy(self._position, memo))

        # Deep copy the velocity vector.
        setattr(new_object, "_velocity", deepcopy(self._velocity, memo))

        # Deep copy the best position vector.
        setattr(new_object, "_best_position", deepcopy(self._best_position, memo))

        # Simply copy the best value (float).
        setattr(new_object, "_best_value", self._best_value)

        # Return an identical particle.
        return new_object
    # _end_def_

    def clone(self):
        """
        Makes a duplicate of the self object.

        :return: a "deep-copy" of the object.
        """
        return deepcopy(self)
    # _end_def_

    def __str__(self) -> str:
        return f"{self.__class__.__name__}: "\
               f"(Position, Velocity) = ({self._position, self._velocity})."
    # _end_def_

# _end_class_
