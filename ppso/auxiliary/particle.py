from copy import deepcopy
from numpy.typing import ArrayLike
from numpy import array, array_equal

# Public interface.
__all__ = ["Particle"]


class Particle(object):
    """
    Description:
        Models the particle in the swarm of the PSO.
    """

    # Object variables.
    __slots__ = ("_position", "_velocity", "_value", "_best_position", "_best_value")

    def __init__(self,
                 initial_position: ArrayLike = None,
                 initial_velocity: ArrayLike = None) -> None:
        """
        Initialize the Particle object. Note that calling the initializer without
        arguments (i.e., Particle()), it will create an empty 'dummy' particle.

        :param initial_position: (ArrayLike) Initial position of the particle.

        :param initial_velocity: (ArrayLike) Initial velocity of the particle.
        """

        # Set the initial particle position to a vector.
        self._position = array(initial_position)

        # Set the initial particle velocity to a vector.
        self._velocity = array(initial_velocity)

        # Initially the best position is the initial vector.
        self._best_position = array(initial_position)

        # Initially the best (function) value is set to Inf.
        self._best_value = float("inf")

        # Initially the function value is set to Inf.
        self._value = float("inf")
    # _end_def_

    @property
    def size(self) -> int:
        """
        Returns the size (length) of the particle.

        :return: (int) the length of the particle.
        """
        return len(self._position)
    # _end_def_

    @property
    def position(self) -> ArrayLike:
        """
        Accessor of the positions in the particle.

        :return: the ArayLike (vector) of the positions.
        """
        return self._position
    # _end_def_

    @position.setter
    def position(self, new_vector: ArrayLike) -> None:
        """
        Updates the position in the particle object.

        :param new_vector: (ArrayLike) New position vector.

        :return: None.
        """
        self._position = new_vector
    # _end_def_

    @property
    def velocity(self) -> ArrayLike:
        """
        Returns the velocity of the particle object.

        :return: (ArrayLike) velocity vector.
        """
        return self._velocity
    # _end_def_

    @velocity.setter
    def velocity(self, new_vector: ArrayLike) -> None:
        """
        Updates the velocity in the particle object.

        :param new_vector: (ArrayLike) New position vector.

        :return: None.
        """
        self._velocity = new_vector
    # _end_def_

    @property
    def best_position(self) -> ArrayLike:
        """
        Returns the best (so far) position of the particle object.

        :return: (ArrayLike) position vector.
        """
        return self._best_position
    # _end_def_

    @best_position.setter
    def best_position(self, new_vector: ArrayLike) -> None:
        """
        Updates the best position (so far) in the particle object.

        :param new_vector: (ArrayLike) New best position vector.

        :return: None.
        """
        self._best_position = new_vector
    # _end_def_

    @property
    def best_value(self) -> float:
        """
        Accessor of the best function value recorded.

        :return: (float) best function value.
        """
        return self._best_value
    # _end_def_

    @best_value.setter
    def best_value(self, new_value: float) -> None:
        """
        Updates the best function value in the particle object.

        :param new_value: (float) New best function value.

        :return: None.
        """
        self._best_value = new_value
    # _end_def_

    @property
    def value(self) -> float:
        """
        Accessor of the current function value.

        :return: (float) function value.
        """
        return self._value
    # _end_def_

    @value.setter
    def value(self, new_value: float) -> None:
        """
        Updates the best function value in the particle object.

        :param new_value: (float) New best function value.

        :return: None.
        """
        self._value = new_value
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

        # Simply copy the value (float).
        setattr(new_object, "_value", self._value)

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

    def __len__(self) -> int:
        """
        Accessor of the total length of the particle.

        :return: the length (int) of the particle.
        """
        return len(self._position)
    # _end_def_

    def __eq__(self, other) -> bool:
        """
        Compares the 'self' particle, with the 'other' particle
        and returns True if they have the same position otherwise False.

        :param other: particle to compare.

        :return: True if the positions are identical else False.
        """

        # Make sure both objects are of the same type 'particle'.
        if isinstance(other, Particle):

            # Compare directly the two positions vectors.
            return array_equal(self._position, other.position)

        # _end_if_
        return False
    # _end_def_

    def __str__(self) -> str:
        """
        Override to print a readable string presentation
        of the Particle object.

        :return: a string representation of a Particle.
        """
        return f"{self.__class__.__name__}: "\
               f"(Position, Velocity) = ({self._position, self._velocity})."
    # _end_def_

# _end_class_
