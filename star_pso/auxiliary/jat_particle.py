from math import inf
from copy import deepcopy
from dataclasses import dataclass, field

from star_pso.auxiliary.data_block import DataBlock

# Public interface.
__all__ = ["JatParticle"]


@dataclass(init=True, repr=True)
class JatParticle(object):
    """
    Description:

        Implements a dataclass for the 'jack of all trades' particle.
        This is class maintains a container (list of data blocks) and
        most information is held in its data blocks.
    """

    # Define the particle as a list of DataBlocks.
    _container: list = field(default_factory=list[DataBlock])

    # Initially the function value is set to -Inf.
    _value: float = -inf

    # Initialize the best (historical) value to -Inf.
    _best_value: float = -inf

    @property
    def container(self) -> list[DataBlock]:
        """
        Accessor of the container list of the particle.

        :return: the list (of data blocks) of the particle.
        """
        return self._container
    # _end_def_

    @property
    def size(self) -> int:
        """
        Returns the size (length) of the particle.

        :return: (int) the length of the particle.
        """
        return len(self._container)
    # _end_def_

    @property
    def position(self) -> list:
        """
        Accessor of the particle's position.

        :return: a list with each data block position.
        """
        return [blk.position for blk in self._container]
    # _end_def_

    @position.setter
    def position(self, v_new) -> None:
        """
        Sets a new value to the particle's position
        by updating each data block in the container.

        :param v_new: new velocity values.

        :return: None.
        """
        for blk, v in zip(self._container, v_new):
            # This calls internally the 'right'
            # method to update each block type.
            blk.position = v
    # _end_def_

    @property
    def best_position(self) -> list:
        """
        Accessor of the particle's best position.

        :return: a list with each data block best position.
        """
        return [blk.best_position for blk in self._container]
    # _end_def_

    @best_position.setter
    def best_position(self, new_vector: list) -> None:
        """
        Updates the 'best position' in the particle object.

        :param new_vector: (list) new best position vector.

        :return: None.
        """
        for blk, new_best in zip(self._container, new_vector):
            blk.best_position = new_best
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
        Updates the best function value in the particle.

        :param new_value: (float) new best function value.

        :return: None.
        """
        self._value = new_value
    # _end_def_

    @property
    def best_value(self) -> float:
        """
        Accessor of the best function value.

        :return: (float) best function value.
        """
        return self._best_value
    # _end_def_

    @best_value.setter
    def best_value(self, new_value: float) -> None:
        """
        Updates the best function value in the particle.

        :param new_value: (float) new best function value.

        :return: None.
        """
        self._best_value = new_value
    # _end_def_

    def reset_position(self) -> None:
        """
        Iterate through all the data blocks in
        the container and reset their positions.

        :return: None.
        """
        for blk in self._container:
            blk.reset_position()
    # _end_def_

    def __getitem__(self, index: int) -> DataBlock:
        """
        Get the DataBlock item at position 'index'.

        :param index: (int) the position that we want to return.

        :return: the reference to the object in position index.
        """
        return self._container[index]
    # _end_def_

    def __setitem__(self, index: int, item: DataBlock) -> None:
        """
        Set the DataBlock 'item' at position 'index'.

        :param index: (int) the position that we want to access.

        :param item: (DataBlock) object we want to assign in the
        particle.

        :return: None.
        """
        self._container[index] = item
    # _end_def_

    def __len__(self) -> int:
        """
        Accessor of the total length of the particle.

        :return: the length (int) of the particle.
        """
        return len(self._container)
    # _end_def_

    def __eq__(self, other) -> bool:
        """
        Compares the jat_particle of self with the other and
        returns True if they are identical, otherwise False.

        :param other: jat_particle to compare.

        :return: True if their containers have the same data
        blocks.
        """

        # Make sure both objects are of the same type.
        if isinstance(other, JatParticle):
            # Compare directly their two containers.
            return self._container == other.container
        # _end_if_
        return NotImplemented
    # _end_def_

    def __deepcopy__(self, memo):
        """
        This custom method overrides the default deepcopy method.

        :param memo: dictionary of objects already copied during
        the current copying pass.

        :return: a new identical "clone" of the self object.
        """

        # Create a new instance.
        new_object = JatParticle.__new__(JatParticle)

        # Don't copy self reference.
        memo[id(self)] = new_object

        # Deep copy the container list.
        setattr(new_object, "_container", deepcopy(self._container, memo))

        # Simply copy the value (float).
        setattr(new_object, "_value", self._value)

        # Simply copy the best value (float).
        setattr(new_object, "_best_value", self._best_value)

        # Return an identical particle.
        return new_object
    # _end_def_

    def __contains__(self, item: DataBlock) -> bool:
        """
        Check for membership.

        :param item: an input DataBlock that we want to check.

        :return: true if the 'item' belongs in the container.
        """
        return item in self._container
    # _end_if_

    def __str__(self) -> str:
        """
        Override to print a readable string presentation
        of the JatParticle object.

        :return: a string representation of a JatParticle.
        """
        return f"{self.__class__.__name__}: "\
               f"(Position, Value) = ({self.position, self.value})."
    # _end_def_

# _end_class_
