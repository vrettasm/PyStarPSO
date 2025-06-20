import unittest

import numpy as np

from star_pso.auxiliary.utilities import BlockType
from star_pso.auxiliary.data_block import DataBlock
from star_pso.auxiliary.jat_particle import JatParticle


class TestJatParticle(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        print(">> TestJatParticle - START -")
        cls.rng = np.random.default_rng()
    # _end_def_

    @classmethod
    def tearDownClass(cls) -> None:
        print(">> TestJatParticle - FINISH -", end='\n\n')
    # _end_def_

    def test_equals(self):
        """
        Tests the __eq__ method of the JatParticle class.

        :return: None.
        """
        # Test variable set.
        var_set = ["a", "b", "c", "d"]

        # Get the size of the vector.
        D = len(var_set)

        # Particle 1.
        p1 = JatParticle([DataBlock(9.0,
                                    BlockType.FLOAT,
                                    lower_bound=-10.0,
                                    upper_bound=+10.0),
                          DataBlock(1,
                                    BlockType.INTEGER,
                                    lower_bound=-3,
                                    upper_bound=+3),
                          DataBlock(np.ones(D) / D,
                                    BlockType.CATEGORICAL,
                                    valid_set=var_set)])

        # Should be TRUE.
        self.assertTrue(p1 == p1)

        # Particle 2 (identical to p1).
        p2 = JatParticle([DataBlock(9.0,
                                    BlockType.FLOAT,
                                    lower_bound=-10.0,
                                    upper_bound=+10.0),
                          DataBlock(1,
                                    BlockType.INTEGER,
                                    lower_bound=-3,
                                    upper_bound=+3),
                          DataBlock(np.ones(D) / D,
                                    BlockType.CATEGORICAL,
                                    valid_set=var_set)])

        # Should be TRUE.
        self.assertTrue(p2 == p2)

        # Should be TRUE.
        self.assertTrue(p1 == p2)

        # Particle 3.
        p3 = JatParticle([DataBlock(5.0,
                                    BlockType.FLOAT,
                                    lower_bound=-10.0,
                                    upper_bound=+10.0),
                          DataBlock(1,
                                    BlockType.INTEGER,
                                    lower_bound=-3,
                                    upper_bound=+3),
                          DataBlock(np.ones(D) / D,
                                    BlockType.CATEGORICAL,
                                    valid_set=var_set)])

        # Should be TRUE.
        self.assertTrue(p3 == p3)

        # Should be FALSE.
        self.assertFalse(p1 == p3)
    # _end_def_

    def test_best_position(self):
        """
        Tests the best_position method. It should assign
        a copy of the input vector.

        :return: None.
        """

        # Test variable set.
        var_set = ["a", "b", "c", "d"]

        # Get the size of the vector.
        D = len(var_set)

        # Particle 1.
        p1 = JatParticle([DataBlock(9.0,
                                    BlockType.FLOAT,
                                    lower_bound=-10.0,
                                    upper_bound=+10.0),
                          DataBlock(1,
                                    BlockType.INTEGER,
                                    lower_bound=-3,
                                    upper_bound=+3),
                          DataBlock(np.ones(D) / D,
                                    BlockType.CATEGORICAL,
                                    valid_set=var_set)])
        # Initial boolean flag.
        positions_are_equal = True

        # Compare the best position with the truth.
        for item1, item2 in zip(p1.best_position,
                                [9.0, 1, np.ones(D) / D]):
            if np.isscalar(item1):
                positions_are_equal &= item1 == item2
            else:
                positions_are_equal &= np.array_equal(item1, item2)
        # _end_for_

        # Make sure the initial assignment is correct.
        self.assertTrue(positions_are_equal)

        # Particle 2.
        p2 = JatParticle([DataBlock(5.0,
                                    BlockType.FLOAT,
                                    lower_bound=-10.0,
                                    upper_bound=+10.0),
                          DataBlock(1,
                                    BlockType.INTEGER,
                                    lower_bound=-3,
                                    upper_bound=+3),
                          DataBlock(np.ones(D) / D,
                                    BlockType.CATEGORICAL,
                                    valid_set=var_set)])

        # Update the 'p1' best position.
        p1.best_position = p2.position

        # Make sure that the positions are equal but not the same.
        equal_but_not_the_same = p1.best_position is not p2.position

        # Compare the 'p1' best position with 'p2'.
        for item1, item2 in zip(p1.best_position, p2.position):
            if np.isscalar(item1):
                equal_but_not_the_same &= item1 == item2
            else:
                equal_but_not_the_same &= np.array_equal(item1, item2)
        # _end_for_

        # Check here.
        self.assertTrue(equal_but_not_the_same)
    # _end_def_

# _end_class_


if __name__ == '__main__':
    unittest.main()
