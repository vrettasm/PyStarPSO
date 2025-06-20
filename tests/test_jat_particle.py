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

# _end_class_


if __name__ == '__main__':
    unittest.main()
