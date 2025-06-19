import unittest

import numpy as np
from star_pso.auxiliary.data_block import DataBlock


class TestDataBlock(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        print(">> TestDataBlock - START -")
        cls.rng = np.random.default_rng()
    # _end_def_

    @classmethod
    def tearDownClass(cls) -> None:
        print(">> TestDataBlock - FINISH -", end='\n\n')
    # _end_def_

    def test_init(self):
        """
        Check if the DataBlock has a valid BlockType.

        :return: None.
        """

        # Check if btype is valid callable.
        with self.assertRaises(TypeError):
            _ = DataBlock(position=0.01,
                          btype="float",
                          lower_bound=-1.0,
                          upper_bound=+1.0)
        # _end_with_
    # _end_def_


if __name__ == '__main__':
    unittest.main()
