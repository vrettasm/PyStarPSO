import unittest

import numpy as np
from star_pso.utils.auxiliary import BlockType
from star_pso.utils.data_block import DataBlock


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

        # Check if the boundaries are correct.
        with self.assertRaises(ValueError):
            _ = DataBlock(position=0.01,
                          btype=BlockType.FLOAT,
                          lower_bound=[-1.0, +2.0],
                          upper_bound=[+1.0, -2.0])
        # _end_with_
    # _end_def_

    def test_position_float(self):
        """
        Test whether the upd_float() returns a float variable.

        :return: None.
        """

        # Create a test data block (FLOAT).
        blk = DataBlock(position=self.rng.random(),
                        btype=BlockType.FLOAT,
                        lower_bound=0.0,
                        upper_bound=1.0)

        # Here we test the upd_float().
        blk.position = self.rng.random()
        self.assertTrue(isinstance(blk.position, float))
    # _end_def_

    def test_position_integer(self):
        """
        Test whether the upd_integer() returns an integer variable.

        :return: None.
        """

        # Create a test data block (INTEGER).
        blk = DataBlock(position=self.rng.integers(-5, 5),
                        btype=BlockType.INTEGER,
                        lower_bound=-10,
                        upper_bound=+10)

        # Here we test the upd_integer().
        blk.position = self.rng.random()
        self.assertTrue(isinstance(blk.position, int))
    # _end_def_

    def test_position_binary(self):
        """
        Test whether the upd_binary() returns a binary variable.

        :return: None.
        """

        # Create a test data block (BINARY).
        blk = DataBlock(position=self.rng.integers(0, 2),
                        btype=BlockType.BINARY)

        # Here we test the upd_binary().
        blk.position = self.rng.random()
        self.assertTrue(blk.position in (0, 1))
    # _end_def_

    def test_position_categorical(self):
        """
        Test whether the upd_categorical() returns
        a vector of probabilities that sum to one.

        :return: None.
        """
        # Test variable set.
        var_set = ["a", "b", "c"]

        # Get the size of the vector.
        n_var = len(var_set)

        # Create a test data block (CATEGORICAL).
        blk = DataBlock(np.ones(n_var)/n_var,
                        BlockType.CATEGORICAL,
                        valid_set=var_set)

        # Here we test the upd_categorical().
        blk.position = self.rng.random(n_var)
        self.assertAlmostEqual(np.sum(blk.position), 1.0)
    # _end_def_

    def test_reset_position_float(self):
        """
        Test whether the init_float() returns a float variable.

        :return: None.
        """

        # Create a test data block (FLOAT).
        blk = DataBlock(position=self.rng.random(),
                        btype=BlockType.FLOAT,
                        lower_bound=0.0,
                        upper_bound=1.0)

        # Here we test the init_float().
        blk.reset_position()

        self.assertTrue(0.0 <= blk.position <= 1.0)
        self.assertTrue(isinstance(blk.position, float))
    # _end_def_

    def test_reset_position_integer(self):
        """
        Test whether the init_integer() returns
        an integer variable within its limits.

        :return: None.
        """
        # Create a test data block (INTEGER).
        blk = DataBlock(position=self.rng.integers(-5, 5),
                        btype=BlockType.INTEGER,
                        lower_bound=-10,
                        upper_bound=+10)

        # Here we test the init_integer().
        blk.reset_position()

        self.assertTrue(-10 <= blk.position <= 10)
        self.assertTrue(isinstance(blk.position, int))
    # _end_def_

    def test_reset_position_binary(self):
        """
        Test whether the init_binary() returns
        a binary variable, within (0, 1).

        :return: None.
        """

        # Create a test data block (BINARY).
        blk = DataBlock(position=self.rng.integers(0, 2, dtype=int),
                        btype=BlockType.BINARY)

        # Here we test the init_binary().
        blk.reset_position()

        self.assertTrue(blk.position in (0, 1))
        self.assertTrue(isinstance(blk.position, int))
    # _end_def_

    def test_rest_position_categorical(self):
        """
        Test whether the init_categorical() returns
        a vector of probabilities that sum to one.

        :return: None.
        """
        # Test variable set.
        var_set = ["a", "b", "c"]

        # Get the size of the vector.
        n_var = len(var_set)

        # Create a test data block (CATEGORICAL).
        blk = DataBlock(np.ones(n_var)/n_var,
                        BlockType.CATEGORICAL,
                        valid_set=var_set)

        # Here we test the init_categorical().
        blk.reset_position()

        self.assertAlmostEqual(np.sum(blk.position), 1.0)
    # _end_def_

    def test_equals(self):
        """
        Tests the __eq__ method of the DataBlock class.

        :return: None.
        """
        # Test variable set.
        var_set = ["a", "b", "c", "d"]

        # Size of the vector.
        n_var = len(var_set)

        # DataBlock 1.
        p1 = DataBlock(9.0,
                       BlockType.FLOAT,
                       lower_bound=-10.0,
                       upper_bound=+10.0)

        # Should be TRUE.
        self.assertTrue(p1 == p1)

        # DataBlock 2.
        p2 = DataBlock(1,
                       BlockType.INTEGER,
                       lower_bound=-3,
                       upper_bound=+3)

        # Should be TRUE.
        self.assertTrue(p2 == p2)

        # DataBlock 3.
        p3 = DataBlock(np.ones(n_var) / n_var,
                       BlockType.CATEGORICAL,
                       valid_set=var_set)

        # Should be TRUE.
        self.assertTrue(p3 == p3)

        # DataBlock 4 (identical to p2).
        p4 = DataBlock(1,
                       BlockType.INTEGER,
                       lower_bound=-3,
                       upper_bound=+3)

        # Should be TRUE.
        self.assertTrue(p4 == p4)

        # Should be TRUE.
        self.assertTrue(p4 == p2)

        # Should be FALSE.
        self.assertFalse(p1 == p2)

        # Should be FALSE.
        self.assertFalse(p2 == p3)

        # Should be FALSE.
        self.assertFalse(p3 == p1)
    # _end_def_

# _end_class_


if __name__ == '__main__':
    unittest.main()
