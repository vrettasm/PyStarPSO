import unittest
import numpy as np
from star_pso.auxiliary.utilities import (nb_clip_item,
                                          check_parameters,
                                          np_median_entropy,
                                          nb_median_hamming_distance)


class TestUtilities(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        print(">> TestUtilities - START -")
    # _end_def_

    @classmethod
    def tearDownClass(cls) -> None:
        print(">> TestUtilities - FINISH -", end='\n\n')
    # _end_def_

    def test_check_params(self) -> None:
        """
        Check for the correct options.

        The input dictionary MUST contain (as minimum)
        the following keys: {"w", "c1", "c2"}.

        :return: None.
        """

        # Check for the right keys.
        with self.assertRaises(KeyError):
            # Key "w" is missing.
            check_parameters(options={"w_": None, "c1": None, "c2": None})

            # Key "c1" is missing.
            check_parameters(options={"w": None, "c_": None, "c2": None})

            # Key "c2" is missing.
            check_parameters(options={"w": None, "c1": None, "c_": None})
    # _end_def_

    def test_nb_clip(self) -> None:
        """
        Check if nb_clip does the right job.

        :return: None.
        """

        # Create a test variable.
        x_test_lower = nb_clip_item(x_new=-1.0,
                                    lower_limit=0.0,
                                    upper_limit=1.0)

        # Check if the lower limit is satisfied.
        self.assertEqual(x_test_lower, 0.0,
                         msg="Lower limit failed.")

        # Create a test variable.
        x_test_upper = nb_clip_item(x_new=2.0,
                                    lower_limit=0.0,
                                    upper_limit=1.0)

        # Check if the upper limit is satisfied.
        self.assertEqual(x_test_upper, 1.0,
                         msg="Upper limit failed.")
    # _end_def_

    def test_nb_median_hamming_distance(self) -> None:
        """
        Check if nb_median_hamming_distance does the right job.
        """
        # Set the matrix dimensions.
        n_rows, n_dims = 100, 15

        # Generate a random binary matrix.
        x = np.random.randint(0, 2, size=(n_rows, n_dims))

        # Compute the median Hamming distance.
        avg_hd = nb_median_hamming_distance(x, normal=True)

        # Ensure the value is in [0, 1].
        self.assertTrue(0.0 < avg_hd < 1.0)

        # Generate a single random vector.
        x0 = np.random.randint(0, 2, n_dims)

        # Create a list with 99 identical copies.
        xl = [x0 for _ in range(99)]

        # Append a new random vector.
        xl.append(np.random.randint(0, 2, n_dims))

        # Create an array.
        z = np.array(xl)

        # Compute the average Hamming distance.
        avg_hd = nb_median_hamming_distance(z, normal=True)

        # This should be around 1% since we have
        # 99 identical vectors and one different.
        self.assertAlmostEqual(avg_hd, 1/100, places=1)
    # _end_def_

    def test_nb_median_entropy(self) -> None:
        """
        Check if np_median_entropy does the right job.
        """

        # Create an empty array of objects.
        x = np.empty(shape=(50, 4), dtype=object)

        # Extract the shape dimensions.
        n_rows, n_cols = x.shape

        # Set of values.
        k = [[1, 0],
             [0, 1, 0],
             [0, 0, 1, 0, 0],
             [0, 0, 0, 1]]

        # Fill the "x" array.
        for i in range(n_rows):
            for j in range(n_cols):
                x[i, j] = np.array(k[j])
        # _end_for_

        # Get the estimate of the spread.
        spread_0 = np_median_entropy(x, normal=True)

        # The spread should be zero, because
        # all particles have the same positions.
        self.assertEqual(spread_0, 0.0)

        # Refill the "x" array.
        for i in range(n_rows):
            for j in range(n_cols):
                # Randomize the positions.
                x[i, j] = np.random.rand(len(k[j]))
        # _end_for_

        # Get the estimate of the spread.
        spread_1 = np_median_entropy(x, normal=True)

        # The spread should be close to 1.0, because
        # all particles have random positions.
        self.assertAlmostEqual(spread_1, 1.0, places=1)
    # _end_def_

# _end_class_


if __name__ == '__main__':
    unittest.main()
