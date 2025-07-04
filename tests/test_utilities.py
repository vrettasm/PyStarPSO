import unittest

from star_pso.auxiliary.utilities import (nb_clip_item,
                                          check_parameters)


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

    def test_nb_average_hamming_distance(self) -> None:
        """
        Check if nb_average_hamming_distance does the right job.
        """
        # Set the matrix dimensions.
        n_rows, n_dims = 100, 15

        # Generate a random binary matrix.
        x = np.random.randint(0, 2, size=(n_rows, n_dims))

        # Compute the average Hamming distance.
        avg_hd = nb_average_hamming_distance(x, normal=True)

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
        avg_hd = nb_average_hamming_distance(z, normal=True)

        # This should be around 1% since we have 99 identical
        # vectors and one different.
        self.assertTrue(np.isclose(avg_hd, 1/100, atol=0.01))
    # _end_def_

# _end_class_


if __name__ == '__main__':
    unittest.main()
