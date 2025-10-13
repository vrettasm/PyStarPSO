import unittest
from star_pso.auxiliary.utilities import (nb_clip_item, cost_function,
                                          check_velocity_parameters,
                                          linear_rank_probabilities)


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

        The input dictionary MUST contain as minimum
        the following keys: {"w0", "c1", "c2"}.

        :return: None.
        """

        # Check for the right keys.
        with self.assertRaises(KeyError):
            # Key "w0" is missing.
            check_velocity_parameters(options={"w_": None, "c1": None, "c2": None})

            # Key "c1" is missing.
            check_velocity_parameters(options={"w0": None, "c_": None, "c2": None})

            # Key "c2" is missing.
            check_velocity_parameters(options={"w0": None, "c1": None, "c_": None})
        # _end_with_
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

    def test_cost_function_min(self) -> None:
        """
        Check the default behaviour of the cost_function decorator.
        """

        @cost_function
        def func_1(num):
            return num
        # _end_def_

        # Test variable.
        x = 1

        # Get the result from the function call.
        result_x = func_1(x)

        # Here x == f(x).
        self.assertEqual(x, result_x["f_value"])
    # _end_def_

    def test_cost_function_max(self) -> None:
        """
        Check the minimize=True, behaviour of the cost_function decorator.
        """
        @cost_function(minimize=True)
        def func_2(num):
            return num
        # _end_def_

        # Test variable.
        y = 1

        # Get the result from the function call.
        result_y = func_2(y)

        # Here y == -f(y).
        self.assertEqual(y, -result_y["f_value"])
    # _end_def_

    def test_linear_rank_probabilities(self) -> None:
        """
        Check the behaviour of the linear_rank_probabilities function.
        We check for the correct type, the positive value and whether
        the sum of the probabilities sums to 1.0

        :return: None.
        """
        # Test input is integer.
        with self.assertRaises(TypeError):
            _ = linear_rank_probabilities(2.3)
        # _end_with_

        # Test input is positive.
        with self.assertRaises(ValueError):
            _ = linear_rank_probabilities(-10)
        # _end_with_

        # Get the probabilities for p_size=10.
        probs, probs_sum = linear_rank_probabilities(10)

        # Test that the probs sum to the same number.
        self.assertEqual(probs.sum(), probs_sum)

        # Test that the probs_sum is equal to 1.0.
        self.assertAlmostEqual(probs_sum, 1.0, places=3)
    # _end_def_

# _end_class_


if __name__ == '__main__':
    unittest.main()
