import unittest

import numpy as np

from star_pso.population.particle import Particle


class TestParticle(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        print(">> TestParticle - START -")
        cls.rng = np.random.default_rng()
    # _end_def_

    @classmethod
    def tearDownClass(cls) -> None:
        print(">> TestParticle - FINISH -", end='\n\n')
    # _end_def_

    def test_equals(self) -> None:
        """
        Tests the __eq__ method of the Particle class.

        :return: None.
        """

        # Particle 1.
        # Initial position is given as 'tuple'.
        p1 = Particle((1, 2, 3))

        # Should be TRUE.
        self.assertTrue(p1 == p1)

        # Make a new reference of the p1.
        p0 = p1

        # Should be TRUE (p0 is p1).
        self.assertTrue(p0 == p1)

        # Particle 2 (identical to p1).
        # Initial position is given as 'list'.
        p2 = Particle([1, 2, 3])

        # Should be TRUE.
        self.assertTrue(p2 == p2)

        # Should be TRUE.
        self.assertTrue(p1 == p2)

        # Particle 3.
        # Initial position is given as 'np.ndarray'.
        p3 = Particle(np.array([2, 3, 4]))

        # Should be TRUE.
        self.assertTrue(p3 == p3)

        # Should be FALSE.
        self.assertFalse(p1 == p3)
    # _end_def_

    def test_best_position(self) -> None:
        """
        Tests the best_position method. It should assign
        a copy of the input vector.

        :return: None.
        """

        # Particle 1.
        p1 = Particle([0, 0, 0, 0])

        # Make sure the initial assignment is correct.
        self.assertTrue(all(p1.best_position == [0, 0, 0, 0]))

        # Change the best position.
        p1.best_position = (1, 1, 1, 1)

        # Make sure the new assignment is correct.
        self.assertTrue(all(p1.best_position == [1, 1, 1, 1]))

        # Particle 2.
        p2 = Particle([1, 2, 3, 4])

        # Update the 'p1' best position.
        p1.best_position = p2.position

        # Make sure that the positions are equal but not the same.
        equal_but_not_the_same = ((p1.best_position is not p2.position) and
                                  all(p1.best_position == p2.position))
        # Check here.
        self.assertTrue(equal_but_not_the_same)
    # _end_def_

# _end_class_


if __name__ == '__main__':
    unittest.main()
