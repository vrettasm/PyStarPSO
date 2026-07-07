import unittest
import numpy as np

from star_pso.population.swarm import Swarm
from star_pso.population.particle import Particle
from star_pso.engines.generic_pso import GenericPSO
from star_pso.utils.auxiliary import linear_rank_probabilities



class TestGenericPSO(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        print(">> TestGenericPSO - START -")

        # Test population size.
        cls.pop_size = 5

        # Test particle size.
        cls.x_dim = 3

        # Make an initial population.
        population: list[Particle] = [
            Particle((i+1) * np.ones(shape=cls.x_dim, dtype=float))
            for i in range(cls.pop_size)
        ]

        # Initial swarm.
        cls.swarm_t0 = Swarm(population)

        # Assign a value/best_value for each particle.
        for p in cls.swarm_t0.population:
            p.value = np.sum(np.abs(p.position))
            p.best_value = -(p.value ** 2)

        # Dummy objective function.
        dummy_f = lambda: None

        # Test pso object.
        cls.test_pso = GenericPSO(initial_swarm=cls.swarm_t0,
                                  obj_func=dummy_f)
    # _end_def_

    @classmethod
    def tearDownClass(cls) -> None:
        print(">> TestGenericPSO - FINISH -", end='\n\n')
    # _end_def_

    def test_allow_parameters_to_update(self) -> None:

        # Initially the "_allow_parameters_to_update"
        # is set to 'True' by default.
        self.assertTrue(self.test_pso._allow_parameters_to_update)

        # Disable parameters.
        self.test_pso.disable_parameters_update()
        self.assertFalse(self.test_pso._allow_parameters_to_update)

        # Enable parameters.
        self.test_pso.enable_parameters_update()
        self.assertTrue(self.test_pso._allow_parameters_to_update)
    # _end_def_

    def test_get_local_best_positions(self):
        # Available modes are:
        # 'fipso', 'multimodal' and 'g_best'
        with self.assertRaises(ValueError):
            self.test_pso.get_local_best_positions("none")

        # Get the local best.
        l_best = self.test_pso.get_local_best_positions("g_best")

        # Get the dimensions of l_best.
        n_rows, n_cols = l_best.shape

        # Test the number of rows.
        self.assertEqual(n_rows, len(self.swarm_t0.population))

        # Test the number of columns.
        self.assertEqual(n_cols, len(self.swarm_t0.population[0]))
    # _end_def_

    def test_fully_informed(self):
        # The rank probabilities do not change.
        p_weights, _ = linear_rank_probabilities(self.pop_size)

        # Test: use_best=False
        f_best_0 = self.test_pso.fully_informed(self.swarm_t0.population,
                                                use_best=False)

        # The positions should be sorted using the 'value'.
        true_vector_0 = (p_weights @ np.array([[1, 1, 1],
                                               [2, 2, 2],
                                               [3, 3, 3],
                                               [4, 4, 4],
                                               [5, 5, 5]]))

        self.assertTrue(np.array_equal(true_vector_0, f_best_0) )

        # Test: use_best=True
        f_best_1 = self.test_pso.fully_informed(self.swarm_t0.population,
                                                use_best=True)

        # The positions should be sorted using the 'best_value'.
        true_vector_1 = p_weights @ np.array([[5, 5, 5],
                                              [4, 4, 4],
                                              [3, 3, 3],
                                              [2, 2, 2],
                                              [1, 1, 1]])

        self.assertTrue(np.array_equal(true_vector_1, f_best_1))
    # _end_def_

# _end_class_


if __name__ == '__main__':
    unittest.main()
