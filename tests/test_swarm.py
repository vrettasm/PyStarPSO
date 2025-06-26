import unittest
import numpy as np

from star_pso.auxiliary.swarm import Swarm
from star_pso.auxiliary.particle import Particle
from star_pso.auxiliary.utilities import BlockType
from star_pso.auxiliary.data_block import DataBlock
from star_pso.auxiliary.jat_particle import JatParticle

class TestSwarm(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        print(">> TestSwarm - START -")
        cls.rng = np.random.default_rng()
    # _end_def_

    @classmethod
    def tearDownClass(cls) -> None:
        print(">> TestSwarm - FINISH -", end='\n\n')
    # _end_def_

    def test_post_init(self):
        # Define the number of optimizing variables.
        D = 2

        # Define the number of particles.
        N = 10

        # Draw random samples for the initial points.
        X_t0 = self.rng.uniform(-1.5, +1.5, size=(N, D))

        # Initialize the particle population.
        swarm_A = Swarm([Particle(x) for x in X_t0])

        # Check if it has categorical variables.
        self.assertFalse(swarm_A.has_categorical)

        # Define the variable set for the categorical
        # optimization variable.
        var_set = ["a", "b", "c"]

        # Define the number of categorical variables.
        D = 3

        # Define the number of particles.
        N = 10

        # Initialize the JAT particle population.
        swarm_B = Swarm([JatParticle([DataBlock(self.rng.uniform(-10.0, +10.0),
                                                BlockType.FLOAT,
                                                lower_bound=-10.0,
                                                upper_bound=+10.0),
                                      DataBlock(np.ones(D) / D,
                                                BlockType.CATEGORICAL,
                                                valid_set=var_set)]) for _ in range(N)])
        # Check if it has categorical variables.
        self.assertTrue(swarm_B.has_categorical)
    # _end_def_

    def test_global_best_index(self):
        # Define the number of optimizing variables.
        D = 2

        # Define the number of particles.
        N = 10

        # Draw random samples for the initial points.
        X_t0 = self.rng.uniform(-1.5, +1.5, size=(N, D))

        # Initialize the particle population.
        swarm_A = Swarm([Particle(x) for x in X_t0])

        # Select randomly a particle position.
        j = self.rng.integers(N)

        # Manually change its f_value.
        swarm_A[j].value = 1000

        # Check the global index property.
        self.assertEqual(j, swarm_A.global_best_index)
    # _end_def_


if __name__ == '__main__':
    unittest.main()
