import unittest
import numpy as np

from star_pso.population.swarm import Swarm
from star_pso.population.particle import Particle
from star_pso.utils.auxiliary import BlockType
from star_pso.utils.data_block import DataBlock
from star_pso.population.jat_particle import JatParticle

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

    def test_init(self):
        """
        Check if the __init__ method handles the
        '_has_categorical' variable correctly.

        :return: None.
        """
        # Define the number of optimizing variables.
        n_dim = 2

        # Define the number of particles.
        n_particles = 10

        # Draw random samples for the initial points.
        x_t0 = self.rng.uniform(-1.5, +1.5, size=(n_particles, n_dim))

        # Initialize the particle population.
        swarm_a = Swarm([Particle(x) for x in x_t0])

        # Check if it has categorical variables.
        self.assertFalse(swarm_a.has_categorical)

        # Set the variables for the categorical
        # optimization variable.
        var_set = ["a", "b", "c"]

        # Set the number of categorical variables.
        n_dim = len(var_set)

        # Set the number of particles.
        n_particles = 10

        # Initialize the JAT particle population.
        swarm_b = Swarm([JatParticle([DataBlock(self.rng.uniform(-10.0, +10.0),
                                                BlockType.FLOAT,
                                                lower_bound=-10.0,
                                                upper_bound=+10.0),
                                      DataBlock(np.ones(n_dim) / n_dim,
                                                BlockType.CATEGORICAL,
                                                valid_set=var_set)]) for _ in range(n_particles)])
        # Check if it has categorical variables.
        self.assertTrue(swarm_b.has_categorical)
    # _end_def_

    def test_global_best_index(self):
        """
        Check if the global_best_index returns the right index.

        :return: None.
        """
        # Define the number of optimizing variables.
        n_dim = 2

        # Define the number of particles.
        n_particles = 10

        # Draw random samples for the initial points.
        x_t0 = self.rng.uniform(-1.5, +1.5, size=(n_particles, n_dim))

        # Initialize the particle population.
        swarm_a = Swarm([Particle(x) for x in x_t0])

        # Select randomly a particle position.
        j = self.rng.integers(n_particles)

        # Manually change its f_value.
        swarm_a[j].value = 1000

        # Check the global index property.
        self.assertEqual(j, swarm_a.global_best_index)
    # _end_def_

    def test_best_particle(self):
        """
        Check if the best_particle returns the one with the highest value.

        :return: None.
        """
        # Set the number of optimizing variables.
        n_dim = 2

        # Set the number of particles.
        n_particles = 100

        # Draw random samples for the initial points.
        x_t0 = self.rng.uniform(-1.5, +1.5, size=(n_particles, n_dim))

        # Initialize the particle population.
        swarm_a = Swarm([Particle(x) for x in x_t0])

        # Assign made up fitness values to the swarm.
        for p in swarm_a.population:
            # Random values in [0, 100).
            p.value = 100*self.rng.random()
        # _end_if_

        # Select randomly a particle position
        # and set manually its value to 1000.
        swarm_a[self.rng.integers(n_particles)].value = 1000

        # Check the value of the best particle.
        self.assertEqual(1000, swarm_a.best_particle().value)
    # _end_def_

# _end_class_


if __name__ == '__main__':
    unittest.main()
