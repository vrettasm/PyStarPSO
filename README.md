# PyStarPSO: Particle swarm optimization algorithms made easy.

![Logo](./logo/main_logo.png)

## Algorithms

This repository implements a collection of particle swarm optimization algorithms in Python3 programming language.

The current implementation offers the following PSO implementations:

  - [Standard](star_pso/engines/standard_pso.py)
  - [Binary](star_pso/engines/binary_pso.py)
  - [Categorical](star_pso/engines/categorical_pso.py)
  - [Integer (Discrete)](star_pso/engines/integer_pso.py)
  - [Jack of all trades](star_pso/engines/jack_of_all_trades_pso.py)
  - [Quantum](star_pso/engines/quantum_pso.py)

All the above methods inherit from the base class [Generic](star_pso/engines/generic_pso.py) which provides some
common functionality.

## Examples
Some optimization examples on how to use these algorithms:

| **Problem**                                                | **Variables** | **Objectives** | **Constraints** |
|:-----------------------------------------------------------|:-------------:|:--------------:|:---------------:|
| [Sphere](examples/sphere.ipynb)                            |    M (=5)     |       1        |       no        |
| [Rosenbrock](examples/rosenbrock_on_a_disk.ipynb)          |    M (=2)     |       1        |        1        |
| [Binh & Korn](examples/binh_and_korn_multiobjective.ipynb) |    M (=2)     |       2        |        2        |
| [Traveling Salesman](examples/tsp.ipynb)                   |    M (=10)    |       1        |       yes       |
| [Zakharov](examples/zakharov.ipynb)                        |    M (=8)     |       1        |       no        |
| [Tanaka](examples/tanaka_multiobjective.ipynb)             |    M (=2)     |       2        |        2        |
| [Shubert](examples/shubert_2D.ipynb)                       |    M (=2)     |       1        |       no        |
| [Gaussian Mixture](examples/gaussian_mixture_2D.ipynb)     |    M (=2)     |       1        |       no        |
| [Test Binary](examples/test_binary_pso.ipynb)              |       X       |       X        |        X        |
| [Test Integer](examples/test_integer_pso.ipynb)            |       X       |       X        |        X        |
| [Test JAT](examples/test_jack_of_all_trades.ipynb)         |       X       |       X        |        X        |
| [Test Categorical](examples/test_categorical_pso.ipynb)    |       X       |       X        |        X        |
| [Test Benchmark](examples/test_benchmark.ipynb)            |       X       |       X        |        X        |

## Benchmarks
We have implemented the following benchmarks of **multimodal** functions:

| **Problem Function**                                                        | **Variables** | **Global Optima** |
|:----------------------------------------------------------------------------|:-------------:|:-----------------:|
| [Equal Maxima](star_pso/benchmarks/equal_maxima.py)                         |       1       |         5         |
| [Five Uneven Peak Trap](star_pso/benchmarks/five_uneven_peak_trap.py)       |       1       |         2         |
| [Uneven Decreasing Maxima](star_pso/benchmarks/uneven_decreasing_maxima.py) |       1       |         1         |
| [Himmelbleau](star_pso/benchmarks/himmelblau.py)                            |       2       |         4         |
| [Gaussian Mixture](star_pso/benchmarks/gaussian_mixture.py)                 |       2       |         2         |
| [Six Hump Camel Back](star_pso/benchmarks/six_hump_camel_back.py)           |       2       |         2         |
| [Rastrigin](star_pso/benchmarks/rastrigin.py)                               |       D       |         K         |
| [Shubert](star_pso/benchmarks/shubert.py)                                   |       D       |      $D*3^D$      |
| [Vincent](star_pso/benchmarks/vincent.py)                                   |       D       |       $6^D$       |

NOTE: The "Shubert" and "Vincent" functions were tested with D = 2, 3 (for simplicity). The code however
is generalized and can solve any number of dimensions. Also, the "Rastrigin" function was tested for D = 2,
4 and K = 6 and 12 respectively.

### Contact

For any questions/comments (**regarding this code**) please contact me at: vrettasm@gmail.com
