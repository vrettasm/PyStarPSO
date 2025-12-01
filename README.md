# PyStarPSO: Particle swarm optimization algorithms made easy.

![Logo](./logo/main_logo.jpg)

## Algorithms

This repository implements a collection of particle swarm optimization algorithms in Python3 programming language.

The current implementation offers the following PSO implementations:

  - [Standard](star_pso/engines/standard_pso.py)
  - [Binary](star_pso/engines/binary_pso.py)
  - [Categorical](star_pso/engines/categorical_pso.py)
  - [Integer (Discrete)](star_pso/engines/integer_pso.py)
  - [Jack of all trades](star_pso/engines/jack_of_all_trades_pso.py)

All the above methods inherit from the base class [Generic](star_pso/engines/generic_pso.py) which provides some
common functionality.

## Benchmarks
We have implemented the following benchmark functions:

| **Problem Function**                                                        | **Variables** | **Global Optima** |
|:----------------------------------------------------------------------------|:-------------:|:-----------------:|
| [Equal Maxima](star_pso/benchmarks/equal_maxima.py)                         |       1       |         5         |
| [Five Uneven Peak Trap](star_pso/benchmarks/five_uneven_peak_trap.py)       |       1       |         2         |
| [Uneven Decreasing Maxima](star_pso/benchmarks/uneven_decreasing_maxima.py) |       1       |         1         |
| [Himmelbleau](star_pso/benchmarks/himmelblau.py)                            |       2       |         4         |
| [Gaussian Mixture](star_pso/benchmarks/gaussian_mixture.py)                 |       2       |         2         |
| [Six Hump Camel Back](star_pso/benchmarks/six_hump_camel_back.py)           |       2       |         2         |
| [Rastriging](star_pso/benchmarks/rastrigin.py)                              |       D       |         K         |
| [Shubert](star_pso/benchmarks/shubert.py)                                   |       D       |      $D*3^D$      |
| [Vincent](star_pso/benchmarks/vincent.py)                                   |       D       |       $6^D$       |

NOTE: The "Shubert" and "Vincent" functions were tested with D = 2, 3 (for simplicity). The code however
is generalized and can solve any number of dimensions. Also the "Rastrigin" function was tested for D = 2, 4
and K = 6 and 12 respectively.

## Examples

### Contact

For any questions/comments (**regarding this code**) please contact me at: vrettasm@gmail.com
