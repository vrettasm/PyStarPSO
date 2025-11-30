# PyStarPSO: Particle swarm optimization algorithms made easy.

![Logo](./logo/main_logo.jpg)

## Algorithms

This repository implements a collection of particle swarm optimization algorithms in Python3 programming language.

The current implementation offers the following PSO implementations:

  - [StandardPSO](star_pso/engines/standard_pso.py)
  - [BinaryPSO](star_pso/engines/binary_pso.py)
  - [CategoricalPSO](star_pso/engines/categorical_pso.py)
  - [IntegerPSO](star_pso/engines/integer_pso.py)
  - [Jack of all trades-PSO](star_pso/engines/jack_of_all_trades_pso.py)

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
| [Shubert 2D](star_pso/benchmarks/shubert.py)                                |       2       |        18         |
| [Shubert 3D](star_pso/benchmarks/shubert.py)                                |       3       |        81         |
| [Vincent](star_pso/benchmarks/vincent.py)                                   |       D       |       $6^D$       |

## Examples

### Contact

For any questions/comments (**regarding this code**) please contact me at: vrettasm@gmail.com
