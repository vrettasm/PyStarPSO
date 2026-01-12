# PyStarPSO: Particle swarm optimization algorithms made easy.

![Logo](./logo/main_logo.png)

## Algorithms

This repository implements a collection of particle swarm optimization algorithms in Python3 programming language.

The current toolkit offers the following PSO implementations (with supported options):

| **Algorithm**                                                    | **Var. Type(s)** | **Adapt parameters** | **G_best** | **FIPSO** | **Multimodal** | **Parallel** |
|:-----------------------------------------------------------------|:----------------:|:--------------------:|:----------:|:---------:|:--------------:|:------------:|
| [Standard](star_pso/engines/standard_pso.py)                     |      Float       |         Yes          |    Yes     |    Yes    |      Yes       |     Yes      |
| [Binary](star_pso/engines/binary_pso.py)                         |   Int. (0, 1)    |         Yes          |    Yes     |    Yes    |       No       |     Yes      |
| [Categorical](star_pso/engines/categorical_pso.py)               |   Categorical    |         Yes          |    Yes     |    Yes    |       No       |     Yes      |
| [Integer](star_pso/engines/integer_pso.py)                       |     Integer      |         Yes          |    Yes     |    Yes    |       No       |     Yes      |
| [Jack of all trades](star_pso/engines/jack_of_all_trades_pso.py) |      Mixed       |         Yes          |    Yes     |    Yes    |       No       |     Yes      |
| [Quantum](star_pso/engines/quantum_pso.py)                       |      Float       |         Yes          |    Yes     |    Yes    |      Yes       |     Yes      |

All the above methods inherit from the base class [Generic](star_pso/engines/generic_pso.py) which provides some common functionality.

Adding new algorithms **MUST** inherit from the base class.

**Note(1):**
Adapting parameters is supported in the Base class (**GenericPSO**), hence is inherited by all algorithms. However,
since the current adapting algorithm version is checking for convergence of the population to a single solution, using
it with the *multimodal* option would not make much sense and in fact it will mess up the results.

**Note(2):**
Moreover, the *Parallel* option is supported only in the evaluation of the objective (or fitness) function. Therefore,
it is beneficial only in cases where the objective function is "heavy" computationally or has many I/Os. In most cases
setting this option to the default (False), will have the best results.

## Installation

There are two options to install the software.

The easiest way is to download it from PyPI. Simply run the following command on a terminal:
    
    pip install starpso

Alternatively one can clone the latest version directly from GitHub using git as follows:

    git clone https://github.com/vrettasm/StarPSO.git

After the download of the code (or the git clone), one can use the following commands:

    cd PyStarPSO
    pip install .

This will install the latest StarPSO version in the package management system.

## Examples
Some optimization examples on how to use these algorithms:

| **Problem**                                               | **Variables** | **Objectives** | **Constraints** |
|:----------------------------------------------------------|:-------------:|:--------------:|:---------------:|
| [Sphere](examples/sphere.ipynb)                           |    M (=5)     |       1        |       no        |
| [Rosenbrock](examples/rosenbrock_on_a_disk.ipynb)         |    M (=2)     |       1        |        1        |
| [Binh & Korn](examples/binh_and_korn.ipynb)               |    M (=2)     |       2        |        2        |
| [Traveling Salesman](examples/tsp.ipynb)                  |    M (=10)    |       1        |       yes       |
| [Zakharov](examples/zakharov.ipynb)                       |    M (=8)     |       1        |       no        |
| [Tanaka](examples/tanaka.ipynb)                           |    M (=2)     |       2        |        2        |
| [Shubert](examples/shubert_2D.ipynb)                      |    M (=2)     |       1        |       no        |
| [Gaussian Mixture](examples/gaussian_mixture_2D.ipynb)    |    M (=2)     |       1        |       no        |
| [OneMax](examples/one_max.ipynb)                          |    M (=30)    |       1        |       no        |
| [SumAbs](examples/sum_abs.ipynb)                          |    M (=15)    |       1        |       no        |
| [JackofAllTrades](examples/jack_of_all_trades.ipynb)      |    M (=3)     |       1        |       no        |
| [Categorical PSO](examples/categorical_pso.ipynb)         |    M (=4)     |       1        |       no        |
| [Benchmark Functions](examples/benchmark_functions.ipynb) |       M       |       1        |       no        |

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

All the above benchmarks inherit from the base class [TestFunction](star_pso/benchmarks/test_function.py) which provides
some common functionality.  Adding new algorithms **MUST** inherit from the base class.

NOTE: The "Shubert" and "Vincent" functions were tested with D = 2, 3 (for simplicity). The code however
is generalized and can solve any number of dimensions. Also, the "Rastrigin" function was tested for D = 2,
4 and K = 6 and 12 respectively.

### Contact

For any questions/comments (**regarding this code**) please contact me at: vrettasm@gmail.com
