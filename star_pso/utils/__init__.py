from typing import Union
from collections import namedtuple
from numpy.typing import ArrayLike

# Make a type alias for the position's type.
ScalarOrArray = Union[int, float, ArrayLike]

# Declare a named tuple with the parameters
# we want to use in the velocity equations:
# 1) "w0": inertia weight
# 2) "c1": cognitive coefficient
# 3) "c2": social coefficient
# 4) "mode": "fipso", "g_best", "multimodal".
VOptions = namedtuple("VOptions",
                      ["w0", "c1", "c2", "mode"], defaults=[False])
