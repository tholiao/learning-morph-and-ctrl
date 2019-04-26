import numpy as np

from walkers import ScalableWalker

DEFAULT_SCENE = "scenes/walker.ttt"
DEFAULT_WALKER = ScalableWalker

N_MRPH_PARAMS = [3, 3, 6]
N_CTRL_PARAMS = [4, 8, 8]

MORPHOLOGY_BOUNDS = [
    [[0.7] * 3, [1.4] * 3],
    [[0.7] * 3, [1.4] * 3],
    [[0.7] * 6, [1.4] * 6]
]

CONTROLLER_BOUNDS = [
    [[1, -np.pi, 0, 0], [45, np.pi, 1, 1]],
    [[1, -np.pi, 0, 0, 0, 0, .5, .5], [45, np.pi, .4, .4, .4, .4, 1, 1]],
    [[1, -np.pi, 0, 0, 0, 0, .5, .5], [45, np.pi, .4, .4, .4, .4, 1, 1]]
]
