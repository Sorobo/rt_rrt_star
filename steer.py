import numpy as np
from config import STEP_SIZE

def steer(x_rand, n_closest):
    #return x_rand if np.linalg.norm(x_rand - n_closest.x) <= STEP_SIZE else n_closest.x + STEP_SIZE * (x_rand - n_closest.x) / np.linalg.norm(x_rand - n_closest.x)
    direction = x_rand - n_closest.x
    distance = np.linalg.norm(direction)
    if distance <= STEP_SIZE:
        return x_rand
    else:
        return n_closest.x + (direction / distance) * STEP_SIZE
    