
import random

def avoid_negative_zero(x):
    if x == -0.0:
        return 0.0
    else:
        return x
    

def initialize_random_generator(seed):
    rng = random.Random(seed)
    return rng