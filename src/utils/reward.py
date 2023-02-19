import numpy as np
import random


def get_reward(F, RF):
    R = np.zeros((RF, F))
    for j in range(0, RF):
        for i in range(0, F):
            if random.random() < .2:
                R[j, i] = np.random.uniform(low=-1, high=1)
    return R


def reward_feature(M, N, r):
    r = np.transpose(r)
    reward = np.zeros((M, N))
    i = 1
    k = 0
    while i < M:
        j = 1
        while j < N:
            reward[i, j] = r[k]
            reward[i - 1, j] = r[k]
            reward[i, j - 1] = r[k]
            reward[i - 1, j - 1] = r[k]
            j = j + 2
            k = k + 1
        i = i + 2

    return reward
