import numpy as np
from utils.util_functions import *
import random
from scipy.stats import dirichlet
from scipy.special import factorial
from utils.params import *
from utils.reward import *
from utils.transition import *
from utils.gen_trajectories import *


def acc_ratio(traj, r1, v1, r2, v2):
    q1 = qfromv(v1, r1)
    q2 = qfromv(v2, r2)
    bq1 = eta * q1
    bq2 = eta * q2
    nq1 = np.log(bq1.sum(axis=1))
    nq2 = np.log(bq2.sum(axis=1))
    nq1 = bq1 - nq1.reshape((nq1.shape[0], 1))
    nq2 = bq2 - nq2.reshape((nq2.shape[0], 1))
    x = 0
    for t in range(0, traj.shape[0]):
        s = int(traj[t, 0])
        a = int(traj[t, 1])
        x = x + traj[t, 2] * (nq1[s, a] - nq2[s, a])
    ratio = np.exp(x)
    return ratio


def qfromv(v, r):
    r = reward_feature(M, N, r).reshape(X, 1)
    Q = np.zeros((X, A))
    P = get_transitions(M, N, A, p, q, obstacles)
    for a in range(0, A):
        for s in range(0, X):
            Q[s, a] = r[s] + gamma * (np.dot(P[s, :, a], v))

    return Q
