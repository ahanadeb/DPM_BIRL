import numpy as np
from utils.util_functions import *
from utils.acc_ratio import *
import random
from scipy.stats import dirichlet
from scipy.special import factorial
from utils.params import *
from utils.reward import *
from utils.transition import *
from utils.gen_trajectories import *


def calLogLLH(r, traj, p1):
    r = reward_feature(M, N, r).reshape(X, 1)
    P = get_transitions(M, N, A, p, q, obstacles)
    v, V_hist, p1, time = policy_iteration(X, P, r, A, gamma, max_iter=100)
    q1 = qfromv(v, r)
    dQ = calgradQ(p1)

    bq = eta * q1
    nq = np.log(bq.sum(axis=1))
    nq = bq - nq.reshape((nq1.shape[0], 1))

    llh = 0
    for i in range(len(traj)):
        s = int(traj[i, 0])
        a = int(traj[i, 1])
        n = int(traj[i, 2])
        llh = llh + n * nq[s, a]
    # softmax policy
    pi_sto = np.exp(bq)
    pi_sto = pi_sto / pi_sto.sum(axis=1)
    # compute dlogPi/dw
    dlogPi = np.zeros((F, X * A))
    for f in range(0, F):
        x = dQ[f, :].reshape((X, A))
        y = (pi_sto * x).sum(axis=1)
        z = eta * (x - y)
        dlogPi[f, :] = z.reshape((1, X * A))

    grad = np.zeros((F, 1))
    for i in range(len(traj)):
        s = int(traj[i, 0])
        a = int(traj[i, 1])
        n = int(traj[i, 2])
        j = (a - 1) * X + s
        grad[:, 0] = grad[:, 0] + n * dlogPi[:, j]

    return llh, grad


def calgradQ(p):
    Epi = np.zeros((X, X * A))
    p = conv_policy(p)
    idx = (p) * X + np.arange(X).reshape((X, 1))
    idx = (idx - 1) * X + np.arange(X).reshape((X, 1))
    print('idx', idx)
    for i in range(0, len(idx)):
        Epi[int(idx[i] - 1), 0] = 1
    P = get_transitions(M, N, A, p, q, obstacles).reshape((X, X * A))
    F_state_feat = stateFeature(X, F, M, N)
    dQ = np.linalg.inv(
        np.eye(X * A) - P * Epi) * F_state_feat  # divide by feature vector SHAPE ISSUE dq is ultimately F, X*A
    return dQ


def calLogPrior(r):
    x = r - mu
    prior = r * np.transpose(r) / (2 * np.power(sigma, 2))
    grad = -x / np.power(sigma, 2)
    return prior, grad


def stateFeature(X, F, M, N):
    F = np.zeros((X, F))
    for x in range(1, M + 1):
        for y in range(1, N + 1):
            s = loc2s(x, y, M)
            i = np.ceil(x / B)
            j = np.ceil(y / B)
            f = loc2s(i, j, M / B);
            F[int(s) - 1, int(f) - 1] = 1;
    return F


def loc2s(x, y, M):
    x = max(1, min(M, x));
    y = max(1, min(M, y));
    s = (y - 1) * M + x;

    return s