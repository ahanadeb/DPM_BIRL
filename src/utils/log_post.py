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


def calLogLLH(r, traj, p):
    r = reward_feature(M, N, r).reshape(X, 1)
    P = get_transitions(M, N, A, p, q, obstacles)
    v, V_hist, p, time = policy_iteration(X, P, r, A, gamma, max_iter=100)
    q = qfromv(v, r)
    dQ = calgradQ(p)

    bq = eta * q
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

    grad = 0
    for i in range(len(traj)):
        s = int(traj[i, 0])
        a = int(traj[i, 1])
        n = int(traj[i, 2])
        j = (a - 1) * X + s
        grad = grad + n * dlogPi[:, j]

    return llh, grad


def calgradQ(p):
    Epi = np.zeros((X, X * A))
    p = conv_policy(p)
    idx = (p - 1) * X + np.arange(X).reshape((X, 1))
    idx = (idx - 1) * X + np.arange(X).reshape((X, 1))
    for i in range(0, len(idx)):
        Epi[idx[i], 0] = 1
    P = get_transitions(M, N, A, p, q, obstacles).reshape((X * A, X))
    dQ = (np.eye(X * A) - P * Epi)  # divide by feature vector SHAPE ISSUE dq is ultimately F, X*A
    return dQ

def calLogPrior(r):
    x = r - mu
    prior = r*np.transpose(r) / (2* np.power(sigma, 2))
    grad = -x /np.power(sigma, 2)
    return prior, grad