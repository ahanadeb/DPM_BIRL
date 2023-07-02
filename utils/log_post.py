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
from utils.cluster_assignment import *
np.set_printoptions(threshold=np.inf)

def calLogLLH(r, traj, p1):
    r = reward_feature(M, N, r).reshape(X, 1)
    P = get_transitions(M, N, A, p, q, obstacles)
    v, V_hist, p1, time = policy_iteration(X, P, r, A, gamma, max_iter=100)
    q1 = qfromv(v, r)
    dQ = calgradQ(p1)
   # print("dQ", dQ)

    bq = eta * q1
    #print(bq)
    nq = np.log(np.sum(np.exp(bq),axis=1))
    #print(nq.shape)

    nq1 = bq - nq.reshape((nq.shape[0], 1))

    #print(breake)
    llh = 0
    for i in range(len(traj)):
        s = int(traj[i, 0])
        a = int(traj[i, 1])
        n = int(traj[i, 2])
        llh = llh + n * nq1[s, a]
    # softmax policy
    pi_sto = np.exp(bq)
    pi_sto = pi_sto / pi_sto.sum(axis=1).reshape((pi_sto.sum(axis=1).shape[0], 1))
    # compute dlogPi/dw
    dlogPi = np.zeros((F, X * A))
    for f in range(0, F):
        x = dQ[:, f].reshape((X, A))
        y = (pi_sto * x).sum(axis=1)
        z = eta * (x - y.reshape((y.shape[0], 1)))
        dlogPi[f, :] = z.reshape((1, X * A))

    grad = np.zeros((F, 1))
    for i in range(len(traj)):
        s = int(traj[i, 0])
        a = int(traj[i, 1])
        n = int(traj[i, 2])
        j = (a - 1) * X + s
        grad[:, 0] = grad[:, 0] + n * dlogPi[:, j]

    return llh, grad


def calgradQ(p1):
    Epi = np.zeros((X, X * A))
    p1 = conv_policy(p1)
    idx = (p1) * X + np.arange(X).reshape((X, 1))
    idx = (idx - 1) * X + np.arange(X).reshape((X, 1))
    for i in range(0, len(idx)):
        a = np.floor(idx[i] / X)
        b = idx[i] - a * X
        Epi[int(b - 1), int(a - 1)] = 1
    #print("Epi", np.nonzero(Epi))
    P = get_transitions(M, N, A, p, q, obstacles).reshape((X * A, X))
    F_state_feat = stateFeature(X, F, M, N)
    F_state_feat = np.tile(F_state_feat, (A, 1))
    #print("check this", np.linalg.inv(
    #    np.eye(X * A) - np.matmul(P, Epi)))

    #dQ = np.matmul(np.linalg.inv(
     #   np.eye(X * A) - np.matmul(P, Epi)), F_state_feat)  # output F, X*A
    a =np.eye(X * A) - np.matmul(P, Epi)
    b = F_state_feat
    dQ=np.linalg.solve(a, b)
    #print(np.linalg.solve(a,b))
    #print(breake)
    return dQ


def calLogPrior(r):
    x = r - mu
    prior = np.sum(-1 * (x * np.transpose(x)) / (2 * np.power(sigma, 2)))

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


def calDPMLogPost(traj_set, C):
    logDPprior = np.log(assignment_prob(C.assignment, alpha))
    logLLH = 0
    logPrior = 0
    NC = int(np.max(C.assignment))
    for k in range(0, NC + 1):
        r1 = C.reward[:, k]
        if not C.policy_empty:
            r = reward_feature(M, N, r1).reshape(X, 1)
            P = get_transitions(M, N, A, p, q, obstacles)
            V, V_hist, policy, time = policy_iteration(X, P, r, A, gamma, max_iter=100)
        else:
            policy = C.policy[:, :, k]

        t = []
        for l in range(0, len(C.assignment)):
            if C.assignment[l] == k:
                t = np.append(t, l)
        if len(t) > 1:
            traj = traj_merj(traj_set, t)
        else:
            traj = traj_form(traj_set[:, :, int(t[0])])

        llh, gradL = calLogLLH(r1, traj, policy)
        prior, gradP = calLogPrior(r1)
        logLLH = logLLH + llh
        logPrior = logPrior + prior
    print("llh ", logDPprior, " ", logLLH," ",logPrior  )
    logPost = logDPprior + logLLH + logPrior

    return logPost
