import numpy as np
from utils.util_functions import *
import random
from scipy.stats import dirichlet
from scipy.special import factorial
from utils.params import *
from utils.reward import *
from utils.transition import *
from utils.gen_trajectories import *
from utils.acc_ratio import *


def initialise_c(tl):
    c = np.zeros((tl, 1))
    for i in range(0, tl):
        c[i] = int(random.randint(0, tl - 1))

    return c


def sample_reward(F, mu, sigma, lb, ub):
    r = np.zeros((F, 1))
    r = mu + np.random.randn(F) * sigma
    r = np.maximum(lb, np.minimum(ub, r));
    r = r.reshape((1, F))
    return r


def assignment_prob(c, alpha):
    z = count(c).astype(int)
    ix = np.arange(1, len(z) + 1).astype(int)
    Z = np.sum(np.multiply(z, ix))
    k = np.power(ix, z) * factorial(z)
    pr = np.math.factorial(Z) / np.prod(np.arange(alpha, alpha + Z)) * np.power(alpha, sum(z)) / np.prod(k)
    return pr


def count(c):
    N = len(c)
    nrCl = np.max(c)
    szCl = np.zeros((1, nrCl))
    z = np.zeros((1, N))
    for k in range(0, nrCl):
        for j in range(0, len(c)):
            if c[j] == k + 1:
                szCl[0, k] = szCl[0, k] + 1
    for i in range(0, nrCl):
        if szCl[0, i] > 0:
            z[0, int(szCl[0, i]) - 1] = z[0, int(szCl[0, i]) - 1] + 1
    return z[0]


def log_post(c):
    log_cluster_ass = np.log(assignment_prob(c, alpha))

    return log_post


class Cluster:
    assignment = []
    reward = []
    policy = []
    values = []
    # haveto add grads


def init_cluster(C, tn, F, X, A):
    C.assignment = np.random.randint(1, tn + 1, size=(1, tn))[0]
    C.reward = np.zeros((tn, F))
    C.policy = np.zeros((X, A, tn))
    C.value = np.zeros((X, tn))
    NC = np.max(C.assignment)
    P = get_transitions(M, N, A, p, q, obstacles)
    for i in range(0, NC):
        C.reward[i, :] = sample_reward(F, mu, sigma, lb, ub)
        r = reward_feature(M, N, C.reward[i, :]).reshape(X, 1)
        V, V_hist, policy, time = policy_iteration(X, P, r, A, gamma, max_iter=100)
        C.policy[:, :, i] = policy
        C.value[:, i] = V
    return C


def newV():
    r = sample_reward(F, mu, sigma, lb, ub)
    r2 = reward_feature(M, N, r).reshape(X, 1)
    v, V_hist, p, time = policy_iteration(X, P, r2, A, gamma, max_iter=100)
    return r, p, v


def update_cluster(C, m, iter=2):
    c = C.assignment[m]
    r1 = C.reward[c, :]
    p1 = C.policy[:, :, c]
    v1 = C.policy[:, c]
    NC = np.max(C.assignment)
    prior = np.zeros((NC + 1, 1))
    for k in range(0, NC + 1):
        for l in range(0, len(C.assignment)):
            if C.assignment[l] == k:
                prior[k] = prior[k] + 1
    prior[m] = prior[m] - 1
    prior[NC + 1] = alpha
    c2 = sampleMult(prior)
    if c2 > NC:  # if its a new cluster, accept it with prob
        r2, p2, v2 = newV()
    else:
        r2 = C.reward[c2, :]
        p2 = C.policy[:, :, c2]
        v2 = C.policy[:, c2]
    # if its an existing cluster, acceptance ratio
    traj = traj_form(traj_set[:, :, m])
    ratio = acc_ratio(traj, r1, v1, r2, v2)
    rand_n = random.uniform(0, 1)
    if rand_n < ratio:
        C.assignment[m] = c2
        if c2 > NC:
            C.reward[c2, :] = r2
            C.policy[:, :, c2] = p2
            C.policy[:, c2] = v2

    return C


def sampleMult(p):
    s = sum(p)
    if s != 1:
        p = p / s
    q = np.cumsum(p)
    rand = random.uniform(0, 1)
    c2 = 0
    for i in range(0, len(q)):
        if q[i] > rand:
            c2 = i
    return c2

def relabel_cluster(C):

    return C