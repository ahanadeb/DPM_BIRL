import numpy as np
from utils.util_functions import *
import random
from scipy.stats import dirichlet
from scipy.special import factorial
from utils.params import *
from utils.reward import *
from utils.transition import *
from utils.gen_trajectories import *
from utils.log_post import *


def update_weight(k, traj_set, C):
    sigma = 0.01
    for i in range(0, weight_iter):
        t = []
        for l in range(0, len(C.assignment)):
            if C.assignment[l] == k:
                t = np.append(t, l)
        if len(t) > 1:
            traj = traj_merj(traj_set, t)
        else:
            traj = traj_form(traj_set[:, :, int(t[0])])
        r = C.reward[:, k]
        p = C.policy[:, :, k]
        v = C.value[:, k]
        if np.all(C.llh == 0):  # figure out shape of llh
            llh, gradL = calLogLLH(r, traj, p)
            prior, gradP = calLogPrior(r)
            C.llh[k, 0] = llh
            C.prior[k, 0] = prior
            C.gradL[:, k] = gradL
            C.gradP[:, k] = gradP
        logP = C.llh[k] + C.prior[k]
        grad = C.gradL[:, k] + C.gradP[:, k]
        eps = np.random.uniform(low=-1, high=1, size=(F, 1))
        r2 = w + np.power(sigma, 2) * grad / 2 + sigma * eps
        r2 = np.max(lb, np.min(ub, r2))
        r3 = reward_feature(M, N, r2).reshape(X, 1)
        P = get_transitions(M, N, A, p, q, obstacles)
        v2, V_hist, p2, time = policy_iteration(X, P, r3, A, gamma, max_iter=100)
        llh2, gradL2 = calLogLLH(r2, trajInfo, p2)
        prior2, gradP2 = calLogPrior(r2)
        logP2 = llh2 + prior2
        grad2 = gradL2 + gradP2

        a = eps + (sigma / 2) * (grad + grad2)
        a = np.exp(-.5 * sum(a ** 2)) * np.exp(logP2)
        b = np.exp(-.5 * sum(eps ** 2)) * np.exp(logP)

        ratio = a / b
        rand_n = random.uniform(0, 1)
        if rand_n < ratio:
            C.reward[:, k] = r2
            C.policy[:, k] = p2
            C.value[:, k] = v2
            C.llh[k] = llh2
            C.prior[k] = prior2
            C.gradL[:, k] = gradL2
            C.gradP[:, k] = gradP2

    return C
