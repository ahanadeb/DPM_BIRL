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
from utils.log_post import *
from utils.saveHist import *



def evd(hist, reward_gt, tn):
    # print("hist length", len(hist.policy))
    P = get_transitions(M, N, A, p, q, obstacles)
    e = 0
    for i in range(0, tn):
        k = int(hist.assignment[i])
        # changed k to i here in the next line
        r1 = reward_feature(M, N, reward_gt[:, i]).reshape(X, 1)
        r2 = hist.reward[:, k]
        r2 = reward_feature(M, N, r2).reshape(X, 1)
        V, V_hist, opt_policy, time = policy_iteration(X, P, r2, A, gamma, max_iter=100)
        V_eval = evaluate_analytical(P, opt_policy, r1, gamma)

        V_true, V_hist, p1, time = policy_iteration(X, P, r1, A, gamma, max_iter=100)
        e = e + V_true - V_eval

    e = e / tn
    return np.mean(e)
