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

def evd(hist, reward_gt, maxiter):
    print("hist length", len(hist.policy))
    EVD=[]
    P = get_transitions(M, N, A, p, q, obstacles)
    for j in range(0,maxiter):
        e = 0
        for i in range(0, tn):
            r = reward_feature(M, N, reward_gt[:,i]).reshape(X, 1)
            k=hist.assignment[j][i]
            policy=hist.policy[j][k]
            V_eval = evaluate_analytical(P, policy, r, gamma)
            V_true, V_hist, p1, time = policy_iteration(X, P, r, A, gamma, max_iter=100)
            e = e+ np.abs(np.sum(V_true-V_eval))
        EVD.append(e)
    return EVD
