import matplotlib.pyplot as plt
import numpy as np
from utils.util_functions import *
from utils.reward import *
from utils.transition import get_transitions
from utils.policies import *
from utils.plot import *
from utils.gen_trajectories import *
from utils.cluster_assignment import *
from utils.params import *
from utils.acc_ratio import *
import random


def solve():
    print(f'Hi,')
    r = get_reward(F, RF)
    reward = reward_feature(M, N, r[0, :]).reshape(X, 1)
    P = get_transitions(M, N, A, p, q, obstacles)
    V, V_hist, policy, time = policy_iteration(X, P, reward, A, gamma, max_iter=100)
    # plt = plot_3("A", "B", "C", V_hist[:, 0], V_hist[:, 5], M, N)
    plt.show()
    traj = gen_trajectories(X, P, policy, tl)
    return traj


def cluster():
    # data generation part
    traj_set=trajectory_set(F, RF, tn, tl)
    print("traj",traj_set[:,:,0])
    print("transformed", traj_form(traj_set[:,:,0]))

    # now we have generated nT number of optimal trajectories

    P = get_transitions(M, N, A, p, q, obstacles)
    #c = initialise_c(tl)
    r = sample_reward(F, mu, sigma, lb, ub)

    r2 = reward_feature(M, N, r).reshape(X, 1)
    v, V_hist, policy, time = policy_iteration(X, P, r2, A, gamma, max_iter=100)
    print(qfromv(v, r))
    #print(policy)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # solve()
    cluster()
   # c = np.array([2,5,3,1,2,4])
   # pr= count(c)
   # print("count", pr)
   # pr = assignment_prob(c, alpha)
   # print(pr)
   # g = Cluster()
   # g2 = init_cluster(g, tn, F, X, A)
   # g2.reward
