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
from utils.DPMHL import *
import random


def solve():
    print(f'Hi,')
    r = get_reward(F, RF)
    reward = reward_feature(M, N, r[0, :]).reshape(X, 1)
    P = get_transitions(M, N, A, p, q, obstacles)
    V, V_hist, policy, time = policy_iteration(X, P, reward, A, gamma, max_iter=100)
    print(policy)
    policy = conv_policy(policy)
    print(policy)
    print(P.shape)
    # plt = plot_3("A", "B", "C", V_hist[:, 0], V_hist[:, 5], M, N)
    #plt.show()
    #traj = gen_trajectories(X, P, policy, tl)



def cluster():
    # data generation part
    traj_set = trajectory_set(F, RF, tn, tl)
    print("traj", traj_set[:, :, 0])
    print("transformed", traj_form(traj_set[:, :, 0]))

    # now we have generated nT number of optimal trajectories

    P = get_transitions(M, N, A, p, q, obstacles)
    # c = initialise_c(tl)
    r = sample_reward(F, mu, sigma, lb, ub)

    r2 = reward_feature(M, N, r).reshape(X, 1)
    v, V_hist, policy, time = policy_iteration(X, P, r2, A, gamma, max_iter=100)
    print(qfromv(v, r))
    # print(policy)


def test():
    C = Cluster()
    C = init_cluster(C, tn, F, X, A)
    print(C.reward.shape)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # solve()
    #dpmhl(maxiter=2)
    # c = np.array([2,5,3,1,2,4])
    # pr= count(c)
    # print("count", pr)
    # pr = assignment_prob(c, alpha)
    # print(pr)
    # g = Cluster()
    # g2 = init_cluster(g, tn, F, X, A)
    # g2.reward
    traj_set = trajectory_set(F, RF, tn, tl)
    #print("traj", traj_set[:, :, 0])
    #print("transformed", traj_form(traj_set[:, :, 0]))
    #t=[1,2]
    ##a=traj_form(traj_set[:, :, 0])
    #b=traj_form(traj_set[:, :, 1])
    #print(a)
    #print('b')
    #print(b)
    #t = [0,1]
    #s= traj_merj(traj_set, t)
    #print('s')
    #print(s)
    #print(np.sum(s[:,2]))

    solve()




