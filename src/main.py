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
import sys


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

def new():
    C = Cluster()
    C = init_cluster(C, tn, F, X, A)
    traj_set = trajectory_set(F, RF, tn, tl)
    C = relabel_cluster(C)
    pr = calDPMLogPost(traj_set, C)
    print('pr',pr)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #np.set_printoptions(threshold=np.inf)
    EVD=[]
    y=[]
    for i in range(3,8):
        maxiter=5
        tn = 3*i
        y.append(int(tn))
        traj_set, rewards_gt = trajectory_set(F, RF, tn, tl)
        maxC = dpmhl(traj_set,maxiter,tn)
        e = evd(maxC, rewards_gt, maxiter,tn)
        EVD.append(e)
        print("EVD = ", EVD)
    print("Completed. EVD = ", EVD)
    plt.plot(y, np.asarray(EVD))
    plt.xlabel('no. of trajectories')
    plt.ylabel('EVD for the new trajectory')
    plt.savefig('figure.png')
    plt.show()

    #traj_set, rewards_gt = trajectory_set(F, RF, tn, tl)
    #print(rewards_gt.shape)
    #new()



