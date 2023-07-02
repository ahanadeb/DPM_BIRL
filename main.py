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
from utils.transition import *
from utils.evd import *



if __name__ == '__main__':
    EVD = []
    y = []
    i = 2
    drive=0
    r = get_reward(F, RF, drive)
    # r = r[1, :]
    # #print(r)
    # r = reward_feature(M, N, r).reshape(X, 1)
    # P1=get_transitions_r(M, N, A, r)
    traj_set_prev = 0
    rewards_prev = 0
    y_prev = 0

    while i < 8:
        e_avg=0
        maxiter = 1
        tn = RF * i
        y.append(int(tn))
        traj_set, rewards_gt, y2 = trajectory_set(F, RF,r,i-2, traj_set_prev,rewards_prev,y_prev, tn, tl,drive)
        print("y2", y2)
        maxC = dpmhl(traj_set, maxiter, tn, y2)
        e = evd(maxC, rewards_gt, tn)
        e_avg = e_avg + e / 1
        EVD.append(e)
        print("EVD = ", EVD)
        i = i + 1
        traj_set_prev = traj_set
        rewards_prev = rewards_gt
        y_prev = y2
    print("Completed. EVD = ", EVD)
    plt.plot(y, np.asarray(EVD))
    plt.xlabel('no. of trajectories per agent')
    plt.ylabel('Average EVD')
    plt.savefig('figure.png')
    plt.show()


