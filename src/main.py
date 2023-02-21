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



if __name__ == '__main__':
    EVD = []
    y = []
    i = 2
    while i < 11:
        maxiter = 10
        tn = RF * i
        y.append(int(tn))
        traj_set, rewards_gt = trajectory_set(F, RF, tn, tl)
        maxC = dpmhl(traj_set, maxiter, tn)
        e = evd(maxC, rewards_gt, maxiter, tn)
        EVD.append(e)
        print("EVD = ", EVD)
        i = i + 2
    print("Completed. EVD = ", EVD)
    plt.plot(y, np.asarray(EVD))
    plt.xlabel('no. of trajectories per agent')
    plt.ylabel('EVD for the new trajectory')
    plt.savefig('figure.png')
    plt.show()


