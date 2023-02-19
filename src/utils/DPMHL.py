import numpy as np
from utils.util_functions import *
import random
from scipy.stats import dirichlet
from scipy.special import factorial
from utils.params import *
from utils.reward import *
from utils.transition import *
from utils.cluster_assignment import *


def dpmhl(maxiter):
    C = Cluster()
    C = init_cluster(C, tn, F, X, A)
   # traj_set = trajectory_set(F, RF, tn, tl)
    #C.assignment = np.array((2,6,3,1,2,4))-1
    C = relabel_cluster(C)
    for i in range(0, maxiter):
        # first cluster update state
        x = np.random.randint(0, tn, size=(1, tn))[0]
        for m in x:
            C = update_cluster(C, m, traj_set)
        C = relabel_cluster(C)
        x = np.random.randint(0, np.max(C.assignment)+1, size=(1, np.max(C.assignment)))[0]
        for k in x:
            C = update_weight(k,traj_set, C)


    return C
