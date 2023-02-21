import numpy as np
from utils.util_functions import *
import random
from scipy.stats import dirichlet
from scipy.special import factorial
from utils.params import *
from utils.reward import *
from utils.transition import *
from utils.cluster_assignment import *
from utils.update_weight import *
from tqdm import tqdm
from utils.saveHist import *
from utils.evd import *


def dpmhl(traj_set, maxiter,tn):
    # initialisations
    C = Cluster()
    C = init_cluster(C, tn, F, X, A)
    C = relabel_cluster(C,tn)
    pr = calDPMLogPost(traj_set, C)
    maxC = MaxC()
    hist = Hist()
    hist = init_h(hist)
    maxC.logpost = -np.inf
    maxC, hist, bUpdate, h = saveHist(C, pr, maxC, hist)
    print('init pr = ', pr)
    for i in tqdm(range(maxiter)):
        # first cluster update state
        x = np.random.randint(0, tn - 1, size=(1, tn))[0]
        for m in x:
            C = update_cluster(C, m, traj_set)
        C = relabel_cluster(C,tn)
        x = np.random.randint(0, int(np.max(C.assignment)) + 1, size=(1, int(np.max(C.assignment))))[0]
        for k in x:
            C = update_weight(k, traj_set, C)
        pr = calDPMLogPost(traj_set, C)
        maxC, hist, bUpdate, h = saveHist(C, pr, maxC, hist)
        print(i, 'th iteration, pr = ', pr, " ", maxC.logpost, " ", np.transpose(maxC.assignment))


    return maxC
