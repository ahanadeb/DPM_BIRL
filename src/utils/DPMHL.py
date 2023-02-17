import numpy as np
from utils.util_functions import *
import random
from scipy.stats import dirichlet
from scipy.special import factorial
from utils.params import *
from utils.reward import *
from utils.transition import *
from utils.cluster_assignment import *

def dpmhl(C, maxiter):
    C = Cluster()
    C = init_cluster(C, tn, F, X, A)
    for i in range(0,maxiter):
        #first cluster update state
        x = np.random.randint(0, tn, size=(1, tn))[0]
        for m in x:
            C = update_cluster(C,m)
        C = relabel_cluster(C)

    return C