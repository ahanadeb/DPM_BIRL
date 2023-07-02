import numpy as np
import random
import pandas as pd


def get_reward(F, RF, drive):
    R = np.zeros((RF, F))
    p=0
    while p<RF:
        r = np.zeros((F,1))-1
        l = np.random.permutation(F)
        k = int(np.ceil(.1*F))
        idx = l[0:k]

        m= np.random.rand(k,1)-1


        j=0
        for i in idx:
            r[i,0] = m[j]
            j=j+1
        R[p,:] = np.transpose(r)
        if np.all(r==0):
            p=p-1
        p=p+1
    #print("orig reward", R)
    path = './rewards_mod.csv'
    if drive == 1:
        path = "/content/DPM_BIRL/rewards_mod.csv"
    df = pd.read_csv(path, sep=',', header=None)
    u=df.values

    print("rewards", u)
    #pd.DataFrame(R).to_csv("rewards_mod.csv", header=None, index=None)
    return u



def reward_feature(M, N, r):
    #r = np.transpose(r)
    reward = np.zeros((M, N))
    i = 1
    k = 0
    while i < M:
        j = 1
        while j < N:
            reward[i, j] = r[k]
            reward[i - 1, j] = r[k]
            reward[i, j - 1] = r[k]
            reward[i - 1, j - 1] = r[k]
            j = j + 2
            k = k + 1
        i = i + 2

    return reward
