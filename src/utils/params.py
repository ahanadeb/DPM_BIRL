import numpy as np

M = 8  # no of rows
N = 8  # no of cols
X = M * N  # state space
gamma = 0.9  # discount factor
A = 4  # number of actions
p = 0.8  # success rate
q = 0.2
B = 2  # block size
F = int(np.power((N / B), 2))  # number of features
RF = 3  # number of reward functions to generate
tl = 40  # trajectory length
tn = 6  # number of trajectories
mu = 0  # for reward prior
sigma = 0.1  # for rewards prior
lb = -1  # lower bound
ub = 1  # upper bound
alpha = 1  # Dirichlet process prior
eta = 10  # confidence / inverse temperature
weight_iter=2 #iterations for updating weight

# defining obstacles
obstacles = np.zeros((M, N))  # no obstacles
# obstacles[1:5,4] = 1
# obstacles[5,2:5] = 1
