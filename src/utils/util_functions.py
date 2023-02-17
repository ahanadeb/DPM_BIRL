import numpy as np
from tqdm import tqdm
import time
from utils.policies import *


def trans(P, pi):  # constructs X by X transition matrix for policy pi
    X = np.size(P, 0)
    P_pi = np.zeros((X, X))
    for m in range(0, X):
        for n in range(0, X):
            P_pi[m][n] = np.dot(P[m][n], pi[m])
    return P_pi


def scalarize(m, n, M, N):  # turns a pair of (m,n) coordinates into a flat 1-dimensional representation x
    x = N * m + n
    return x


def vectorize(x, M, N):  # turns a flat 1-dimensional representation x into a pair of (m,n) coordinates
    n = np.mod(x, N)
    m = (x - n) // N
    return (m, n)


def move(m, n, delta_m, delta_n, M, N,
         obstacles):  # from a given grid position (m,n), move in direction (delta_m,delta_n) if possible
    mtr = np.maximum(np.minimum(m + delta_m, M - 1), 0)
    ntr = np.maximum(np.minimum(n + delta_n, N - 1), 0)
    if obstacles[mtr, ntr] == 1:
        return scalarize(m, n, M, N)
    else:
        return scalarize(mtr, ntr, M, N)


def vector_to_matrix(value_vector, M,
                     N):  # turns a function on the state space represented as a flat X-dimensional vector to an M by N matrix
    X = N * M
    value_matrix = np.zeros((M, N))
    for x in range(0, X):
        [m, n] = vectorize(x, M, N)
        value_matrix[m, n] = value_vector[x]
        if obstacles[m, n] == 1:
            value_matrix[m, n] = np.nan
    return value_matrix


def matrix_to_vector(
        value_matrix):  # turns a function on the state space represented as an M by N matrix into a flat X-dimensional vector
    [M, N] = [np.size(value_matrix, 0), np.size(value_matrix, 1)]
    X = N * M
    value_vector = np.zeros((X))
    for x in range(0, X):
        [m, n] = vectorize(x, M, N)
        value_vector[x] = value_matrix[m, n]
    return value_vector


def value_iteration(P, r, gamma, A, X, max_iter):
    start = time.time()
    V_hist = np.zeros((X, max_iter))
    print("Running value iteration for ", max_iter, " iterations-")
    V = np.zeros((X))
    for i in tqdm(range(max_iter)):
        V_new = np.zeros((V.shape[0]))
        for m in range(0, X):
            v = np.zeros((A))

            for a in range(0, A):
                v[a] = r[m, a] + gamma * (np.dot(P[m, :, a], V))

            V_new[m] = np.max(v)
        V = V_new
        V_hist[:, i] = V
    print("Elapsed time: ", (time.time() - start), " seconds.\n")
    return V, V_hist, time.time() - start


def evaluate_analytical(P, pi, r, gamma):  ### policy evaluation subroutine (analytical solution)
    X = np.size(P, 0)
    value = np.zeros((X))
    # print("P, ", P)
    P_pi = trans(P, pi)
    # print("P_pi", P_pi)
    I = np.eye(X)
    A = (I - gamma * P_pi)
    A_inverse = np.linalg.inv(A)
    r = np.sum(r * pi, axis=1)
    # print("R ", r)
    value = A_inverse.dot(r)
    return value


def evaluate_power_iter(P, pi, r, gamma):  ### policy evaluation subroutine (power iteration)
    X = np.size(P, 0)
    value = np.zeros((X))
    P_pi = trans(P, pi)
    r = np.sum(r * pi, axis=1)
    # for i in range(0, 10000):
    #    for m in range(0, X):
    #        for a in range(0, 2):
    #            value[m]=value[m]+pi[m,a]*np.dot(P_pi[m], (r[:,a]+gamma*value))

    for i in range(0, 100):
        for m in range(0, X):
            # value[m] = np.dot(P_pi[m], r) + gamma * (np.dot(P_pi[m], value))
            value[m] = r[m] + gamma * (np.dot(P_pi[m], value))
    return value


def policy_iteration(X, P, r, A, gamma, max_iter):
    start = time.time()
    V_hist = np.zeros((X, max_iter))
    policy = np.zeros((X, A))  # + .5
    V_new = np.zeros((X))
    V = np.zeros((X))
    print("Running policy iteration for ", max_iter, " iterations-")
    for i in tqdm(range(max_iter)):
        V = evaluate_analytical(P, policy, r, gamma)
        for m in range(0, X):
            v = np.zeros((A))
            for a in range(0, A):
                v[a] = r[m] + gamma * (np.dot(P[m, :, a], V))
            V_new[m] = np.max(v)

            policy[m] = np.zeros((A))
            policy[m, np.argmax(v)] = 1
        V = V_new
        V_hist[:, i] = V
    print("Elapsed time: ", (time.time() - start), " seconds.\n")
    return V, V_hist, policy, time.time() - start


