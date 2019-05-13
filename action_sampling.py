import numpy as np
import random
from sklearn import preprocessing
import copy
import math

def sample_persistent_action(dim, H, A_p, neta):
    """
    params:
    - dim: the dimension of the action vector
    - H: the H vector (history vector)
    - neta: the value sampled from a normal
    - A_p: previous action vector
    returns:
    - B: a unit vector of norm 1 such that H.B = 1
    """
    d = dim
    j = 0
    K = 0.0
    W = list(range(dim))

    H = H.reshape(-1,)
    # create an empty B here
    D = np.zeros(dim)

    # shuffle the dimension
    random.shuffle(W)
    # go over each index
    for elem in range(d-1):
        i = W[elem]
        j += 1
        # draw a random number D_{i} \sim U[-1,-1]
        # draw a number for B
        D[i] = np.random.uniform(-1, 1)
        # update the K buffer
        K -= (D[i] * H[i])
    b = W[-1]
    if H[b] == 0:
        raise Exception("H[b] is zero!")
    # put the last element in
    D[b] = K/H[b]
    # get the L2 norm for H
    # H_norm = preprocessing.normalize(H, norm='l2')
    H_norm = np.linalg.norm(H)
    D_norm = np.linalg.norm(D)
    D = D * np.tan(neta) * (H_norm/D_norm)
    # print(np.dot(D.reshape(-1), H.reshape(-1)))
    E = A_p + D
    B = E - (A_p - H)
    # normalize B
    B = B/np.linalg.norm(B)

    return B




def sample_persistent_action_noHvector(dim, A_p, neta, max_action_limit):
    """
    params:
    - dim: the dimension of the action vector
    - neta: the value sampled from a normal
    - A_p: previous action vector
    - max_action_limit: the maximum action the environment allows
    returns:
        - A: A new action vector
    """
    d = dim
    P = np.around(np.random.uniform(-max_action_limit,+max_action_limit,(d,)),decimals=10)

    A_p = np.asarray(A_p)

    D = np.dot(A_p,P)

    # The projection of P on A_p
    Vp = D/(np.linalg.norm(A_p,2)**2)*A_p

    # The direction of rotation
    Vr = P - Vp

    l = np.linalg.norm(Vp,2)*math.tan(neta)
    k = l / np.linalg.norm(Vr,2)

    # Rotated vector
    Q = k * Vr + Vp

    if D > 0:
        A = Q
    elif D < 0:
        A = -Q
    else:
        A = P
        print("Two consecutive same actions!")

    # Clip the action if out of range
    #A = np.clip(A,-max_action_limit,+max_action_limit)

    return A
