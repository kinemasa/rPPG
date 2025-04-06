# color_vector/independence.py

import numpy as np

def Make_Moment(K, res):
    M = np.empty([K, K])
    for m1 in range(0, K):
        for m2 in range(0, K - m1):
            E12 = np.mean((res[0, :] ** m1) * (res[1, :] ** m2))
            E1E2 = np.mean(res[0, :] ** m1) * np.mean(res[1, :] ** m2)
            M[m1, m2] = E12 - E1E2
    return M

def get_fuctorial(x):
    countfactT = 1
    for ifactT in range(1, x + 1):
        countfactT *= ifactT
    return countfactT

def get_weight(ga1, ga2, gb1, gb2):
    Sigma = 1
    G = 1
    if (ga1 + gb1) % 2 == 0:
        k = (ga1 + gb1) // 2
        J2k = (get_fuctorial(2 * k) * (2 * np.pi)**0.5) / ((4**k) * get_fuctorial(k) * Sigma**(2 * k - 1))
        sg = ((-1)**((ga1 - gb1) / 2) * J2k) / (get_fuctorial(ga1) * get_fuctorial(gb1))
        G *= sg
    else:
        G = 0
    if (ga2 + gb2) % 2 == 0:
        k = (ga2 + gb2) // 2
        J2k = (get_fuctorial(2 * k) * (2 * np.pi)**0.5) / ((4**k) * get_fuctorial(k) * Sigma**(2 * k - 1))
        sg = ((-1)**((ga2 - gb2) / 2) * J2k) / (get_fuctorial(ga2) * get_fuctorial(gb2))
        G *= sg
    else:
        G = 0
    return G

def fmin_Cal_Cost_Burel(K, M):
    CostGMM = 0
    for a1 in range(K):
        for a2 in range(K - a1):
            for b1 in range(K):
                for b2 in range(K - b1):
                    CostGMM += get_weight(a1, a2, b1, b2) * M[a1, a2] * M[b1, b2]
    return CostGMM

def f_burel(s, sensor):
    x1, y1 = np.cos(s[0]), np.sin(s[0])
    x2, y2 = np.cos(s[1]), np.sin(s[1])
    H = np.array([[x1, y1], [x2, y2]])
    res = H @ sensor
    K = 4
    M = Make_Moment(K, res)
    cost = fmin_Cal_Cost_Burel(K, M)
    return cost