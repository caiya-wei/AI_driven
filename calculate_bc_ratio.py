# -- coding:utf-8 --

import numpy as np
import networkx as nx
import pandas as pd
# from networkx import Graph


# Define the parameters x_11, x_12, x_21, x_22, x = (x_11, x_12, x_21, x_22)^T
def good_probability(k_s1, s1, s2, d, eps):
    r1 = s1[2] - s1[3]
    r2 = s2[2] - s2[3]
    la1 = s1[0]
    la2 = s2[0]
    w_ij = 2. / (k_aver * (k_aver - 1))
    # m11=(1.*(n-2))/n + w_ij*r1 + \
    # w_ij*la1*(k-2)*r1*(1-2*eps) + w_ij*(n-2)*(1-la1)
    m11 = 1 - w_ij - la1 * (2. / k_aver - w_ij) + w_ij * r1 * (1. + la1 * (1. - 2. * eps) * (k_s1 - 2.))
    m12 = w_ij * la1 * r1 * (1. - 2. * eps) * (k_aver - k_s1)
    m13 = 0
    m14 = 0

    m21 = 0
    m22 = 1 - w_ij - la1 * (2. / k_aver - w_ij)
    m23 = w_ij * r1 + w_ij * la1 * r1 * (1. - 2. * eps) * (k_s1 - 1.)
    m24 = w_ij * la1 * r1 * (1. - 2. * eps) * (k_aver - k_s1 - 1.)

    m31 = w_ij * la2 * r2 * (1. - 2. * eps) * (k_s1 - 1.)
    m32 = w_ij * r2 + w_ij * r2 * la2 * (1. - 2 * eps) * (k_aver - k_s1 - 1.)
    m33 = 1 - w_ij - la2 * (2. / k_aver - w_ij)
    m34 = 0

    m41 = 0
    m42 = 0
    m43 = w_ij * la2 * r2 * k_s1 * (1. - 2. * eps)
    m44 = 1. - w_ij - la2 * (2. / k_aver - w_ij) + w_ij * r2 + w_ij * r2 * la2 * (1. - 2. * eps) * (k_aver - k_s1 - 2.)

    M = np.array([[m11, m12, m13, m14], [m21, m22, m23, m24], [m31, m32, m33, m34], [m41, m42, m43, m44]])

    # The vector v is given by Eq.56-57
    v = np.zeros((4, 1))
    for i in range(4):
        if i < 2:
            # v[i]=(la1*(n - 2)*w_ij*(s1[3] + eps*r1) + w_ij*s1[3])
            v[i] = (w_ij * s1[3] + la1 * (2. / k_aver - w_ij) * (eps * r1 + s1[3]))
        else:
            # v[i]=(la2*(n - 2)*w_ij*(s2[3] + eps*r2) + w_ij*s2[3])
            v[i] = (w_ij * s2[3] + la2 * (2. / k_aver - w_ij) * (eps * r2 + s2[3]))

    # x0=(y1,y2,y3,y4)T
    x0_1 = np.array([s1[1], s1[1], s2[1], s2[1]])
    # print("x0_1", x0_1)
    x0 = x0_1.reshape(4, 1)

    # 1-d*M
    matrixval = np.identity(4) - np.dot(d, M)

    # x=(x11,x12,x21,x22)=(1-dM)-1*((1-d)x0+dv)
    x = np.dot(np.linalg.pinv(matrixval), (np.dot((1 - d), x0) + np.dot(d, v)))
    # x=np.dot(np.linalg.inv(matrixval),v)

    #print('x is \n', x)

    return x

Runs = 10**5
results = []
for i in range(Runs):
    N = np.random.randint(100, 1000)
    k = np.random.randint(2, N/5)
    if k % 2 > 0:
        k += 1
    G = nx.watts_strogatz_graph(N, k, 0.1)
    degree = np.array([val for (node, val) in G.degree()])
    k_mean = np.mean(degree)
    j = np.random.randint(0, N)
    k_aver = degree[j]

# selection strength
    w_ij = 2. / (k_aver * (k_aver - 1.))
    w = 0.01
    la = 1
    delta = 0.5
    d = delta / (delta + (1 - delta) * w_ij)
#print('d is', d)
    eps = 0


#q0 = 1. - c / (delta * b)
#print("q0 is", q0)
#q1 = 1. - ((1. + (k_aver - 2.) * delta) / (1. + (k_aver - 2.) * (1. - 2. * eps))) * c / (delta * b)
#print("q1 is", q1)
#s3 = [la, 0.99, 0.99, q1]


    s1 = [la, 0.99, 0.99, 0.01]
    s2 = [la, 0.01, 0.01, 0.01]

    x = good_probability(1, s1, s2, d, eps)
#print("x is", x)
    x_11 = x[0]
    x_12 = x[1]
    x_21 = x[2]
    x_22 = x[3]

    #b_to_c_ratio = (1. + (k_aver - 1.) * x_12) / (x_12 - x_21) + np.random.uniform(-0.3, 0.3) * k_aver
    b_to_c_ratio = (1. + (k_aver - 1.) * x_12) / (x_12 - x_21)
    results.append(k_mean)
    results.append(b_to_c_ratio)
    results.append((k_mean + 1.) / (k_mean - 1.))
    results.append((b_to_c_ratio + 1.) / (b_to_c_ratio - 1.))

    if i % 1000 ==0:
        print('Runs are', i)
        




