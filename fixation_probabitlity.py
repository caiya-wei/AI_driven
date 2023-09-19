import numpy as np
import networkx as nx
import pandas as pd
from scipy.sparse import csr_matrix, triu, diags, eye
from scipy.sparse.linalg import bicgstab
import random

def CartProd(mAdja, mAdjb):
    # Compute the Cartesian product of two graphs given by their weighted
    
    wa = np.array(mAdja.sum(axis=0))[0]
    wb = np.array(mAdjb.sum(axis=0))[0]

    triu_mAdja = triu(mAdja).tocoo()
    nodes1a, nodes2a = triu_mAdja.row, triu_mAdja.col
    weightsa = triu_mAdja.data

    triu_mAdjb = triu(mAdjb).tocoo()
    nodes1b, nodes2b = triu_mAdjb.row, triu_mAdjb.col
    weightsb = triu_mAdjb.data

    n = len(wa)
    en = len(nodes1a)
    m = len(wb)
    em = len(nodes1b)

    prodnodes1a = np.outer(nodes1a, np.ones(m, dtype=int)).ravel() + np.outer(np.ones(en, dtype=int), np.arange(n*m, step=n)).ravel()
    prodnodes2a = np.outer(nodes2a, np.ones(m, dtype=int)).ravel() + np.outer(np.ones(en, dtype=int), np.arange(n*m, step=n)).ravel()
    prodedgesa = np.outer(weightsa, wb).ravel()

    prodnodes1b = np.outer(n*nodes1b, np.ones(n, dtype=int)).ravel() + np.outer(np.ones(em, dtype=int), np.arange(n)).ravel()
    prodnodes2b = np.outer(n*nodes2b, np.ones(n, dtype=int)).ravel() + np.outer(np.ones(em, dtype=int), np.arange(n)).ravel()
    prodedgesb = np.outer(weightsb, wa).ravel()

    prodAdj = csr_matrix((prodedgesa, (prodnodes1a, prodnodes2a)), shape=(n*m, n*m)) \
             + csr_matrix((prodedgesb, (prodnodes1b, prodnodes2b)), shape=(n*m, n*m))

    # maximum between prodAdj and its transpose
    prodAdj = csr_matrix.maximum(prodAdj, prodAdj.transpose())

    return prodAdj


def NormLapl(mAdj):
    """
    Computes the normalized Laplacian of the graph given by the weighted
    adjacency matrix mAdj
    """
    # Ensure mAdj is a sparse matrix
    #if not isinstance(mAdj, csr_matrix):
    #    mAdj = csr_matrix(mAdj)
        
    # Make sure mAdj is symmetric
    mAdj = mAdj.maximum(mAdj.transpose())

    n = mAdj.shape[0]
    
    # Sum along the rows
    s = np.array(mAdj.sum(axis=1)).ravel()

    # Find non-zero elements (equivalent of MATLAB's find)
    nod1, nod2 = mAdj.nonzero()
    w = np.array(mAdj[nod1, nod2]).ravel()  # Convert matrix to 1-D array for consistency

    # Normalize weights
    norm_w = np.array([val / s[n1] for val, n1 in zip(w, nod1)])

    # Create sparse matrix
    L = csr_matrix((norm_w, (nod1, nod2)), shape=(n, n))

    # Subtract identity matrix
    L = L - eye(n)

    return L


def RemeetTimes(mAdj):
    """
    Computes remeeting times of the simple random walk on a graph
    given by its weighted adjacency matrix mAdj
    """

    # Make sure mAdj is symmetric
    mAdj = mAdj.maximum(mAdj.transpose())

    n = mAdj.shape[0]
    C = CartProd(mAdj, mAdj)

    s = np.array(C.sum(axis=0))[0]

    L = diags(s, 0, (n*n, n*n)) - C

    LL = NormLapl(mAdj)

    diagcoords = np.arange(n**2, step=n+1)
    othercoords = np.delete(np.arange(n**2), diagcoords)
    

    s = s[othercoords]
    L = L[othercoords][:,othercoords]

    remTime = np.zeros((n, n))

    try:
        result, info = bicgstab(L, s, maxiter = 1000)
        if info == 0:
            remTime[othercoords // n, othercoords % n] = result

        else:
            print("The solver did not converge, error code", info)
    except np.linalg.LinAlgError:
        print("The matrix L is singular, can't find unique solution")

    LL_remtimes = LL @ remTime + remTime @ LL.T
    remTime += np.diag(np.diag(1 + 0.5 * LL_remtimes))
    
    return remTime

def good_probability(k_s1, s1, s2, d, eps):
    r1 = s1[2] - s1[3]
    r2 = s2[2] - s2[3]
    la1 = s1[0]
    la2 = s2[0]
    
    m11 = 1 - theta - la1 * (theta_bar - theta) + theta * r1 * (1. + la1 * (1. - 2. * eps) * (k_s1 - 2.))
    m12 = theta * la1 * r1 * (1. - 2. * eps) * (k_mean - k_s1)
    m13 = 0
    m14 = 0

    m21 = 0
    m22 = 1 - theta - la1 * (theta_bar - theta)
    m23 = theta * r1 + theta * la1 * r1 * (1. - 2. * eps) * (k_s1 - 1.)
    m24 = theta * la1 * r1 * (1. - 2. * eps) * (k_mean - k_s1 - 1.)

    m31 = theta * la2 * r2 * (1. - 2. * eps) * (k_s1 - 1.)
    m32 = theta * r2 + theta * r2 * la2 * (1. - 2 * eps) * (k_mean - k_s1 - 1.)
    m33 = 1 - theta - la2 * (theta_bar - theta)
    m34 = 0

    m41 = 0
    m42 = 0
    m43 = theta_bar * la2 * r2 * k_s1 * (1. - 2. * eps)
    m44 = 1. - theta - la2 * (theta_bar - theta) + theta * r2 + theta * r2 * la2 * (1. - 2. * eps) * (k_mean - k_s1 - 2.)

    M = np.array([[m11, m12, m13, m14], [m21, m22, m23, m24], [m31, m32, m33, m34], [m41, m42, m43, m44]])

    # The vector v is given by Eq.56-57
    v = np.zeros((4, 1))
    for i in range(4):
        if i < 2:
            # v[i]=(la1*(n - 2)*w_ij*(s1[3] + eps*r1) + w_ij*s1[3])
            v[i] = (theta * s1[3] + la1 * (theta_bar - theta) * (eps * r1 + s1[3]))
        else:
            # v[i]=(la2*(n - 2)*w_ij*(s2[3] + eps*r2) + w_ij*s2[3])
            v[i] = (theta * s2[3] + la2 * (theta_bar - theta) * (eps * r2 + s2[3]))

    # x0=(u1,u2,u3,u4)T
    x0_1 = np.array([s1[1], s1[1], s2[1], s2[1]])
    x0 = x0_1.reshape(4, 1)

    # 1-d*M
    matrixval = np.identity(4) - np.dot(d, M)

    # x=(x11,x12,x21,x22)=(1-dM)-1*((1-d)x0+dv)
    x = np.dot(np.linalg.pinv(matrixval), (np.dot((1 - d), x0) + np.dot(d, v)))

    return x

def FixProba(mAdj, mInt=None, Rem=None):
    """
    Computes the critical b/c ratio for a graph given by weighted 
    (replacement) adjacency matrix mAdj.
    An optional argument represents a distinct interaction matrix mInt
    Another optional argument can be used to pass a precomputed matrix of
    remeeting times. This is useful for analyzing different interaction
    matrices with the same replacement matrix
    """
    n = mAdj.shape[0]
    w = np.array(mAdj.sum(axis=1)).ravel()
    W = np.sum(w)
    pi = (w / W).T

    if Rem is None:
        Rem = RemeetTimes(mAdj)
    else:
        Rem = Rem
    Rem = Rem - np.diag(np.diag(Rem))

    if mInt is None:
        mInt = mAdj
    else:
        mInt = mInt

    P10 = NormLapl(mAdj).toarray() + np.eye(n)
    P01 = NormLapl(mInt).toarray() + np.eye(n)
    P20 = P10.dot(P10)
    P21 = P20.dot(P01)

    T20 = np.sum((P20 * Rem).T, axis = 0).dot(pi)
    T21 = np.sum((P21 * Rem).T, axis = 0).dot(pi)
    T01 = np.sum((P01 * Rem).T, axis = 0).dot(pi)

    x = good_probability(1, s1, s2, d, eps)
    x_12 = x[1][0]
    x_21 = x[2][0]

    #pho = (T20 * x_12) / ((T21 - T01) * (1 - x_21))
    pho_c = 1 / n + (delta / (2*n)) * (-c * T20 * x_12 + b * (T21 - T01) * (1 - x_21)) 

    return pho_c


results = []

list_b = np.linspace(1, 20, 50) 

for b in list_b:
    for runs in range(50):
        monte_carlo = []
        N = 100
        k = 5
        #p = np.random.uniform(0, 0.05)
        G = nx.barabasi_albert_graph(N, k)
        M_adj = nx.adjacency_matrix(G)
        degree = np.array([val for (node, val) in G.degree()])
        k_mean = np.mean(degree)
        #print('k_mean is', k_mean)

        la = 0
        eps = 0
        theta = np.sum(degree) / (N * (N - 1))
        theta_bar = np.sum(degree) / N
        d_c = 0.5
        d = d_c / (d_c + (1 - d_c) * theta)
        delta = 0.01
        c = 1.

        s1 = [la, 0.99, 0.01, 0.99]
        s2 = [la, 0.01, 0.01, 0.01]
     
        pho = FixProba(M_adj)
        monte_carlo.append(pho)
    
    print('b is', b)
    results.append(b)
    results.append(np.mean(monte_carlo))
    results.append(np.std(monte_carlo))
        
results = np.array(results)
results = results.reshape(-1, 3)

cloumn = ['b', 'pho mean', 'pho std']
results_df = pd.DataFrame(data=results, columns=cloumn)
results_df2 = results_df.dropna(axis=0, how='any')
results_df2.to_excel('fig4_ba_wsls_10.xlsx') 



