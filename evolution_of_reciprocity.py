# -- coding:utf-8 --

import numpy as np
import networkx as nx
import pandas as pd
# from networkx import Graph

# Functions to generate networkx-graphs

# Create a regular square lattice graph with N1*N2 nodes and k=4 edges per node
def create_square_lattice_graph4(N1, N2, periodic = True):
    g = nx.grid_graph(dim=[N1, N2], periodic=periodic)
    return g

# Create a regular square lattice graph with N1*N2 nodes and k=6 edges per node
def create_square_lattice_graph6(N1, N2, periodic = True):
    g = nx.grid_graph(dim=[N1, N2], periodic=periodic)
    for n in g.nodes():
        g.add_edge(n, ((n[0]+1) % N1, (n[1]+1) % N2))
    return g

# Create a regular triangular lettice graph with N1*N2 nodes and k=8 edges per node
def create_triangular_lattice_graph(N1, N2):
    g = nx.grid_graph(dim=[N1, N2], periodic=True)
    for i in range(N1):
        for j in range(N2):
            g.add_edge((i, j), ((i+1)%N1, (j+1)%N2))
            g.add_edge(((i+1)%N1, j), (i, (j+1)%N2))
    return g

# Create a regular triangular lettice graph with N1*N2 nodes and k=10 edges per node
def create_square_lattice_graph10(N1, N2, periodic = True):
    g = create_triangular_lattice_graph(N1, N2)
    for n in g.nodes():
        g.add_edge(n, ((n[0]+1) % N1, (n[1]+1) % N2))
    return g

#Create a regular hexagonal lattice graph with int(N1*N2*2/3.0) nodes and k=3 edges per node.
#    Example parameters:
#    N1=10 and N2=15 will produce a grid with N=100 nodes.
#    N1=25 and N2=30 will produce a grid with N=500 nodes.
#    Note:
#    For certain parameters it happens that some nodes on the boundary of the grid have two or four neighbors. The function will print a warning in these cases.

def create_hexagonal_lattice_graph(N1, N2):

    # raise ValueError("These values are not allowed.")
    if N1<3 or N2<3 or N1%3==2 or N2%3==2:
        print("Warning: With these values N1 and N2, there may be some nodes on the boundary of the lattice with 2 or 4 edges.\
        If you prefer all nodes having exactly three edges, choose values which do not satisfy N1<3 or N2<3 or N1%3==2 or N2%3==2.")
    g = create_square_lattice_graph6(N1, N2, periodic=True)
    for i in range(N1):
        for j in range(N2):
            if (i+j)%3 == 0:
                g.remove_node((i%N1, j%N2))
            if i == 0 and j%3 == 2:
                g.remove_edge((i,j), (N1-1,j-1))
            if i%3 == 2 and j == 0:
                u,v = (i, j), (i-1, N2-1)
                if g.has_edge(u, v):
                    g.remove_edge(u, v)
    return g

#Create a regular hexagonal lattice graph with int N1*N2 nodes and k=2 edges per node.
def create_square_lattice_graph2(N1, N2):

    g = nx.grid_graph(dim=[N1, N2], periodic=True)
    for i in range(N1):
        for j in range(N2):
            if (i+j)%2 == 0:
                g.remove_node((i%N1, j%N2))
            if i == 0 and j%2 == 2:
                g.remove_edge((i,j), (N1-1,j-1))
            if i%2 == 2 and j == 0:
                u,v = (i, j), (i-1, N2-1)
                if g.has_edge(u, v):
                    g.remove_edge(u, v)
    return g

#Create a connected random graph with N nodes and k edges per node on average
#Algorithm:
#First, add nodes and connect them to already connected nodes to ensure that the graph is connected.
#Second, add edges between randomly chosen pairs of nodes until the desired number of edges is present.

#'''Convert a networkx graph G to an adjacency list stored in a numpy array of shape (N,N+1)
#    in which row i stores the indices of the neighbors of node i followed by -1 values until the end of the row.
def convert_networkx_graph_to_adjacency_list(G):
    adj_list = - np.ones((G.number_of_nodes(), G.number_of_nodes()+1), dtype=np.int)
#    index_dict = dict([(n, i) for i, n in enumerate(G.nodes_iter())])
    index_dict = dict([(n, i) for i, n in enumerate(G.nodes())])
    for node in G.nodes():
        edges = G.edges(node)
        adj_list[index_dict[node],:len(edges)] = [index_dict[e[1]] for e in edges]
    return adj_list


# Function to measure fixation probabilities
''' For each value in benefit_to_cost_ratios, do n_graphs*n_runs simulations to measure the fixation probabilities.

    Parameters:
    graph_generator: Function which returns a graph as an adjacency list. This adjacency list should be a numpy array of shape (N,N+1) in which row i stores the indices of the neighbors of node i followed by -1 values until the end of the row.
    graph_generator_arguments: List of arguments passed to the graph_generator function.
    w:  Selection strength.
    c:  Costs.
    n_graphs: The number of graphs generated with graph_generator.
    n_runs: The number of simulations on each graph.
    benefit_to_cost_ratios: List of b/c ratios.
    dview (optional): ipyparallel.client.view.DirectView object for parallelization. The default value is None, meaning no parallelization.
    iteration_type (optional): Either 1 (death-birth process) or 0 (imitation process). Default is 1. 

    Returns:
    Fixation probabilities as a numpy-array of the same length as benefit_to_cost_ratios.
    '''

'''Measure fixation probability by running many simulations starting with a random node as the initial cooperator.

Parameters:
graph_nd: The graph given as a numpy array of shape (N,N+1) in which row i stores the indices of the neighbors of node i followed by -1 values until the end of the row.
n_runs: Number of simulations.
b: Benefit value.
c: Cost value.
w: Selection strength.
seed: Seed for the random number generator.
iteration_type: Either 1 (death-birth process) or 0 (imitation process).

Returns:
The ratio of the number of simulations that ended with a fixation of cooperators.
'''
def measure_fixation_probability(graph_generator, n_runs, b, c, w, iteration_type):

    graph = np.mat(graph_generator)
    N = graph.shape[0]

    C_fixations = 0
    for i in range(n_runs):
        print("iteration step is", i)
        first_cooperator_index = np.random.randint(0, N)
        if run_until_fixation(graph, N, first_cooperator_index, b, c, w, iteration_type):
            C_fixations += 1
    return C_fixations / float(n_runs)

# Algorithms to simulate games on graphs
'''Run a simulation until all nodes have the same strategy. In the initial state all nodes defect except the one that is specified with cooperator_index.

Parameters:
graph: The graph given as a numpy array of shape (N,N+1) in which row i stores the indices of the neighbors of node i followed by -1 values until the end of the row.
N: Size of the graph.
cooperator_index: Index of the initial cooperator.
b: Benefit value.
c: Cost value.
w: Selection strength.
iteration_type: Either 1 (death-birth process) or 0 (imitation process).

Returns:
True or false, depending on whether the fixation of cooperators was successful.  
'''
def run_until_fixation(graph, N, cooperator_index, b, c, w, iteration_type):
    #i = 0

#    uniform_int_distribution[int] dist1 = uniform_int_distribution[int](0, N - 1)
#    uniform_real_distribution[double] dist2 = uniform_real_distribution[double](0.0, 1.0)

# Initialization

    # strategies = np.zeros(N, dtype=np.int)
    # strategies[cooperator_index] = 1

    strategies = []
    for i in range(N):
        strategies.append(s2)
    strategies[cooperator_index] = s1

    #neighbors_index = []
    adjacent_cooperators = np.zeros(N, dtype=np.int)
    adjacent_defectors = np.zeros(N, dtype=np.int)
    for i in range(N):
        adj_cooperators = 0
        adj_defectors = 0
        j = 0
        neighbors_index = graph[i, j]
        while neighbors_index >= 0:
            # if strategies[neighbors_index]:
            #if sum(strategies[neighbors_index][1:]):
            if strategies[neighbors_index] == s1:
                adj_cooperators += 1
            else:
                adj_defectors += 1
            j += 1
            neighbors_index = graph[i, j]
        adjacent_cooperators[i] = adj_cooperators
        adjacent_defectors[i] = adj_defectors

# Simulation
    n_cooperators = 1
    counter = 0
    if iteration_type:  # death-birth
        iteration = iteration_DB
    else:  # imitation
        iteration = iteration_IM
    while n_cooperators > 0 and n_cooperators < N:
        update_node_index = np.random.randint(0, N)
        # print('update_node_index is', update_node_index)
        random_number = np.random.rand()
        n_cooperators += iteration(graph, strategies, adjacent_cooperators, adjacent_defectors, update_node_index, \
                                   random_number, b, c, w)
        # print('Number of cooperators are', n_cooperators)
        counter += 1
        # print('iteration step is', counter)
    return n_cooperators == N


'''Do one death-birth update.

Parameters:
graph: The graph given as a numpy array of shape (N,N+1) in which row i stores the indices of the neighbors of node i followed by -1 values until the end of the row.
strategies: Integer array of shape (N) which stores the strategy of each node. If node i is a cooperator(defector), then strategies[i] is 1(0).
adjacent_cooperators: Integer array of shape (N) which stores the number of adjacent cooperators of each node.
adjacent_defectors: Integer array of shape (N) which stores the number of adjacent defectors of each node.
update_node_index: Index of the node where the death and the birth event happen.
random_number: A float value between 0 and 1.
b: Benefit value.
c: Cost value.
w: Selection strength.

Returns:
-1, 0 or +1, depending on whether the number of cooperators decreased by one, stayed constant, or increased by one.
'''
def iteration_DB(graph, strategies, adjacent_cooperators, adjacent_defectors, update_node_index, random_number, b, c, w):
    previous_strategy = strategies[update_node_index]

# Sum up fitness of adjacent neighbors
    F_C = 0.00001
    F_D = 0.00001
    neighbors_index = graph[update_node_index, 0]
    i = 0
    while neighbors_index >= 0:
        f = fitness(strategies[neighbors_index], adjacent_cooperators[neighbors_index], adjacent_defectors[neighbors_index], b, c, w)
        # if strategies[neighbors_index]:
        if strategies[neighbors_index] == s1:
            F_C += f
        else:
            F_D += f
        i += 1
        neighbors_index = graph[update_node_index, i]

    # Birth step
    diff_n_cooperators = 0
    if random_number < F_C / (F_C + F_D):
        # print('random number is', random_number)
        # print('fc/(fc+fd) is', F_C / (F_C + F_D))
        strategies[update_node_index] = s1
        # if not previous_strategy:
        if not previous_strategy == s1:
            diff_n_cooperators = 1
    else:
        # print('random number is', random_number)
        # print('fc/(fc+fd) is', F_C / (F_C + F_D))
        strategies[update_node_index] = s2
        # if previous_strategy:
        if previous_strategy == s1:
            diff_n_cooperators = -1

# Update the stored information about the neighbors of each node
    if diff_n_cooperators != 0:
        i = 0
        while True:
            neighbors_index = graph[update_node_index, i]
            if neighbors_index < 0:
                break
            adjacent_cooperators[neighbors_index] += diff_n_cooperators
            adjacent_defectors[neighbors_index] -= diff_n_cooperators
            i += 1

    return diff_n_cooperators

'''Do one imitation step.

Parameters:
graph: The graph given as a numpy array of shape (N,N+1) in which row i stores the indices of the neighbors of node i followed by -1 values until the end of the row.
strategies: Integer array of shape (N) which stores the strategy of each node. If node i is a cooperator(defector), then strategies[i] is 1(0).
adjacent_cooperators: Integer array of shape (N) which stores the number of adjacent cooperators of each node.
adjacent_defectors: Integer array of shape (N) which stores the number of adjacent defectors of each node.
update_node_index: Index of the node where the death and the birth event happen.
random_number: A float value between 0 and 1.
b: Benefit value.
c: Cost value.
w: Selection strength.

Returns:
-1, 0 or +1, depending on whether the number of cooperators decreased by one, stayed constant, or increased by one.
'''
def iteration_IM(graph, strategies, adjacent_cooperators, adjacent_defectors, update_node_index, random_number, b, c, w):
    previous_strategy = strategies[update_node_index]

# Sum up fitness of adjacent neighbors
    F_C = 0.00001
    F_D = 0.00001
    neighbors_index = graph[update_node_index, 0]
    i = 0
    while neighbors_index >= 0:
        f = fitness(strategies[neighbors_index], adjacent_cooperators[neighbors_index], adjacent_defectors[neighbors_index], b, c, w)
        # if strategies[neighbors_index]:
        if strategies[neighbors_index] == s1:
            F_C += f
        else:
            F_D += f
        i += 1
        neighbors_index = graph[update_node_index, i]

# Imitation

    diff_n_cooperators = 0
    f0 = fitness(strategies[update_node_index], adjacent_cooperators[update_node_index], adjacent_defectors[update_node_index], b, c, w)
    # if not previous_strategy and random_number < F_C / (F_C + F_D + f0):
    if not previous_strategy == s1 and random_number < F_C / (F_C + F_D + f0):
        strategies[update_node_index] = s1
        diff_n_cooperators = 1
    # elif previous_strategy and random_number < F_D / (F_C + F_D + f0):
    elif previous_strategy == s1 and random_number < F_D / (F_C + F_D + f0):
        strategies[update_node_index] = s2
        diff_n_cooperators = -1

# Update the stored information about the neighbors of each node
    if diff_n_cooperators != 0:
        i = 0
        while True:
            neighbors_index = graph[update_node_index, i]
            if neighbors_index < 0:
                break
            adjacent_cooperators[neighbors_index] += diff_n_cooperators
            adjacent_defectors[neighbors_index] -= diff_n_cooperators
            i += 1
    return diff_n_cooperators

'''Calculate the fitness of a node. 

Parameters:
strategy: Integer value, 1 (0) if the node is a cooperator (defector).
n_cooperators: The number of adjacent nodes that cooperate.
n_defectors: The number of adjacent nodes that defect.
b: Benefit value.
c: Cost value.
w: Selection strength.
'''
# Cooperators adopt the strategy s1, while defectors adopt s2

def fitness(strategy, n_cooperators, n_defectors, b, c, w):
    x_11 = good_probability(n_cooperators, n_defectors, s1, s2, d, eps)[0]
    x_12 = good_probability(n_cooperators, n_defectors, s1, s2, d, eps)[1]
    x_21 = good_probability(n_cooperators, n_defectors, s1, s2, d, eps)[2]
    x_22 = good_probability(n_cooperators, n_defectors, s1, s2, d, eps)[3]
    # if strategy:
    if strategy == s1:
        # payoff = b * n_cooperators - c * (n_cooperators + n_defectors)
        payoff = ((n_cooperators - 1) * x_11 + n_defectors * x_21) * b \
                  - ((n_cooperators - 1) * x_11 + n_defectors * x_12) * c
    else:
        # payoff = b * n_cooperators
        payoff = (n_cooperators * x_12 + (n_defectors - 1) * x_22) * b \
                  - (n_cooperators * x_21 + (n_defectors - 1) * x_22) * c
    return 1 - w + w * payoff

# Define the parameters x_11, x_12, x_21, x_22, x = (x_11, x_12, x_21, x_22)
def good_probability(n_cooperators, n_defectors, s1, s2, d, eps):
    r1 = s1[2] - s1[3]
    r2 = s2[2] - s2[3]
    la1 = s1[0]
    la2 = s2[0]
    num_k = n_cooperators + n_defectors

    # m11=(1.*(n-2))/n + w_ij*r1 + \
    # w_ij*la1*(k-2)*r1*(1-2*eps) + w_ij*(n-2)*(1-la1)
    m11 = 1. - w_ij - la1 * (2. / num_k - w_ij) + w_ij * r1 * (1. + la1 * (1. - 2. * eps) * (n_cooperators - 2.))
    m22 = 1. - w_ij - la1 * (2. / num_k - w_ij)
    m33 = 1. - w_ij - la2 * (2. / num_k - w_ij)
    m44 = 1. - w_ij - la2 * (2. / num_k - w_ij) + w_ij * r2 + w_ij * r2 * la2 * (1. - 2. * eps) * (num_k - n_cooperators - 2.)
    m12 = w_ij * la1 * (num_k - n_cooperators) * r1 * (1. - 2. * eps)
    m13 = 0
    m14 = 0
    m21 = 0
    # m22=(1.*(n-2))/n+w_ij*(n-2)*(1-la1)
    # m22 = 1 - w_ij - la1*(2./n-w_ij)
    m23 = w_ij * r1 + w_ij * la1 * (n_cooperators - 1.) * r1 * (1. - 2. * eps)
    m24 = w_ij * la1 * r1 * (num_k - n_cooperators - 1.) * (1. - 2. * eps)
    m31 = w_ij * la2 * r2 * (n_cooperators - 1.) * (1. - 2. * eps)
    m32 = w_ij * r2 + w_ij * la2 * (num_k - n_cooperators - 1.) * r2 * (1. - 2. * eps)
    # m33 = w_ij*(n-2)*(1-la2)+(1.*(n-2))/n

    m34 = 0
    m41 = 0
    m42 = 0
    m43 = w_ij * la2 * n_cooperators * r2 * (1. - 2. * eps)
    # m44=(1.*(n - 2))/n + w_ij*(n - 2)*(1 - la2) +\
    # w_ij*r2 + w_ij*la2*(n-k-2)*r2*(1-2*eps)

    M = np.array([[m11, m12, m13, m14], [m21, m22, m23, m24], \
                  [m31, m32, m33, m34], [m41, m42, m43, m44]])


    # The vector v is given by Eq.56-57
    v = np.zeros((4, 1))
    for i in range(4):
        if i < 2:
            # v[i]=(la1*(n - 2)*w_ij*(s1[3] + eps*r1) + w_ij*s1[3])
            v[i] = (w_ij * s1[3] + la1 * (2. / num_k - w_ij) * (eps * r1 + s1[3]))
        else:
            # v[i]=(la2*(n - 2)*w_ij*(s2[3] + eps*r2) + w_ij*s2[3])
            v[i] = (w_ij * s2[3] + la2 * (2. / num_k - w_ij) * (eps * r2 + s2[3]))

    # x0=(y1,y2,y3,y4)T
    x0_1 = np.array([s1[1], s1[1], s2[1], s2[1]])
    x0 = x0_1.reshape(4, 1)

    # 1-d*M
    matrixval = np.identity(4) - np.dot(d, M)

    # x=(x11,x12,x21,x22)=(1-dM)-1*((1-d)x0+dv)
    x = np.dot(np.linalg.pinv(matrixval), (np.dot((1 - d), x0) + np.dot(d, v)))
    # x=np.dot(np.linalg.inv(matrixval),v)

    #print('x is \n', x)

    return x

# graph_generator_arguments = nx.random_regular_graph(2, 100)
# graph_generator_arguments = nx.cycle_graph(100)
N_t = 500
# selection strength
iteration_type = 1
w = 0.01
c = 1
n_runs = 10 ** 4
la = 1
delta = 0.5
eps = 0.
s1 = [la, 0.99, 0.99, 0.01]
s2 = [la, 0.01, 0.01, 0.01]
benefit_to_cost_ratios = np.linspace(1, 20, 8, dtype=np.uint32)
k_aver_list = [4, 6, 8, 10]

results = []
for k_aver in k_aver_list:
    G = nx.random_regular_graph(k_aver, N_t)
    graph_adj_list = convert_networkx_graph_to_adjacency_list(G)
    w_ij = 2. / (k_aver * (k_aver - 1.))
    d = delta / (delta + (1. - delta) * w_ij)
    for bc_ratio in benefit_to_cost_ratios:
        b = c * bc_ratio
        pho = measure_fixation_probability(graph_adj_list, n_runs, b, c, w, iteration_type)
        results.append(pho)



