# -- coding:utf-8 --

import numpy as np
import networkx as nx
import pandas as pd



#'''Convert a networkx graph G to an adjacency list stored in a numpy array of shape (N,N+1)
#    in which row i stores the indices of the neighbors of node i followed by -1 values until the end of the row.
def convert_networkx_graph_to_adjacency_list(G):
    adj_list = - np.ones((G.number_of_nodes(), G.number_of_nodes()+1), dtype=int)
    index_dict = dict([(n, i) for i, n in enumerate(G.nodes())])
    for node in G.nodes():
        edges = G.edges(node)
        adj_list[index_dict[node],:len(edges)] = [index_dict[e[1]] for e in edges]
    return adj_list

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

def iteration_DB(graph, strategies, adjacent_cooperators, adjacent_defectors, update_node_index, random_number, w):
    
    previous_strategy = strategies[update_node_index]

# Sum up fitness of adjacent neighbors
    F_C = 0.001
    F_D = 0.001
    neighbors_index = graph[update_node_index, 0]
    i = 0
    while neighbors_index >= 0:
        f = fitness(strategies[neighbors_index], adjacent_cooperators[neighbors_index], adjacent_defectors[neighbors_index], w, F_C, F_D)
        if strategies[neighbors_index] == s1:
            F_C += f
        else:
            F_D += f
        i += 1
        neighbors_index = graph[update_node_index, i]

    # Birth step
    diff_n_cooperators = 0
    #print('random number', random_number)
    #print('F_c, F_d', F_C, F_D)
    if random_number < F_C / (F_C + F_D):
        strategies[update_node_index] = s1
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

def iteration_IM(graph, strategies, adjacent_cooperators, adjacent_defectors, update_node_index, random_number, w):
    previous_strategy = strategies[update_node_index]

# Sum up fitness of adjacent neighbors
    F_C = 0.
    F_D = 0.
    neighbors_index = graph[update_node_index, 0]
    i = 0
    while neighbors_index >= 0:
        f = fitness(strategies[neighbors_index], adjacent_cooperators[neighbors_index], adjacent_defectors[neighbors_index], w)
        # if strategies[neighbors_index]:
        if strategies[neighbors_index] == s1:
            F_C += f[0]
        else:
            F_D += f[1]
        i += 1
        neighbors_index = graph[update_node_index, i]

# Imitation

    diff_n_cooperators = 0
    f0 = fitness(strategies[update_node_index], adjacent_cooperators[update_node_index], adjacent_defectors[update_node_index], w)
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
    return diff_n_cooperators, strategies

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

def fitness(strategy, neighbor_c, neighbor_d, F_C, F_D, w):

    fitness = 1 - w + w * payoff(strategy, neighbor_c, neighbor_d, b, c, F_C, F_D)

    return fitness

def payoff(strategy, neighbor_c, neighbor_d, b, c, F_C, F_D):

    q_11 = repuation_rl(strategy, neighbor_c, neighbor_d, F_C, F_D)[0][0]
    q_12 = repuation_rl(strategy, neighbor_c, neighbor_d, F_C, F_D)[1][0]
    q_21 = repuation_rl(strategy, neighbor_c, neighbor_d, F_C, F_D)[2][0]
    q_22 = repuation_rl(strategy, neighbor_c, neighbor_d, F_C, F_D)[3][0]

    if strategy == s1:
        pi = (neighbor_c * q_11 + neighbor_d * q_21) * b \
                  - (neighbor_c * q_11 + neighbor_d * q_12) * c
    else:
        pi = (neighbor_c * q_12 + neighbor_d * q_22) * b \
                  - (neighbor_c * q_21 + neighbor_d * q_22) * c
     
    return pi


def repuation_rl(strategy, neighbor_c, neighbor_d, F_C, F_D):
    global q_i, d_s_q

    for i in range(4):
        if strategy == s1 and q_i[i][0] > 0.:
            d_s_q[i][0] = 1. * neighbor_c
        elif strategy == s2 and q_i[i][0] <= 0.:
            d_s_q[i][0] = 1. * neighbor_d
        elif strategy == s2 and q_i[i][0] > 0.:
            d_s_q[i][0] = - 1. * neighbor_d
        elif strategy == s1 and q_i[i][0] <= 0.:
            d_s_q[i][0] = - 1. * neighbor_c
    
    if strategy == s1:
        q_i = (1 - alpha) * q_i + alpha * (d_s_q / (neighbor_c + neighbor_d) + \
                                       delta_2 * (F_C / (F_C + F_D)) * (q_i + d_s_q / (neighbor_c + neighbor_d)))
    else:
        q_i = (1 - alpha) * q_i + alpha * (d_s_q / (neighbor_c + neighbor_d) + \
                                       delta_2 * (F_D / (F_C + F_D)) * (q_i + d_s_q / (neighbor_c + neighbor_d)))
    
    #print('q_i', q_i[0][0], q_i[1][0], q_i[2][0], q_i[3][0])

    return q_i


def run_until_fixation(graph, N, cooperator_index, iteration_type):

    strategies = []
    for i in range(N):
        strategies.append(s2)
    strategies[cooperator_index] = s1

    adjacent_cooperators = np.zeros(N, dtype=int)
    adjacent_defectors = np.zeros(N, dtype=int)
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
        random_number = np.random.uniform(0, 1)
        n_cooperators += iteration(graph, strategies, adjacent_cooperators, adjacent_defectors, update_node_index, \
                                   random_number, w)
    
        counter += 1
    
    return n_cooperators == N

def measure_fixation_probability(n_runs):
    global q_i, d_s_q

    C_fixations = 0
    for i in range(n_runs):
        q_i = np.array([[0.], [0.], [0.], [0.]])
        d_s_q = np.zeros((4, 1))
        
        first_cooperator_index = np.random.randint(0, N)
        if run_until_fixation(graph, N, first_cooperator_index, iteration_type):
            C_fixations += 1
            #print('cooperator fixation', C_fixations)
    return C_fixations / float(n_runs)

# initialize the parameters

iteration_type = 1
n_runs = 200
w = 0.01
c = 1.
la = 0
s1 = [la, 1., 1., 1.]
s2 = [la, 0., 0., 0.]

#q_i = np.array([[0.], [0.], [0.], [0.]])
#d_s_q = np.zeros((4, 1))
delta_2 = 0.98

df = pd.read_excel("Network input/High school temporal contacts.xlsx")
G = nx.from_pandas_edgelist(df, 'source', 'target')
average_degree = 2 * G.number_of_edges() / G.number_of_nodes()
N = G.number_of_nodes()
print('average degree, N', average_degree, N)

graph_adj_list = convert_networkx_graph_to_adjacency_list(G)
graph = np.mat(graph_adj_list)

alpha_list = [0.1, 0.2]
benefit_list = [average_degree / 5, average_degree / 4, average_degree / 3, \
                average_degree / 2, average_degree]

results = []
for b in benefit_list:
    print('b is', b)
    for alpha in alpha_list:
        for i in range(50):

            pho_c = measure_fixation_probability(n_runs)
            results.append(pho_c)

results = np.array(results)

results_df = pd.DataFrame(results)

results_df.to_excel('fig4_high_school.xlsx')




    







    
        
    

