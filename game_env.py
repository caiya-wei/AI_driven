from time import sleep
import gymnasium as gym
import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.special import softmax
import os
import pickle

def initialize_strategies(initial_cooperators, initial_defectors):
    strategies = np.array([1] * initial_cooperators + [0] * initial_defectors)
    np.random.shuffle(strategies)
    return strategies

class GameEnv(gym.Env):
    def __init__(self, b=20, dis=1):
        self.N = 50  # Population size
        self.initial_cooperators = int(self.N / 2)  # Initial number of cooperators
        self.initial_defectors = self.N - self.initial_cooperators  # Initial number of defectors
        self.b = b  # Benefit of cooperation
        self.c = 1  # Cost of cooperation
        self.w = 0.01  # Selection intensity
        self.k_mean = 4  # Mean degree
        self.neighbor_distance = dis  # Neighbor distance
        self.weight_extr = 0.5  # Weight of extrinsic reward
        self.weight_intr = 0.5  # Weight of intrinsic reward
        self.graph_path = "./G_pkl/graph_er_50.pkl"

        self.action_space = gym.spaces.MultiBinary(self.N)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.N + 2, self.N), dtype=np.int16)

        self.min_reward = float('inf')
        self.max_reward = float('-inf')
        
        self.load_or_create_graph()
        self.reset()

    def load_or_create_graph(self):
        if os.path.exists(self.graph_path):
            with open(self.graph_path, 'rb') as f:
                self.G_0 = pickle.load(f)
                # print(f"G mean degree is {sum([len(list(self.G_0.neighbors(node))) for node in self.G_0.nodes]) / self.N}")
        else:
            self.G_0 = nx.random_regular_graph(self.k_mean, self.N)
            with open(self.graph_path, 'wb') as f:
                pickle.dump(self.G_0, f)
    
    def play_game(self, node1, node2):
        strategy1 = self.strategies[node1]
        strategy2 = self.strategies[node2]

        if strategy1 == 1 and strategy2 == 1:
            return self.b - self.c  # Cooperation-Cooperation
        elif strategy1 == 1 and strategy2 == 0:
            return -self.c  # Cooperation-Defection
        elif strategy1 == 0 and strategy2 == 1:
            return self.b  # Defection-Cooperation
        else:
            return 0  # Defection-Defection

    def calculate_fitness(self, node):
        neighbors_node = np.where(self.G_adj_matrix[node] == 1)[0]
        payoffs = np.vectorize(self.play_game)(node, neighbors_node)
        fitness = 1 + self.w * np.sum(payoffs)
        return fitness

    def update_strategy_death_birth(self):
        neighbors_death = np.where(self.G_adj_matrix[self.death_node_index] == 1)[0]

        cooperators = [neighbor for neighbor in neighbors_death if self.strategies[neighbor] == 1]
        defectors = [neighbor for neighbor in neighbors_death if self.strategies[neighbor] == 0]

        F_C = sum(self.calculate_fitness(cooperator) for cooperator in cooperators)
        F_D = sum(self.calculate_fitness(defector) for defector in defectors)

        if F_C + F_D == 0:
            return

        self.probability_cooperator = F_C / (F_C + F_D)
        
        self.mean_cooperation_rate = np.mean(self.strategies)

        new_strategy = 1 if random.random() < self.probability_cooperator else 0
        self.strategies[self.death_node_index] = new_strategy

    def initialize_neighbors(self):
        self.initial_neighbors = {}
        for node in range(self.N):
            neighbors = set()
            for distance in range(1, self.neighbor_distance + 1):
                new_neighbors = np.where(np.linalg.matrix_power(self.G_adj_matrix, distance)[node] > 0)[0]
                neighbors.update(new_neighbors)
            neighbors.discard(node)
            self.initial_neighbors[node] = neighbors

    def get_neighbors(self):
        return self.initial_neighbors[self.death_node_index]

    def get_state(self):
        node_neighbors = np.zeros(self.N)
        node_neighbors[list(self.neighbors)] = 1
        state = np.vstack([np.array(self.G_adj_matrix), node_neighbors, self.strategies])
        return state

    def reset(self, **kwargs):
        self.probability_cooperator = 0.5
    
        self.strategies = initialize_strategies(self.initial_cooperators, self.initial_defectors)
        self.mean_cooperation_rate = np.mean(self.strategies)
        self.G = self.G_0
        self.G_adj_matrix = nx.adjacency_matrix(self.G).todense()
        self.initialize_neighbors()
        self.death_node_index = random.randint(0, self.N - 1)
        self.neighbors = self.get_neighbors()
        self.update_strategy_death_birth()
        state = self.get_state()
        return np.array(state), {}

    def step(self, action):
        self.update_G_adj_matrix(action)
        reward = self.get_reward()
        
        # Normalize the reward to [-1, 1] range
        if self.max_reward != self.min_reward:
            reward = 2 * (reward - self.min_reward) / (self.max_reward - self.min_reward) - 1
        else:
            reward = 0  # If min_reward equals max_reward, it means no variation in reward

        self.step_reward = reward

        self.death_node_index = random.randint(0, self.N - 1)
        self.neighbors = self.get_neighbors()
        self.update_strategy_death_birth()
        state = self.get_state()
        done = self.check_game_over()

        return np.array(state), reward, done, False, {}
    
    def check_game_over(self):
        if all(strategy == 1 for strategy in self.strategies) or all(strategy == 0 for strategy in self.strategies):
            return True
        return False

    def get_reward(self):
        reward_intr = 0
        reward_extr = 0
        for neighbor in self.neighbors:
            if self.strategies[self.death_node_index] == self.strategies[neighbor]:
                reward_extr += 1
            else:
                reward_extr -= 1
            
        reward_intr = self.calculate_causal_influence()

        reward = self.weight_extr * reward_extr + self.weight_intr * reward_intr 

        # Update min and max reward
        self.min_reward = min(self.min_reward, reward)
        self.max_reward = max(self.max_reward, reward)
        
        return reward

    def calculate_causal_influence(self):
        causal_influence = 0
        for neighbor in self.neighbors:
            if self.strategies[self.death_node_index] == 1:
                p_actual = self.compute_action_probability(neighbor)
                p_counterfactual = self.compute_counterfactual_probability(neighbor)
                causal_influence += self.kl_divergence(p_actual, p_counterfactual)
            else:
                p_actual = self.compute_action_probability(neighbor)
                p_counterfactual = self.compute_counterfactual_probability(neighbor)
                causal_influence -= self.kl_divergence(p_actual, p_counterfactual)
        return causal_influence

    def compute_action_probability(self, neighbor):
        # Placeholder for actual probability computation
        return softmax(np.random.random(self.N))

    def compute_counterfactual_probability(self, neighbor):
        # Placeholder for counterfactual probability computation
        return softmax(np.random.random(self.N))

    def kl_divergence(self, p, q):
        return np.sum(p * np.log(p / q))

    def is_graph_connected(self, G_adj_matrix_temp):
        visited = [False] * self.N
        start_node = next((i for i in range(self.N) if any(G_adj_matrix_temp[i])), None)
        if start_node is None:
            return False

        self._dfs(start_node, visited, G_adj_matrix_temp)
        return all(visited)

    def _dfs(self, node, visited, G_adj_matrix_temp):
        visited[node] = True
        for i in range(self.N):
            if G_adj_matrix_temp[node][i] == 1 and not visited[i]:
                self._dfs(i, visited, G_adj_matrix_temp)

    def update_G_adj_matrix(self, action):
        if self.death_node_index is None:
            return

        if not self.neighbors:
            return

        if np.sum(action) == 0:
            forced_neighbor = np.random.choice(list(self.neighbors))
            action[forced_neighbor] = 1

        action_neighbors = np.zeros(self.N, dtype=int)
        action_neighbors[list(self.neighbors)] = action[list(self.neighbors)]
        
        G_adj_matrix_temp = self.G_adj_matrix.copy()
        for neighbor in self.neighbors:
            if action_neighbors[neighbor] == 0:
                G_adj_matrix_temp[self.death_node_index, neighbor] = 0
                G_adj_matrix_temp[neighbor, self.death_node_index] = 0
                if not self.is_graph_connected(G_adj_matrix_temp):
                    action_neighbors[neighbor] = 1
                    G_adj_matrix_temp[self.death_node_index, neighbor] = 1
                    G_adj_matrix_temp[neighbor, self.death_node_index] = 1

        if np.sum(G_adj_matrix_temp[:, self.death_node_index]) == 0:
            forced_neighbor = np.random.choice(list(self.neighbors))
            action_neighbors[forced_neighbor] = 1

        for neighbor in self.neighbors:
            self.G_adj_matrix[self.death_node_index, neighbor] = action_neighbors[neighbor]
            self.G_adj_matrix[neighbor, self.death_node_index] = action_neighbors[neighbor]
        np.fill_diagonal(self.G_adj_matrix, 0)

    def draw_graph(self, iteration, pos):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('white')
        fig.set_size_inches(5, 3)
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

        ax.clear()

        G = nx.from_numpy_array(self.G_adj_matrix)
        nx.set_node_attributes(G, {i: self.strategies[i] for i in range(self.N)}, 'strategy')

        color_map = ['#1D5D9B' if G.nodes[node]['strategy'] == 1 else '#B83B5E' for node in G.nodes()]
        node_size = [(len(list(G.neighbors(node))) / self.k_mean) * 30 for node in G.nodes]

        for node, (x, y, z) in pos.items():
            ax.scatter(x, y, z, color=color_map[node], s=node_size[node], edgecolors='#219C90', linewidths=0.5)
        
        for edge in G.edges:
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]
            ax.plot([x0, x1], [y0, y1], [z0, z1], color='#A9A9A9', linewidth=0.6)

        ax.set_axis_off()

        plt.savefig(f"./graphs/graph_{iteration + 1}.svg", \
                            format='svg', bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()
        # plt.show()
