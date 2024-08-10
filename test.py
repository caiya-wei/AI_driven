from env.game_env import GameEnv
from stable_baselines3 import PPO
# import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
import networkx as nx

# 读取模型
model = PPO.load("./model/ppo_10M_N50_dis1_b3_tr2_env_3.zip")

results = []
fix_counts_c = 0
runs = 200 # 仿真次数
start_time = time.time()

# b_to_c = np.linspace(2, 20, 10)
K_dis = [1, 2, 3]

start_time = time.time()

for b_item in K_dis:
    # print(f"b/c={b_item}")
    print(f"K_dis = {b_item}")
    fix_counts_c = 0
    for run in range(runs):
        # if run % 100 == 0:
        #     print(f"第{run + 1}次仿真")
        
        env = GameEnv(dis=b_item)
        state,_ = env.reset()

        while True:
        
            # 使用模型进行预测
            action, _ = model.predict(state, deterministic=True)
            state, reward, done, _, _ = env.step(action)

            if all(strategy == 1 for strategy in env.strategies):
                fix_counts_c += 1
                G_T = nx.from_numpy_array(env.G_adj_matrix)
                k_mean = sum([len(list(G_T.neighbors(node))) for node in G_T.nodes]) / env.N
                results.append(k_mean)
                results.append(b_item)
                break
            elif all(strategy == 0 for strategy in env.strategies):
                break

    # fix_rate_c = fix_counts_c / float(runs)
    # results.append(fix_rate_c)
    print(results)

results = np.array(results)
# results = results.reshape(len(b_to_c), 1)
results = results.reshape(-1, 2)

print(f"耗时：{time.time() - start_time}")

# inde = ['b/c=2', 'b/c=3', 'b/c=3.5', 'b/c=4', 'b/c=4.5', 'b/c=5']
colu = ['k_mean', 'K_dis']
results_df = pd.DataFrame(results, columns=colu)

results_df.to_excel('./results/k_mean_rr.xlsx')


    
    