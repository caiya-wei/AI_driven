import time
import gym
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from env.game_env import GameEnv

class CustomCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
    
    def _on_step(self):
        # 假设你的环境有一个方法 get_mean_cooperation_rate() 来获取 mean_cooperation_rate
        mean_cooperation_rate = np.mean([env.get_attr("mean_cooperation_rate", i)[0] for i in range(self.model.get_env().num_envs)])
        self.logger.record('custom/mean_cooperation_rate', mean_cooperation_rate)

        step_reward = np.mean([env.get_attr("step_reward", i)[0] for i in range(self.model.get_env().num_envs)])
        self.logger.record('custom/step_reward', step_reward)
        
        # Save the model every `save_freq` steps
        if self.n_calls % self.save_freq == 0:
            save_path = f"{self.save_path}_{self.num_timesteps}"
            self.model.save(save_path)
            print(f"Model saved at {save_path}")

        return True

# 创建并行环境
env = make_vec_env(GameEnv, n_envs=4)

# 创建PPO模型
model = PPO(
    "MlpPolicy",
    env, 
    verbose=1,
    learning_rate=0.0003,
    batch_size=512,
    n_steps=128,
    n_epochs=10,
    tensorboard_log="./logs/ppo_cartpole_tensorboard/"
)

# 创建自定义回调实例
custom_callback = CustomCallback(save_freq=1_000_000, save_path="./model/ppo")

# 训练模型，添加回调函数
model.learn(total_timesteps=50_000_000, tb_log_name="50M_N100_dis1_facebook", callback=custom_callback)

model.save(f"./model/ppo_{model._total_timesteps}_N100_dis1_facebook")
