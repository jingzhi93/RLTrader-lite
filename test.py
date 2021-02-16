import gym
from gym.utils import seeding
import numpy as np
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

class EnvMinimal(gym.Env):

    def __init__(self):
        self.action_space = gym.spaces.MultiDiscrete([3, 10])
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(11, 7, 50), dtype=np.float32)
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self,action):
        print(action)
        return self.get_obs(), 1, False, {}

    def reset(self):
        self.last_u = None
        return self.get_obs()

    def get_obs(self):
        return 1

env = EnvMinimal()
env = DummyVecEnv([lambda: env])

model = PPO2(MlpPolicy, env, verbose=0)
model.learn(total_timesteps=1000)

