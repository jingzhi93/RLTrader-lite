from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from myenv import CryptoEnv
import pandas as pd
import os
import matplotlib.pyplot as plt
# import talib
import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv('data/btc_price_yahoo.csv')
# df['CCI'] = talib.CCI(df['High'], df['Low'], df['Close'])
# df['RSI'] = talib.RSI(df['Close'])
df.dropna(axis=0, inplace=True)

df_train = df.iloc[:-300,:]

env = DummyVecEnv([lambda: CryptoEnv(df_train)])

# Instanciate the agent
model = PPO2(MlpPolicy, env, verbose=1)

# Train the agent
model.learn(total_timesteps=int(1e4))

# Render the graph of rewards
env.render()

# Save the agent
# model.save('PPO2_CRYPTO')

# Trained agent performence
# env = DummyVecEnv([lambda: CryptoEnv(df_test, serial=True)])
# obs = env.reset()
# env.render()
# daily_profits = []
# for i in range(100000):
#     action, _states = model.predict(obs)
#     obs, rewards, done, info = env.step(action)
#     profits = env.render()
#     daily_profits.append(profits)
#
# fig, ax = plt.subplots()
# ax.plot(daily_profits, '-o', label='BTC', marker='o', ms=10, alpha=0.7, mfc='orange')
# ax.grid()
# plt.xlabel('step')
# plt.ylabel('profit')
# # ax.legend(prop=font)
# plt.show()
