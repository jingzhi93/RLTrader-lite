from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from myenv import CryptoEnv
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


def train(path='data/train/btc_price_yahoo_140917_191231.csv'):
    df_train = pd.read_csv(path)
    df_train.dropna(axis=0, inplace=True)
    env = DummyVecEnv([lambda: CryptoEnv(df_train, serial=False)])
    # Instanciate the agent
    model = PPO2(MlpPolicy, env, seed=42, verbose=1)
    # Train the agent
    model.learn(total_timesteps=int(2e4))
    model.save('PPO2_CRYPTO')
    return model

def validate_and_render(path, model, render_mode):
    file_name = path.split('/')[-1]
    df = pd.read_csv(path)
    df.dropna(axis=0, inplace=True)
    # Trained agent performence
    env = DummyVecEnv([lambda: CryptoEnv(df, serial=True)])
    obs = env.reset()
    daily_profits = []
    for i in range(len(df)-12): #lookback_size
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        profits = env.render(render_mode)
        daily_profits.append(profits)

    fig, ax = plt.subplots()
    ax.plot(daily_profits, '-o', label='BTC', marker='o', ms=10, alpha=0.7, mfc='orange')
    ax.grid()
    plt.xlabel('step')
    plt.ylabel('profit')
    plt.title(file_name)
    plt.show()

if __name__ == '__main__':
    train_mode = True
    model = train()
    #true test (unseen)
    validate_and_render('data/test/btc_price_yahoo_200101_210103.csv', model, render_mode='system')
    #test historical year
    validate_and_render('data/test/btc_price_yahoo_170101_171231.csv', model, render_mode='human')
    validate_and_render('data/test/btc_price_yahoo_180101_181231.csv', model, render_mode='human')
    validate_and_render('data/test/btc_price_yahoo_190101_191231.csv', model, render_mode='human')
