import matplotlib.pyplot as plt
import numpy as np
import gym
from gym import spaces
import random
import pandas as pd
from gym import spaces
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from render_env_ori import BTCTradingGraph

MAX_TRADING_SESSION = 90
NUM_FEATURES = 5

class CryptoEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, commission=0.00075, lookback_window_size=10, initial_balance=100000,
                 serial=False):
        super(CryptoEnv).__init__()
        self.viewer = None
        self.commission = commission
        self.lookback_window_size = lookback_window_size
        self.initial_balance = initial_balance
        self.serial = serial
        self.min_max_scaler = MinMaxScaler()
        self.df = self._prepare_features(df)

        # action space, buy, sell, hold
        self.action_space = spaces.MultiDiscrete([3, 10])
        self.observation_space = spaces.Box(low=0, high=1, shape=(10, lookback_window_size + 1), dtype=np.float16)
        self.features = [col for col in self.df.columns if col not in ['index', 'Timestamp']]

    def _prepare_features(self, df):
        df_final = df.copy()
        for ohlcv in ['Open', 'High', 'Low', 'Close', 'Volume']:
            for lag in range(1, self.lookback_window_size+1):
                df_final[f'{ohlcv}_lag_{lag}'] = df_final[ohlcv].shift(lag)
        df_final.dropna(axis=0, inplace=True)
        df_final = df_final.reset_index()
        return df_final

    def _reset_session(self):
        self.current_idx = 0
        if self.serial:
            self.frame_start = 0
            self.step_left = len(self.df) - 1
            self.active_df = self.df.iloc[self.current_idx: ][self.features] #[days, features]
        else:
            self.step_left = np.random.randint(1, MAX_TRADING_SESSION) #define a trading session, maximum 3 months
            self.frame_start = np.random.randint(0, len(self.df) - self.step_left) #randomly create a starting point
            self.active_df = self.df.iloc[self.frame_start: self.frame_start + self.step_left] #[days, features]

    def _next_observation(self):
        prices_df = self.active_df[self.features].iloc[self.current_idx]
        curr_prices_arr = np.array(
            [[prices_df['Open']],
             [prices_df['High']],
             [prices_df['Low']],
             [prices_df['Close']],
             [prices_df['Volume']]]
        )
        history_prices_arr = np.array(
            [[prices_df[f'Open_lag_{i}'] for i in range(1, self.lookback_window_size+1)],
            [prices_df[f'High_lag_{i}'] for i in range(1, self.lookback_window_size+1)],
            [prices_df[f'Low_lag_{i}'] for i in range(1, self.lookback_window_size+1)],
            [prices_df[f'Close_lag_{i}'] for i in range(1, self.lookback_window_size+1)],
            [prices_df[f'Volume_lag_{i}'] for i in range(1, self.lookback_window_size+1)]]
        )
        prices_arr = np.hstack((curr_prices_arr, history_prices_arr))
        prices_arr_norm = prices_arr / prices_arr[:, -1].reshape(len(prices_arr[:, -1]), -1)
        scaled_history = self.min_max_scaler.fit_transform(self.account_history)
        obs = np.append(prices_arr_norm, scaled_history[:, -(self.lookback_window_size + 1):], axis=0)
        return obs


    def reset(self):
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.btc_held = 0
        self._reset_session()
        self.account_history = np.repeat([
            [self.net_worth],
            [0], #step
            [0], #amount
            [0], #total
            [0] #type
        ], self.lookback_window_size+1, axis=1)
        self.trades = []
        return self._next_observation()

    def _get_current_price(self):
        return float(random.uniform(self.active_df.iloc[self.current_idx]["Open"],
                                       self.active_df.iloc[self.current_idx]["Close"]))


    def _take_action(self, action, current_price):
        action_type = action[0]
        amount = action[1] / 10
        btc_bought = 0
        btc_sold = 0
        cost = 0
        sales = 0

        if action_type < 1:
            btc_bought = self.balance / current_price * amount
            cost = btc_bought * current_price * (1 + self.commission)
            self.btc_held += btc_bought
            self.balance -= cost

        elif action_type < 2:
            btc_sold = self.btc_held * amount
            sales = btc_sold * current_price * (1 - self.commission)
            self.btc_held -= btc_sold
            self.balance += sales

        if btc_sold > 0 or btc_bought > 0:
            self.trades.append({
                'step': self.frame_start + self.current_idx,
                'amount': btc_sold if btc_sold > 0 else btc_bought,
                'total': sales if btc_sold > 0 else cost,
                'type': "sell" if btc_sold > 0 else "buy"
            })
        self.net_worth = self.balance + self.btc_held * current_price
        self.account_history = np.append(self.account_history, [
            [self.net_worth],
            [btc_bought],
            [cost],
            [btc_sold],
            [sales]], axis=1)


    def step(self, action):
        # print(action)
        done = False
        current_price = self._get_current_price()
        self._take_action(action, current_price)
        self.step_left -= 1
        self.current_idx += 1
        if self.step_left == 0:
            self.balance += self.btc_held * current_price
            self.btc_held = 0
            # pick other session
            self._reset_session()
        if self.net_worth <= 0:
            done = True
        obs = self._next_observation()
        reward = self.net_worth - self.initial_balance
        if reward > 0:
            reward = 1
        else:
            reward = 0
        return obs, reward, done, {}

    def render(self, mode='human', close=False):
        profit = self.net_worth - self.initial_balance
        if mode == 'human':
            if self.viewer == None:
                self.viewer = BTCTradingGraph(self.df)
            self.viewer.render(self.frame_start + self.current_idx,
                               self.net_worth,
                               self.trades,
                               window_size=self.lookback_window_size)
        elif mode == 'system':
            # Render the environment to the screen
            print('-' * 30)
            print(f'Step: {self.frame_start + self.current_idx}')
            print(f'Balance: {self.balance}')
            print(f'BTC held: {self.btc_held} ')
            print(f'Net worth: {self.net_worth}')
            print(f'Profit: {profit}')
        return profit

