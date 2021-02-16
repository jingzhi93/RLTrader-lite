import matplotlib.pyplot as plt
import numpy as np
import gym
from gym import spaces
import random
from gym import spaces
from sklearn.preprocessing import MinMaxScaler
from render_env_ori import BTCTradingGraph

MAX_TRADING_SESSION = 1400

class CryptoEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, commission=0.00075, lookback_window_size=10, initial_balance=100000, serial=False):
        print('init')
        super(CryptoEnv).__init__()
        self.viewer = None
        self.df = df.reset_index()
        self.commission = commission
        self.lookback_window_size = lookback_window_size
        self.initial_balance= initial_balance
        self.serial = serial
        self.scaler = MinMaxScaler()
        self.prev_networth = None

        #action space, buy, sell, hold
        self.action_space = spaces.MultiDiscrete([3, 10])
        self.observation_space = spaces.Box(low=0, high=1, shape=(10, lookback_window_size+1), dtype=np.float16)


    def _reset_session(self):
        """
        Pick a trading session from the dataframe
        :return:
        """
        self.current_step = 0

        if self.serial:
            self.step_left = len(self.df) - self.lookback_window_size - 1
            self.frame_start = self.lookback_window_size
        else:
            self.step_left = np.random.randint(1, MAX_TRADING_SESSION)
            self.frame_start = np.random.randint(self.lookback_window_size, len(self.df) - self.step_left)

        self.active_df = self.df[self.frame_start - self.lookback_window_size: self.lookback_window_size + self.step_left]

    def _next_observation(self):
        end = self.current_step + self.lookback_window_size + 1
        price = np.array([
            self.active_df['Open'].values[self.current_step:end],
            self.active_df['High'].values[self.current_step:end],
            self.active_df['Low'].values[self.current_step:end],
            self.active_df['Close'].values[self.current_step:end],
            self.active_df['Volume'].values[self.current_step:end],
        ])

        scaled_history = self.scaler.fit_transform(self.account_history)
        # print(self.current_step, end)
        # print(price)
        # print(scaled_history)
        obs = np.append(price, scaled_history[:, -(self.lookback_window_size + 1):], axis=0)
        print('Observation \n', obs)
        return obs

    def reset(self):
        print('reset')
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.btc_held = 0

        self._reset_session()

        self.account_history = np.repeat([
            [self.net_worth],
            [0],
            [0],
            [0],
            [0]
        ], self.lookback_window_size+1, axis=1)

        self.trades = []
        return self._next_observation()


    def _take_action(self, action, current_price):
        print('take_action')
        buy_sell_action = action[0]
        amount = action[1] / 10

        btc_bought = 0
        btc_sold = 0
        cost = 0
        sales = 0

        #buy
        if buy_sell_action == 0:
            btc_bought = (self.balance / current_price) * amount
            cost = btc_bought * current_price * (1+ self.commission)
            self.btc_held += btc_bought
            self.balance -= cost

        elif buy_sell_action == 1:
            btc_sold = self.btc_held * amount
            sales = btc_sold * current_price * (1 - self.commission)
            self.btc_held -= btc_sold
            self.balance += sales

        if btc_sold > 0 or btc_bought >0:
            self.trades.append({
                'step': self.frame_start + self.current_step,
                'amount': btc_sold if btc_sold>0 else btc_bought,
                'total': sales if btc_sold > 0 else cost,
                'type': 'sell' if btc_sold > 0 else 'buy'
            })

        self.net_worth = self.balance + self.btc_held * current_price
        self.account_history = np.append(self.account_history,
                                         [[self.net_worth],
                                         [btc_bought],
                                         [cost],
                                         [btc_sold],
                                         [sales]], axis=1)


    def _get_current_price(self):
        print('Current step', self.current_step, 'Current Price', float(self.active_df['Close'].values[self.current_step]))
        return float(self.active_df['Close'].values[self.current_step])

    def step(self, action, end=True):
        print('step')
        current_price = self._get_current_price() + 0.01
        self._take_action(action, current_price)
        print('step_left', self.step_left)
        self.step_left -= 1
        self.current_step += 1

        #when step_left ==0, sell all btc cins on our hand
        if self.step_left == 0:
            self.balance += self.btc_held * current_price
            self.btc_held = 0
            #pick other session
            self._reset_session()

        obs = self._next_observation()
        # print(obs)

        reward = self.net_worth - self.initial_balance
        if reward >0:
            reward =1
        else:
            reward = 0
        done = self.net_worth < 0
        return obs, reward, done, {}

    def render(self, mode='human', close=False):

        if mode == 'human':
            if self.viewer == None:
                self.viewer = BTCTradingGraph(self.df, 'Title')

            self.viewer.render(self.frame_start + self.current_step, self.net_worth, self.trades,
                               window_size=self.lookback_window_size)
        elif mode == 'system':
            print('Price: ' + str(self._current_price()))
            print('Bought: ' + str(self.account_history['asset_bought'][self.current_step]))
            print('Sold: ' + str(self.account_history['asset_sold'][self.current_step]))
            print('Net worth: ' + str(self.net_worths[-1]))

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None