import numpy as np

# import  jax.numpy       as  jnp
# from    jax             import  grad, jit, vmap

import pandas as pd

import gym
from gym import spaces, error
from gym.utils import closer, seeding


class Environment(gym.Env):
    """Stock Trading Environment of GYM API

    The continous action space is normalized between (-1, 1)
    and scaled as 'hmax_norm'

    The observation space is to be in (0, inf)

        len(observation_space) = 181 = (
            [Current Balance]
            + [prices: 1 to stock_dim=30]
            + [owned shares: 1 to stock_dim=30]
            + [macd: 1 to stock_dim=30]
            + [rsi: 1 to stock_dim=30]
            + [cci: 1 to stock_dim=30]
            + [adx: 1 to stock_dim=30]
        )
    """

    def __init__(
        self,
        df,
        current_day=0,
        stock_dim=30,  # total number of stocks in our portfolio
        hmax_norm=100,  # shares normalization factor, e.g.) 100 shares per trade
        init_account_balance=1000000,  # initial amount of money we have in our account
        transaction_fee_percent=0.0001,  # trasaction fee, e.g.) 0.1% resonable percentage
        reward_scale=1.0e-4,  # reward scaling factor
        random_seed=931016,
    ):
        # Set data
        self.df = df
        self.current_day = current_day

        # Set parameter
        self.stock_dim = stock_dim
        self.observation_dim = 6 * stock_dim + 1
        self.owned_share_indice = slice(stock_dim + 1, 2 * stock_dim + 1)
        self.hmax_norm = hmax_norm
        self.init_account_balance = init_account_balance
        self.transaction_fee_percent = transaction_fee_percent
        self.reward_scale = reward_scale

        # Set subclasses
        self.action_space = spaces.Box(low=-1, high=1, shape=(stock_dim,))
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(self.observation_dim,)
        )
        self.reward_range = (-np.inf, np.inf)

        # Declare metadata container
        self.metadata = {"render/mode": []}

        # Initialize state
        _ = self.reset()
        if current_day != 0:  # Starting current_day is Not first current_day
            self.current_day = current_day
            index_list = np.arange(
                self.current_day * self.stock_dim,
                (self.current_day + 1) * self.stock_dim,
            )
            self.data = self.df.loc[index_list, :]

        # Get random number generator
        self.rng, _ = seeding.np_random(random_seed)

    def reset(self):
        # Reset memorize all the total balance change
        self.asset_history = [self.init_account_balance]
        self.reward_history = []

        # Reset variables
        self.reward = 0
        self.cost = 0
        self.num_trade = 0

        # Reset data
        self.current_day = 0
        index_list = np.arange(
            self.current_day * self.stock_dim,
            (self.current_day + 1) * self.stock_dim,
        )
        self.data = self.df.loc[index_list, :]
        self.b_terminal = False

        # Reset state
        self.state = (
            [self.init_account_balance]
            + self.data.adjcp.values.tolist()
            + [0] * self.stock_dim
            + self.data.macd.values.tolist()
            + self.data.rsi.values.tolist()
            + self.data.cci.values.tolist()
            + self.data.adx.values.tolist()
        )

        return self.state

    def _sell_stock(self, index, action):
        # update balance
        self.state[0] += (
            self.state[index + 1]
            * min(abs(action), self.state[index + self.stock_dim + 1])
            * (1 - self.transaction_fee_percent)
        )

        self.state[index + self.stock_dim + 1] -= min(
            abs(action), self.state[index + self.stock_dim + 1]
        )

        self.cost += (
            self.state[index + 1]
            * min(abs(action), self.state[index + self.stock_dim + 1])
            * self.transaction_fee_percent
        )

        # Count trade
        self.num_trade += 1

    def _buy_stock(self, index, action):
        # perform buy action based on the sign of the action
        available_amount = self.state[0] // self.state[index + 1]

        # update balance
        self.state[0] -= (
            self.state[index + 1]
            * min(available_amount, action)
            * (1 + self.transaction_fee_percent)
        )

        self.state[index + self.stock_dim + 1] += min(
            available_amount, action
        )

        self.cost += (
            self.state[index + 1]
            * min(available_amount, action)
            * self.transaction_fee_percent
        )

        # Count trade
        self.num_trade += 1

    def step(self, actions, b_panic=False):
        # Check simulation termination
        self.b_terminal = (
            self.stock_dim * self.current_day
            >= len(self.df.index.unique()) - 1
        )
        if self.b_terminal:  ## Add terminal returns (call render)
            """ 
            df_total_value = pd.DataFrame(self.asset_memory)
            df_total_value['daily_return']= df_total_value.pct_change(1)
            sharpe = (4**0.5)*df_total_value['daily_return'].mean()/ \
                  df_total_value['daily_return'].std()
            """
            return self.state, self.reward, self.b_terminal, self.metadata

        # Get trade strategy
        if b_panic:  # Over the threshold of turbulence index, do panic sell
            actions = np.array([-self.hmax_norm] * self.stock_dim)
        else:
            actions = actions * self.hmax_norm

        # Get total asset before trading
        begin_total_asset = self.state[0] + sum(
            np.array(self.state[1 : self.stock_dim + 1])
            * np.array(self.state[self.owned_share_indice])
        )

        # Do trading
        for index, action in enumerate(actions):
            if action < 0 and self.state[index + self.stock_dim + 1] > 0:
                self._sell_stock(index, action)
            elif action > 0:
                self._buy_stock(index, action)

        # Update variables
        index_list = np.arange(
            self.current_day * self.stock_dim,
            (self.current_day + 1) * self.stock_dim,
        )
        self.data = self.df.loc[index_list, :]
        self.turbulence = self.data["turbulence"].values[0]
        self.current_day += 1

        # Load next state
        self.state = (
            [self.state[0]]
            + self.data.adjcp.values.tolist()
            + list(self.state[self.owned_share_indice])
            + self.data.macd.values.tolist()
            + self.data.rsi.values.tolist()
            + self.data.cci.values.tolist()
            + self.data.adx.values.tolist()
        )

        # Get total asset after trading
        end_total_asset = self.state[0] + sum(
            np.array(self.state[1 : self.stock_dim + 1])
            * np.array(self.state[self.owned_share_indice])
        )

        # Update memory buffer
        self.asset_history.append(end_total_asset)

        # Update reward
        self.reward = end_total_asset - begin_total_asset
        self.reward_history.append(self.reward)
        self.reward = self.reward * self.reward_scale

        return self.state, self.reward, self.b_terminal, self.metadata

    def render(self, mode="human"):
        """Renders the environment.

        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:
        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).

        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.

        Args:
            mode (str): the mode to render with

        Example:

        class MyEnv(Env):
            metadata = {'render.modes': ['human', 'rgb_array']}
            def render(self, mode='human'):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode == 'human':
                    ... # pop up a window and render
                else:
                    super(MyEnv, self).render(mode=mode) # just raise an exception
        """
        pass


class Framework(gym.Wrapper):
    """Wraps the environment to allow a modular transformation.

    This class is the base class for all wrappers.
    The subclass could override some methods to change the behavior of
    the original environment without touching the original code.
    """

    def __init__(
        self,
        env,
        previous_state=list(),
        b_initial=True,
        turbulence_threshold=140,  # turbulence index: 90-150 reasonable threshold
    ):
        super(Framework, self).__init__()

        # Define wrapping environment
        self.env = env
        self.previous_state = previous_state
        self.b_initial = b_initial

        # Set parameter
        self.stock_dim = self.env.stock_dim
        self.owned_share_indice = self.env.owned_share_indice
        self.transaction_fee_percent = self.env.transaction_fee_percent
        self.turbulence_threshold = turbulence_threshold

        # setter action_space, observation_space, reward_range, metadata,
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.reward_range = self.env.reward_range

        # Declare metadata container
        self.metadata = self.env.metadata

    def reset(self, **kwargs):
        if self.b_initial:
            # Initialize turbulence factor
            self.turbulence = 0

            return self.env.reset(**kwargs)
        else:
            # Initialize turbulence factor
            self.turbulence = 0

            # Previous total asset
            self.env.asset_history = [
                self.previous_state[0]
                + sum(
                    np.array(self.previous_state[1 : self.stock_dim + 1])
                    * np.array(self.previous_state[self.owned_share_indice])
                )
            ]
            self.env.reward_history = []

            # Reset variable
            self.env.cost = 0
            self.env.reward = 0
            self.env.num_trade = 0

            # Reset data
            self.env.current_day = 0
            index_list = np.arange(
                self.env.current_day * self.stock_dim,
                (self.env.current_day + 1) * self.stock_dim,
            )
            self.env.data = self.df.loc[index_list, :]
            self.env.b_terminal = False

            # Reset state
            self.env.state = (
                [self.previous_state[0]]
                + self.env.data.adjcp.values.tolist()
                + self.previous_state[self.owned_share_indice]
                + self.env.data.macd.values.tolist()
                + self.env.data.rsi.values.tolist()
                + self.env.data.cci.values.tolist()
                + self.env.data.adx.values.tolist()
            )

            return self.state

    def _buy_stock(self, index, action):
        if self.turbulence < self.turbulence_threshold:
            self.env._buy_stock(index, action)
        else:
            pass

    def _sell_stock(self, index, action):
        if self.turbulence < self.turbulence_threshold:
            self.env._sell_stock(index, action)
        else:  # if turbulence goes over threshold, just clear out all positions
            # Update balance
            self.env.state[0] += (
                self.env.state[index + 1]
                * self.env.state[index + self.stock_dim + 1]
                * (1 - self.transaction_fee_percent)
            )

            self.env.state[index + self.stock_dim + 1] = 0

            self.cost += (
                self.env.state[index + 1]
                * self.env.state[index + self.stock_dim + 1]
                * self.transaction_fee_percent
            )

            # Count trading
            self.num_trade += 1

    def step(self, action):
        if self.turbulence < self.turbulence_threshold:
            return self.env.step(action)
        else:
            return self.env.step(action, b_panic=True)

    def render(self, mode="human", **kwargs):
        return self.env.render(mode, **kwargs)
