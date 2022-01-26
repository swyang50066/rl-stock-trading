import  pandas          as  pd

import  numpy           as  np
#import  jax.numpy       as  jnp
#from    jax             import  grad, jit, vmap

import  gym
from    gym             import  spaces, error
from    gym.utils       import  closer, seeding
from    gym.logger      import  deprecation


class Environment(gym.Env):
    ''' The main OpenAI Gym class. 
        
        It encapsulates an environment with arbitrary behind-the-scenes dynamics. 
        An environment can be partially or fully observed.
    
        The main API methods that users of this class need to know are:
            
            step
            reset
            render
            close
            seed

        And set the following attributes:
        
            action_space: 
                The Space object corresponding to valid actions
            observation_space: 
                The Space object corresponding to valid observations
            reward_range: 
                A tuple corresponding to the min and max possible rewards
    
        Note that a default reward range set to [-inf,+inf] already exists. 
        Set it if you want a narrower range.
        The methods are accessed publicly as "step", "reset", etc...
    '''
    def __init__(self, df, day=0,
                       STOCK_DIM=30,    # total number of stocks in our portfolio
                       HMAX_NORMALIZE=100,    # shares normalization factor, e.g.) 100 shares per trade
                       INITIAL_ACCOUNT_BALANCE=1000000,    # initial amount of money we have in our account
                       TRANSACTION_FEE_PERCENT=.0001,    # trasaction fee, e.g.) 0.1% resonable percentage
                       REWARD_SCALING=1.e-4,    # reward scaling factor
                ):
        # Set data
        self.df = df
        self.day = day

        # Set parameter
        self.STOCK_DIM = STOCK_DIM
        self.OWNED_SHARE_INDICE = slice(STOCK_DIM+1, 2*STOCK_DIM+1)
        self.HMAX_NORMALIZE = HMAX_NORMALIZE
        self.INITITAL_ACCOUNT_BALANCE = INITIAL_ACCOUNT_BALANCE
        self.TRANSACTION_FEE_PERCENT = TRANSACTION_FEE_PERCENT
        self.REWARD_SCALING = REWARD_SCALING

        # Set subclasses
        '''
        The continous action space is normalized between -1 and 1 and scaled as
        'STOCK_DIM'

        The observation space is to be in (0, inf)
        len(observation_space) = (
            [Current Balance]
            + [prices: 1-30]
            + [owned shares: 1-30] 
            + [macd: 1-30]
            + [rsi: 1-30] 
            + [cci: 1-30] 
            + [adx: 1-30]
        )
        '''
        self.action_space = spaces.Box(low=-1, high=1, shape=STOCK_DIM)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(181,))
        self.reward_range = (-np.inf, np.inf)
        
        # Declare metadata container
        self.metadata = {
            "render/mode": []
        }
       
        # Initialize state 
        _ = self.reset(bFirstday)
        if day != 0:    # Starting day is Not first day 
            self.day = day
            self.data = self.df.loc[self.day, :]
        
        # Get random number generator
        self.rng, _ = seeding.np_random(seed)

    def reset(self):
        # Reset memorize all the total balance change
        self.assetHistory = [self.INITIAL_ACCOUNT_BALANCE]
        self.rewardHistory = []
        
        # Reset variables
        self.reward = 0
        self.cost = 0
        self.numTrade = 0
        
        # Reset data
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.bTerminal = False
        
        # Reset state
        self.state = (
            [INITIAL_ACCOUNT_BALANCE] 
            + self.data.adjcp.values.tolist()
            + [0]*STOCK_DIM 
            + self.data.macd.values.tolist()
            + self.data.rsi.values.tolist()
            + self.data.cci.values.tolist() 
            + self.data.adx.values.tolist()
        )

        return self.state

    def _sellStock(self, index, action):
        # update balance
        self.state[0] += (
            self.state[index+1]
            * min(abs(action), self.state[index+self.STOCK_DIM+1]) 
            * (1 - self.TRANSACTION_FEE_PERCENT)
        )
                
        self.state[index+self.STOCK_DIM+1] -= (
            min(abs(action), self.state[index+self.STOCK_DIM+1])
        )

        self.cost += (
            self.state[index+1]
            * min(abs(action), self.state[index+self.STOCK_DIM+1]) 
            * self.TRANSACTION_FEE_PERCENT
        )

        # Count trade
        self.numTrade += 1
    
    def _buyStock(self, index, action):
        # perform buy action based on the sign of the action
        available_amount = self.state[0] // self.state[index+1]
            
        #update balance
        self.state[0] -= (
            self.state[index+1]
            * min(available_amount, action)
            * (1 + self.TRANSACTION_FEE_PERCENT)
        )

        self.state[index+self.STOCK_DIM+1] += (
            min(available_amount, action)
        )

        self.cost += (
            self.state[index+1]
            * min(available_amount, action)
            * self.TRANSACTION_FEE_PERCENT
        )

        # Count trade
        self.numTrade+=1

    def step(self, actions, bPanic=False):
        #
        self.bTerminal = self.day >= len(self.df.index.unique())-1

        if self.bTerminal:
            #### Add terminal returns (call render)
            ''' 
            df_total_value = pd.DataFrame(self.asset_memory)
            df_total_value['daily_return']= df_total_value.pct_change(1)
            sharpe = (4**0.5)*df_total_value['daily_return'].mean()/ \
                  df_total_value['daily_return'].std()
            '''
            return self.state, self.reward, self.bTerminal, {}
    
        if bPanic:    # Over the threshold of turbulence index, do panic sell
            actions = np.array([-self.HMAX_NORMALIZE] * self.STOCK_DIM)
        else:
            actions = actions * self.HMAX_NORMALIZE

        begin_total_asset = (
            self.state[0]
            + sum(
                np.array(self.state[1:self.STOCK_DIM+1])
                * np.array(self.state[self.OWNED_SHARE_INDICE)]
            )
        )

        # Do trading
        for action in actions:
            if action < 0 and self.state[index+self.STOCK_DIM+1] > 0:
                self._sellStock(index, action)
            elif action > 0:
                self._buyStock(index, action)

        #
        self.day += 1
        self.data = self.df.loc[self.day,:]         
        self.turbulence = self.data['turbulence'].values[0]
            
        # load next state
        self.state = (
            [self.state[0]] 
            + self.data.adjcp.values.tolist()
            + list(self.state[self.OWNED_SHARE_INDICE])
            + self.data.macd.values.tolist()
            + self.data.rsi.values.tolist()
            + self.data.cci.values.tolist()
            + self.data.adx.values.tolist()
        )

        end_total_asset = (
            self.state[0]
            + sum(
                np.array(self.state[1:self.STOCK_DIM+1])
                * np.array(self.state[self.OWNED_SHARE_INDICE])
            )
        )
        self.assetHistory.append(end_total_asset)
            
        self.reward = end_total_asset - begin_total_asset            
        self.rewardHistory.append(self.reward)
        self.reward = self.reward * self.REWARD_SCALING

        return self.state, self.reward, self.bTerminal, {}
   
    
    def render(self, mode="human"):
        ''' Renders the environment.
        
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
        '''
        pass
        

class Framework(gym.Wrapper):
    ''' Wraps the environment to allow a modular transformation.
    
        This class is the base class for all wrappers. 
        The subclass could override some methods to change the behavior of 
        the original environment without touching the original code.
    '''
    def __init__(self, env, 
                       previous_state=list(), 
                       bInitial=True,
                       TURBULENCE_THRESHOLD=140,    # turbulence index: 90-150 reasonable threshold
                ):
        super(Framework, self).__init__()
        
        # Define wrapping environment
        self.env = env
        self.previous_state = previous_state
        self.bInitial = bInitial

        # Set parameter
        self.STOCK_DIM = self.env.STOCK_DIM
        self.OWNED_SHARE_INDICE = self.env.OWNED_SHARE_INDICE
        self.TRANSACTION_FEE_PERCENT = self.env.TRANSACTION_FEE_PERCENT
        self.TURBULENCE_THRESHOLD = TURBULENCE_THRESHOLD

        # setter action_space, observation_space, reward_range, metadata, 
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.reward_range = self.env.reward_range
                
        # Declare metadata container
        self.metadata = self.env.metadata
   
    def reset(self, **kwargs):
        if self.bInitial:
            # Initialize turbulence factor
            self.turbulence = 0
        
            return self.env.reset(**kwargs)
        else:
            # Initialize turbulence factor
            self.turbulence = 0
        
            # Previous total asset
            self.env.assetHistory = [(
                self.previous_state[0]
                + sum(
                    np.array(self.previous_state[1:self.STOCK_DIM+1])
                    * np.array(self.previous_state[self.OWNED_SHARE_INDICE])
                )
            )]
            self.env.rewardHistory = []

            # Reset variable
            self.env.cost = 0
            self.env.reward = 0
            self.env.numTrade = 0
            
            # Reset data
            self.env.day = 0
            self.env.data = self.df.loc[self.env.day, :]
            self.env.bTerminal = False

            # Reset state
            self.env.state = (
                [self.previous_state[0]]
                + self.env.data.adjcp.values.tolist()
                + self.previous_state[self.OWNED_SHARE_INDICE]
                + self.env.data.macd.values.tolist()
                + self.env.data.rsi.values.tolist()
                + self.env.data.cci.values.tolist()
                + self.env.data.adx.values.tolist()
            )

            return self.state
    
    def _buyStock(self, index, action):
        if self.turbulence < self.TURBULENCE_THRESHOLD:
            self.env._buyStock(index, action)
        else:
            pass

    def _sellStock(self, index, action):
        if self.turbulence < self.TURBULENCE_THRESHOLD:
            self.env._sellStock(index, action)
        else:    # if turbulence goes over threshold, just clear out all positions
            # update balance
            self.env.state[0] += (
                self.env.state[index+1]
                * self.env.state[index+self.STOCK_DIM+1]
                * (1 - self.TRANSACTION_FEE_PERCENT)
            )
                
            self.env.state[index+self.STOCK_DIM+1] = 0
            
            self.cost += (
                self.env.state[index+1]
                * self.env.state[index+self.STOCK_DIM+1]
                * self.TRANSACTION_FEE_PERCENT
            )

            # Count trading
            self.numTrade += 1

    def step(self, action):
        if self.turbulence < self.TURBULENCE_THRESHOLD:
            return self.env.step(action)
        else:
            return self.env.step(action, bPanic=True)

    def render(self, mode="human", **kwargs):
        return self.env.render(mode, **kwargs)
