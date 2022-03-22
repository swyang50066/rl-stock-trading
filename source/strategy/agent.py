from    abc     import  ABC, abstractmethod

import  numpy   as  np


class Agent(ABC):
    ''' Superclass including abstract methods required for RL agent
    '''
    @abstractmethod
    def build(self):
        ''' Build network model with loss function and optimizer
        '''
        pass

    @abstractmethod
    def setup(self):
        ''' Set optimizer, loss and callback functions
        '''
        pass
            
    @abstractmethod
    def compile(self):
        ''' Compile model with optimizer and loss function
        '''
        pass

    @abstractmethod
    def evolve(self, transitions):
        ''' Evolve model for a iteration
        '''
        pass

    @abstractmethod
    def predict(self, state):
        ''' Do model prediction
        '''
        return None


class Axuiliary(object):
    ''' Axuiliary subclass including utility methods
    '''
    def reshape(self, x):
        ''' Reshape input to be three dimensional
        '''
        if len(x.shape) < 3:    # add batch dim
            return np.expand_dims(x, axis=0)
        else: 
            return x

    def transfer(self, model_a, model_b):
        ''' Transfer model_a weights to model_b
        '''
        model_b.set_weights(model_a.get_weights())

    def save_weight(self, model, file_path):
        ''' Save model weights
        '''
        model.save_weights(file_path)

    def load_weight(self, model, file_path):
        ''' Load model weights
        '''
        model.load_weights(file_path)


class Strategy(ABC, Axuiliary):
    ''' Strategy superclass for q-value based algorithms
    '''
    def __init__(self, env,
                       gamma=.99,
                       init_learning_rate=1.e-4,
                       num_episode=5000,
                ):
        # Agent class
        self._agent = None

        # Environment class
        self.env = env

        # Model parameters
        self.input_dim = env.observation_dim
        self.output_dim = env.stock_dim
        self.num_frame = len(env.df.index.unique())
        self.gamma = gamma
        self.init_learning_rate = init_learning_rate
        self.num_episode = num_episode

    @property
    def agent(self):
        return self._agent

    @agent.setter
    def agent(self, Iagent):
        self._agent = Iagent

    @abstractmethod
    def get_next_action(self, state):
        ''' Return next action of input  state
        '''
        pass

    @abstractmethod
    def get_discounted_reward(self, rewards):
        ''' Evaluate discounted reward
        '''
        pass

    @abstractmethod
    def train(self):
        ''' Train agent under a RL strategy
        '''
        pass

    @abstractmethod
    def trade(self):
        ''' Trade stock with a RL strategy
        '''
        pass

    def initialize(self):
        ''' Initialize agent configuration
        '''
        self._agent.build()
        self._agent.setup()
        self._agent.compile()

    def evolve(self, transitions):
        ''' Train model for a iteration
        '''
        self._agent.evolve(transitions)
        
    def predict(self, inputs, output_type):
        ''' Do model prediction
        '''
        return self._agent.predict(inputs, output_type)
