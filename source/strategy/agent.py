from    abc     import  ABC, abstractmethod

import  numpy   as  np


class Agent(ABC):
    ''' Superclass including abstract methods required for RL agent
    '''
    @abstractmethod
    def build(self):
        ''' Build network model
        '''
        pass
    
    @abstrctmethod
    def set_loss_function(self):
        ''' Set loss function
        '''
        pass

    @abstractmethod
    def set_optimizer(self):
        ''' Set optimizer
        '''
        pass

    @abstractmethod
    def set_callback(self):
        ''' Set callback
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


class Axuiliary(object)
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

    def save_weight(self, model, path, filename):
        ''' Save model weights
        '''
        model.save_weights(path + filename)

    def load_weight(self, model, path, filename):
        ''' Load model weights
        '''
        model.load_weights(path + filename)

        pass


class Strategy(Axuiliary):
    ''' Strategy superclass for q-value based algorithms
    '''
    def __init__(self, ENV_DIM,
                       ACTION_DIM,
                       NUM_FRAME=4,
                       GAMMA=.99,
                       INIT_LEARNING_RATE=1.e-4,
                       NUM_EPISODE=5000,
                ) -> None:
        # Agent class
        self._agent = None

        # Environment and  Model parameters
        self.ACTION_DIM = ACTION_DIM
        self.ENV_DIM = (NUM_FRAME,) + ENV_DIM
        self.GAMMA = GAMMA
        self.INIT_LEARNING_RATE = INIT_LEARNING_RATE
        self.NUM_EPISODE = NUM_EPISODE

    @property
    def agent(self):
        return self._agent

    @actor.setter
    def agent(self, Iagent):
        self._agent = Iagent

    def evolve(self, transitions):
        ''' Train model for a iteration
        '''
        self._agent.evolve(transitions)



class ActorCriticStrategy(Axuiliary):
    ''' Strategy superclass for actor-critic based algorithms 
    '''
    def __init__(self, ENV_DIM,
                       ACTION_DIM,
                       NUM_FRAME=4,
                       GAMMA=.99,
                       INIT_LEARNING_RATE=1.e-4,
                       NUM_EPISODE=5000,
                ) -> None:
        # Agent class
        self._agent = None

        # Environment and  Model parameters
        self.ACTION_DIM = ACTION_DIM
        self.ENV_DIM = (NUM_FRAME,) + ENV_DIM
        self.GAMMA = GAMMA
        self.INIT_LEARNING_RATE = INIT_LEARNING_RATE
        self.NUM_EPISODE = NUM_EPISODE

    @property
    def actor(self):
        return self._actor

    @actor.setter
    def actor(self, Iactor):
        self._actor = Iactor

    @property
    def critic(self):
        return self._critic

    @critic.setter
    def critic(self, Icritic):
        self._critic = Icritic

    def evolve(self, transitions):
        ''' Train model for a iteration
        '''
        self._actor.evolve(transitions)
        self._critic.evolve(transitions)

