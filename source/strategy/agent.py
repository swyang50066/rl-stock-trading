from    abc     import  ABC, abstractmethod

import  numpy   as  np


class Agent(ABC):
    ''' Superclass including abstract methods required for RL agent
    '''
    def __init__(self, INPUT_DIM=1, 
                       OUTPUT_DIM=1, 
                       INIT_LERANING_RATE=1.e-5
                ) -> None:
        # I/O dimensionalities
        self.INPUT_DIM = INPUT_DIM
        self.OUTPUT_DIM = OUTPUT_DIM

        # Init learning rate
        self.INIT_LERANING_RATE = INIT_LEARNING_RATE

    @abstractmethod
    def compile(self):
        ''' Compile model with optimizer and loss function
        '''
        pass

    @abstractmethod
    def transfer(self):
        ''' Transfer model weights to competitive one
            
            e.g.) from model_a to model_b
                model_b.set_weights(model_a.get_weights())
        '''
        pass

    @abstractmethod
    def evolve(self, x, y):
        ''' Train model for a iteration 
        
            e.g.) Return below
                self.model.fit(self.reshape(x), y, epochs=1, verbose=0)
        '''
        pass

    def trade(self, x):
        ''' Prediction for critic value
        
            e.g.) Return below
                self.model.predict(self.reshape(x))
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

    def save_weight(self, model, path, filename):
        ''' Save model weights
        '''
        model.save_weights(path + filename)

    def load_weight(self, model, path, filename):
        ''' Load model weights
        '''
        model.load_weights(path + filename)


class Strategy(Axuiliary):
    ''' Strategy superclass for training and testing policy of RL agent
    '''
    def __init__(self):
        self._agent = None

    @property
    def agent(self):
        return self._agent

    @agent.setter
    def strategy(self, Iagent):
        self._strategy = Iagent()

    def evolve(self, x, y):
        return self._agent.evolve(x, y)

    def trade(self, x):
        return self._agent.trade(x)
