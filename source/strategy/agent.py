from    abc         import  ABC, abstractmethod

import  numpy       as  np
from    keras.optimizers        import  RMSprop


class Agent(ABC):
    ''' Superclass for RL agents
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
    def setLossFunction(self):
        ''' Set loss function
        '''
        pass

    @abstractmethod
    def setOptimizer(self):
        ''' Set optimizer
        '''
        '''
        # Use RMSprop Optimizer rather than Adam-likes
        optimizer =  RMSprop(
            lr=self.INIT_LERANING_RATE, epsilon=.1, rho=.99)
        )
        '''
        pass

    def compile(self):
        ''' Compile model with optimizer and loss function
        '''
        self.model.compile(optimizer=self.optimizer, loss=self.self.loss)

    def evolve(self, x, y):
        ''' Train model for a iteration 
        '''
        self.model.fit(self.reshape(x), y, epochs=1, verbose=0)

    def trade(self, x):
        ''' Prediction for critic value
        '''
        return self.model.predict(self.reshape(x))

    def transfer(self, surrogate):
        ''' Transfer model weight to a surrogate one
        '''
        surrogate.set_weights(self.model.get_weights())

        return surrogate

    def reshape(self, x):
        ''' Reshape input to be three dimensional
        '''
        if len(x.shape) < 3:    # add batch dim
            return np.expand_dims(x, axis=0)
        else: 
            return x

