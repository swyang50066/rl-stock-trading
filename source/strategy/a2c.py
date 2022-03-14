import  numpy       as  np

import  keras.backend       as  K
from    keras.models        import  Model
from    keras.layers        import  Input, Dense, Flatten
from    keras.optimizers    import  RMSprop
from    keras.regularizers  import  l2 
from    keras.utils         import  to_categorical

from    strategy.agent      import  Agent, Strategy


class A2CAgent(Agent):
    '''  Actor for the A2C algorithm
    '''
    def __init__(self, input_dim, output_dim, lr):

        # Build network model
        inputs = self.build(input_dim=input_dim, output_dim=output_dim)

        # Set up model functions
        self.setup(*(inputs + (lr,)))

        # Compile network model with loss function and optimizer
        self.compile()

    def _policy_loss_function(value_true, value_pred, eps=1.e-16):
        ''' Policy loss function
        '''
        advantage = value_true - value_pred
        def _loss(y_true, y_pred):
            # Eligibility
            eligibility = K.sum(y_pred*y_true, axis=1, keepdims=True) 
            eligibility = K.sum(-K.log(eligibility) + eps)
            eligibility *= K.stop_gradient(advantage))
            
            # Entropy
            entropy = K.sum(y_pred*K.log(y_pred + eps), axis=1)
            
            return alpha*entropy - eligibility
        
        return _loss

    def _value_loss_function(y_true, y_pred):
        adventage = y_true - y_pred
        return K.mean(.5*K.square(adventage))

    def setup(self, input_value, output_value, lr):
        ''' Set optimizer, loss and callback functions
        ''' 
        # Set optimizer
        self.optimizer = RMSprop(lr=lr, epsilon=0.1, rho=0.99)

        # Define loss function
        self.loss = [
            self._pol_loss_func(input_value, output_value)
            self._val_loss_func, 
        ]

        # List callbacks
        self.callback = []

    def build(self, input_dim, output_dim):
        ''' Build network model
        '''
        # Inputs
        input_state = Input(shape=(input_dim,))
        input_value = Input(shape=(1,))

        # Shared feed-forward network
        x = Flatten()(input_state)
        x = Dense(64, activation="relu")(x)
        x = Dense(128, activation="relu")(x)

        # Actor pathway
        output_action = Dense(128, activation="relu")(x)
        output_action = Dense(output_dim, activation="softmax")(output_action)

        # Critic pathway 
        output_value = Dense(128, activation="relu")(bn)
        output_value = Dense(1, activation="linear")(output_value)

        # Build model        
        self.model = Model(
            inputs=[inputs_state, input_value],
            outputs=[output_action, output_value]
        )

        return input_value, output_value

    def compile(self, ):
        ''' Compile model with optimizer and loss function
        '''
        self.model.compile(loss=self.loss, optimizer=self.optimizer)

    def evolve(self, transitions):
        ''' Evolve model for a iteration
        '''
        # Parse input variables
        states, actions, discounted_reward, _  = transitions

        # Evaluate input and target values
        inputs = [states, discounted_reward]
        targets = [actions, self.model.predict(states)]

        # Train on batch
        self.model.train_on_batch(x=inputs, y=outputs)


class A2C(Strategy):
    ''' Actor-Critic algorithm
    '''
    def __init__(self, env,
                       input_dim=181,
                       output_dim=3,  
                       gamma=.99,
                       init_learning_rate=1.e-4,
                       num_frame=4,
                       num_episode=5000,
                ) -> None: 
        # Initialize internal setup
        super(Strategy, self).__init__(self, 
            input_dim=input_dim, 
            output_dim=output_dim, 
            gamma=gamma,
            init_learning_rate=init_learning_rate,
            num_frame=num_frame,
            num_episode=num_episode
        )

        # Environment object
        self.env = env

        # Create actor and critic networks
        self.agent = A2CAgent(
            input_dim=self.input_dim, 
            output_dim=self.output_dim, 
            lr=self.init_learning_rate,
        )
        
    def get_next_action(self, state):
        ''' Use actor policy to get next action
        '''
        return np.random.choice(
            np.arange(self.output_dim), 
            size=1, 
            p=self.actor.predict(state).ravel()
        )[0]

    def get_discounted_reward(self, rewards):
        ''' Evaluate discounted reward
        '''
        discounted_reward = 0
        for reward in reversed(rewards):
            discounted_reward = reward + self.gamma*discounted_reward

        return discounted_reward

    def train(self):
        ''' Train agent under A2C algorithm
        '''
        for episode in range(self.num_episode):
            # Reset episode
            curr_state = self.env.reset()
            actions, states, rewards, b_terminal = [], [], [], False

            # Simulate episode
            while not b_terminal:
                # Actor an action by following the current policy
                action = self.get_next_action(curr_state)
                
                # Retrieve next state and reward
                next_state, reward, b_terminal, _ = self.env.step(action)
                
                # Memorize (s, a, r) for training
                actions.append(to_categorical(action, self.output_dim))
                rewards.append(reward)
                states.append(curr_state)
                
                # Update current state
                curr_state = next_state

            # Evolve agents
            discounted_reward = self.get_discounted_reward(rewards)
            transitions = (states, actions, discounted_reward, b_terminal)
            self.evolve(transitions)    

    def trade(self):
        ''' Start stock trade
        '''
        # Initialize episode
        curr_state = self.env.reset()
        portfolio = list()    ## This is going to be substitute a class method

        # Simulate episode
        while not b_terminal:
            # Get action
            action = self.get_next_action(curr_state)
            
            # Observe state
            next_state, reward, b_terminal, _ = self.env.step(action)

            # Memory
            portfolio.append(action)

            # Update current state
            curr_state = next_state
