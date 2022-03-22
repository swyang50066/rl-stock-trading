import  numpy       as  np

import  tensorflow      as  tf
import  tensorflow.keras.backend        as  K
from    tensorflow.keras.models         import  Model
from    tensorflow.keras.layers         import  Input
from    tensorflow.keras.optimizers     import  RMSprop


from    loss                    import  (policy_loss_func, 
                                         value_loss_func,
                                         )
from    network.a2c_network     import  A2CNetwork
from    strategy.agent          import  Agent, Strategy


class A2CAgent(Agent):
    '''  Actor for the A2C algorithm
    '''
    def __init__(self, input_dim, output_dim, num_frame, lr):
        # Parameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_frame = num_frame
        self.lr = lr

    def build(self):
        ''' Build network model
        '''
        # Inputs
        input_state = Input(shape=(self.input_dim,))
        input_value = Input(shape=(1,))
        inputs = [input_state, input_value]
        
        # Outputs
        mu, log_sigma, value = A2CNetwork(output_dim=self.output_dim)(inputs)
        outputs = [tf.concat([mu, log_sigma], axis=-1), value]

        # Advantage; discounted_reward - critic_predicted_value
        self.advantage = input_value - value

        # Build model        
        self.model = Model(inputs=inputs, outputs=outputs)

    def setup(self):
        ''' Set optimizer, loss and callback functions
        '''
        # Set optimizer, loss functions and callbacks
        self.optimizer = RMSprop(
            learning_rate=self.lr, epsilon=.1, rho=.99
        )
        self.losses = [policy_loss_func(self.advantage), value_loss_func]
        self.callbacks = list()

    def compile(self):
        ''' Compile model with optimizer and loss function
        '''
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.losses 
        )

    def evolve(self, transitions):
        ''' Evolve model for a iteration
        '''
        # Parse input variables
        state_buffer, action_buffer, reward_buffer, _  = transitions

        # Critic prediction
        inputs = [state_buffer, np.zeros((state_buffer.shape[0], 1))]
        critic_pred = self.predict(inputs, output_type="value")

        # Train on batch
        inputs = [state_buffer, reward_buffer]
        targets = [action_buffer, critic_pred]
        self.model.train_on_batch(x=inputs, y=targets)

    def predict(self, inputs, output_type="policy"):
        ''' Do model prediction

            'output_type' is a element of tuple ("parmas", "value")
        ''' 
        outputs = self.model.predict(inputs)
        if output_type == "policy":
            return np.split(outputs[0], 2, axis=-1)
        elif output_type == "value":
            return outputs[1]


class A2CStrategy(Strategy):
    ''' Actor-Critic algorithm
    '''
    def __init__(self, env,
                       gamma=.99,
                       init_learning_rate=1.e-4,
                       num_episode=5000,
                ): 
        # Initialize internal setup
        super(A2CStrategy, self).__init__(
            env,
            gamma=gamma,
            init_learning_rate=init_learning_rate,
            num_episode=num_episode
        )

        # Create actor and critic networks
        self.agent = A2CAgent(
            input_dim=self.input_dim, 
            output_dim=self.output_dim, 
            num_frame=self.num_frame,
            lr=self.init_learning_rate,
        )
        
        # Set up agent
        self.initialize()

    def get_next_action(self, state):
        ''' Use actor policy to get next action
        '''
        # Get Gaussian policy   
        mu, log_sigma = self.predict(
            (np.expand_dims(state, axis=0), np.zeros(shape=(1, 1))),
            output_type="policy"
        )

        # Sampling and symmetrically normalize along (-1, 1) 
        action = np.random.normal(mu, np.exp(log_sigma))
        action = np.clip(action, -1, 1).ravel()

        return action

    def get_discounted_reward(self, rewards):
        ''' Evaluate discounted reward
        '''
        cumulated_reward, discounted_rewards = 0, []
        for reward in reversed(rewards):
            cumulated_reward = reward + self.gamma*cumulated_reward
            discounted_rewards.append(cumulated_reward)

        return discounted_rewards[::-1]

    def train(self):
        ''' Train agent under A2C algorithm
        '''
        for episode in range(self.num_episode):
            # Reset episode
            curr_state = self.env.reset()
            action_buffer, state_buffer, reward_buffer = [], [], []
            b_terminal = False

            # Simulate episode
            while not b_terminal:
                print("iteration", episode, len(action_buffer))
                # Actor an action by following the current policy
                action = self.get_next_action(curr_state)
                
                # Retrieve next state and reward
                next_state, reward, b_terminal, _ = self.env.step(action)
                
                # Memorize (s, a, r) transitions
                action_buffer.append(action)
                reward_buffer.append(reward)
                state_buffer.append(curr_state)
                
                # Update current state
                curr_state = next_state

            # Evaluate discounted reward for a episode
            reward_buffer = self.get_discounted_reward(reward_buffer)

            # Manipulate buffers
            state_buffer = np.array(state_buffer)
            action_buffer = np.array(action_buffer)#np.stack([action_buffer], axis=1)
            reward_buffer = np.stack([reward_buffer], axis=1)
            transitions = (
                state_buffer, action_buffer, reward_buffer, b_terminal
            )
            
            # Evolve agents
            self.evolve(transitions)
            
    def trade(self):
        ''' Start stock trade
        '''
        # Initialize episode
        curr_state = self.env.reset()

        # Simulate episode
        while not b_terminal:
            # Get action
            action = self.get_next_action(curr_state)
            
            # Observe state
            next_state, _, _, _ = self.env.step(action)

            # Update current state
            curr_state = next_state
