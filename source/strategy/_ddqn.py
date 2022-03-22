import  copy

import  numpy       as  np

import  tensorflow              as  tf
import  keras.backend           as  K
from    keras                   import  Input, Model
from    keras.optimizers        import  Adam

from    network.commnet         import  Network
from    replay                  import  Replay
from    utils                   import  Scheduler, Logger, Saver


class DDQN(object):
    def __init__(self, numBatch,
                       numAgent, 
                       numFrame, 
                       numAction, 
                       kernel_shape,
                       gamma=.9,
                       lr=1.e-6,
                       maxStepPerEpisode=1):
        # Set parameters
        self.numBatch = numBatch
        self.numAgent = numAgent
        self.numFrame = numFrame
        self.numAction = numAction
        self.kernel_shape = kernel_shape

        self.lr = lr
        self.gamma = gamma

        self.maxStepPerEpisode = maxStepPerEpisode

        # Build model
        self.__build()    

        # Compile model
        self.__compile()

        # Transfer weights from q-network to target-network
        self.transfer()

    def __loss(self, true, pred):
        ''' Loss function
        '''
        # Declare Huber loss
        funcLoss = tf.losses.Huber(reduction=tf.losses.Reduction.AUTO)

        return funcLoss(true, pred)

    def __build(self):
        ''' Build model
        '''
        # Define I/O shapes
        shape = (self.numAgent,) + self.kernel_shape + (self.numFrame,)
        inputs = {"states": Input(shape=shape, 
                                  dtype=tf.float32), 
                  "actions": Input(shape=(self.numAgent,),
                                   dtype=tf.int32)}
        q_outputs = Network(self.numAgent)(inputs)
        target_outputs = Network(self.numAgent, bTrain=False)(inputs)

        # Define networks
        self.q_network = Model(inputs=inputs, outputs=q_outputs)
        self.target_network = Model(inputs=inputs, outputs=target_outputs)

        # Freeze target_network
        self.target_network.trainable = False

        # Define callbacks
        self.callbacks = [Scheduler(),
                          Logger(),
                          Saver(self.q_network,
                                self.maxStepPerEpisode)]

    def __compile(self):
        ''' Compile model
        '''
        # Define optimizer
        optimizer = Adam(self.lr, beta_1=.99, beta_2=.999)

        # Compile operators
        self.q_network.compile(optimizer=optimizer, loss=self.__loss)
        self.target_network.compile(optimizer=optimizer, loss=self.__loss)
        
    def evolve(self, transitions):
        ''' Evolve network model

            transitions are tuple of shape
                (states, actions, rewards, next_states, dones)
        '''
        # Encode transitions
        currStates = transitions[0]
        currActions = transitions[1]
        currRewards = transitions[2]
        nextStates = transitions[3]
        bTerminal = transitions[4]

        # Get maximum q-value of next state
        inputs = {"states": nextStates, "actions": currActions}
        nextQvalue = self.target_network.predict(inputs)
        nextQvalue = K.reshape(nextQvalue, (-1, self.numAction))
        maxNextQvalue = tf.reduce_max(nextQvalue, axis=1)    # N = batchs*agents

        # Get target
        activate = 1. - bTerminal.ravel().astype(np.float32)
        returns = activate*self.gamma*tf.stop_gradient(maxNextQvalue)
        returns = currRewards.ravel() + returns
        returns = K.reshape(returns, currStates.shape[:2])

        # Evolve network weights
        inputs={"states": currStates, "actions": currActions}
        self.q_network.fit(x=inputs,
                           y=returns,
                           verbose=0,
                           epochs=1,
                           callbacks=self.callbacks)

    def transfer(self):
        ''' Copy model weights from q-network to target-network
        '''
        self.target_network.set_weights(self.q_network.get_weights())


class Agent(object):
    def __init__(self, environment,
                       numBatch=4,
                       numFrame=4,
                       gamma=.9,
                       lr=1.e-6,
                       epsilon=1,
                       minEps=.1,
                       delta=.001, 
                       maxEpisode=100,
                       maxStepPerEpisode=50,
                       numEpoch=1,
                       updateFrequency=12,
                       initMemorySize=1e4,
                       numMemory=1e5,
                       numStepPerTrain=1):
        # Set input parameters
        self.environment = environment
        
        # Set training frequencies
        self.numStepPerTrain = numStepPerTrain
        self.maxEpisode = maxEpisode
        self.maxStepPerEpisode = maxStepPerEpisode
        self.initMemorySize = int(initMemorySize)
        self.numEpoch = numEpoch
        self.updateFrequency = updateFrequency     
   
        # Set agent configuration
        self.numBatch = numBatch
        self.numAgent = environment.numAgent
        self.numAction = environment.numAction
        self.numFrame = numFrame

        self.epsilon = epsilon
        self.minEps = minEps
        self.delta = delta

        # Define replay memory
        self.replay = Replay(self.numAgent,
                             int(numMemory),
                             environment.kernel_shape,
                             self.numFrame)
        
        # Define network model
        self.dqn = DDQN(self.numBatch,
                        self.numAgent, 
                        self.numFrame, 
                        self.numAction,
                        environment.kernel_shape, 
                        gamma=gamma,
                        lr=lr,
                        maxStepPerEpisode=maxStepPerEpisode)

    def __initMemoryBuffer(self):
        ''' Initialize memory buffer
        '''
        while len(self.replay) < self.initMemorySize:
            # Reset the environment for the start of the episode.
            states = self.environment.reset()
            bTerminal = [False for _ in range(self.numAgent)]
            
            for _ in range(self.maxStepPerEpisode):
                # Get action and q-value
                actions, qvalues = self.__getNextAction(states)
               
                # Update environment status
                states, rewards, bTerminal, info = self.environment.step(
                    actions, qvalues, bTerminal)

                # Append buffer
                self.replay.append((states, actions, rewards, bTerminal))
               
                # Break if all agents are finished 
                if np.all(bTerminal):
                    break

    def __getNextAction(self, states):
        ''' Return next actions, following self.epsilon-greedy policy
        '''
        # Determine next query based on self.epsilon-greedy policy
        if np.random.random() < self.epsilon:
            actions = np.random.randint(self.numAction, size=self.numAgent)
            qvalues = np.zeros((self.numAgent, self.numAction))
        else:        
            actions, qvalues = self.__getGreedyAction(states)

        return actions, qvalues

    def __getGreedyAction(self, states, bDoubleLearning=True):
        ''' Greedy-policy
        '''
        # Get qvalues
        inputs = {"states": states, 
                  "actions": -np.ones((1, self.numAgent))}
        if bDoubleLearning:
            qvalues = self.dqn.q_network.predict(inputs)
        else:
            qvalues = self.dqn.target_network.predict(inputs)

        # Remove batch domension
        qvalues = np.squeeze(qvalues, axis=0)
        
        # Get actions
        actions = np.argmax(qvalues, axis=-1).flatten()

        print("GREEDY-ACTION")
        print("\t >>> Actions", actions)
        print("\t >>> Q-values", qvalues)
        print("----")

        return actions, qvalues
        
    def train(self):
        ''' Train agent
        '''
        # Initialize memory buffer
        self.__initMemoryBuffer()
        
        # Run play
        numEpisode = 1
        decayedStepPerEpisode = self.maxStepPerEpisode
        while numEpisode <= self.maxEpisode:
            # Reset environment before an episode starts
            states = self.environment.reset()
            bTerminal = [False for _ in range(self.numAgent)]
           
            # Simulation 
            for numStep in range(decayedStepPerEpisode):
                print("Training", numStep+1, decayedStepPerEpisode)
                # Get next actions and q-values
                actions, qvalues = self.__getNextAction(
                    self.replay.getRecentState())

                # Step the agent once, and get the transition tuple
                states, rewards, bTerminal, info = self.environment.step(
                    np.copy(actions), qvalues, bTerminal)

                # Append memory buffer
                self.replay.append((states, actions, rewards, bTerminal))
                
                if numStep % self.numStepPerTrain == 0:
                    # Get batch tuple
                    batch = self.replay.sample(self.numBatch)

                    # Evolve network
                    self.dqn.evolve(batch)

                # Check if all agents arrive at the targets
                if np.all(bTerminal):
                    break
          
            # Copy weights to target network 
            if (numEpisode*self.numEpoch) % self.updateFrequency == 0:
                self.dqn.transfer()
            
            # Decay self.epsilon
            self.epsilon = max(self.minEps, self.epsilon - self.delta)
            
            # Count episode
            numEpisode += 1

            # Linearly decay maximum step per episode
            decayedStepPerEpisode = int(self.maxStepPerEpisode*(1 - .95*(numEpisode/self.maxEpisode)))

            print("-----")
            print("EPISODE (%d/%d)"%(numEpisode, self.maxStepPerEpisode))
            print("TARGET", info["TARGET_0"], info["TARGET_1"])
            print("LOCATION", info["LOCATION_0"], info["LOCATION_1"])
            print("DISTANCE", info["DISTANCE_0"], info["DISTANCE_1"])
            print("-----")


