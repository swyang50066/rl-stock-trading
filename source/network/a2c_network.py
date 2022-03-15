import  tensorflow                      as  tf
import  tensorflow.keras
import  tensorflow.keras.backend        as  K
from    tensorflow.keras.layers         import  Dense, Flatten 



class A2CNetwork(tf.keras.Model):
    def __init__(self, output_dim, mode="train"):
        super(A2CNetwork, self).__init__()
        # Operators
        self.flatten = Flatten()

        self.bn_dense0 = Dense(64, activation="relu")
        self.bn_dense1 = Dense(128, activation="relu")

        self.action_dense0 = Dense(128, activation="relu")
        self.action_dense1 = Dense(output_dim, activation="tanh")

        self.value_dense0 = Dense(128, activation="relu")
        self.value_dense1 = Dense(1, activation="linear")

    def call(self, inputs, **kwargs):
        ''' I/O dimensionality
                dim(input_state) = (num_frame, observation_dim) 
                dim(input_value) = (num_frame, 1)
                dim(output_action) = (num_frame, stock_dim)
                dim(output_value) = (num_value, 1)
        '''
        # Parse inputs
        input_state, input_value = inputs

        # Shared feed-forward network
        x = self.flatten(input_state)
        x = self.bn_dense0(x)
        x = self.bn_dense1(x)

        # Actor pathway
        output_action = self.action_dense0(x)
        output_action = self.action_dense1(output_action)

        # Critic pathway
        output_value = self.value_dense0(x)
        output_value = self.value(output_value)
  
        return [output_action, output_value]
