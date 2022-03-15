import  tensorflow                      as  tf
import  tensorflow.keras
import  tensorflow.keras.backend        as  K
from    tensorflow.keras.layers         import  Dense, Flatten 



class A2CNetwork(tf.keras.Model):
    def __init__(self, mode="train"):
        super(A2CNetwork, self).__init__()

        # Operators
        self.flatten = Flatten()

        self.bn_dense0 = Dense(64, activation="relu")
        self.bn_dense1 = Dense(128, activation="relu")

        self.action_dense0 = Dense(128, activation="relu")
        self.action_dense1 = Dense(output_shape, activation="softmax")

        self.value_dense0 = Dense(128, activation="relu")
        self.value_dense1 = Dense(1, activation="linear")


    def call(self, inputs, **kwargs):
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
