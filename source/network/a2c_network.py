import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Flatten


class A2CNetwork(tf.keras.Model):
    def __init__(self, output_dim, mode="train"):
        super(A2CNetwork, self).__init__()

        # Operators
        self.bn_dense0 = Dense(
            64, activation="relu", name="bottle_neck_dense0"
        )
        self.bn_dense1 = Dense(
            128, activation="relu", name="bottle_neck_dense1"
        )

        self.action_dense0 = Dense(
            128, activation="relu", name="action_dense0"
        )
        self.action_dense1 = Dense(
            output_dim, activation="tanh", name="action_dense1"
        )
        self.action_dense2 = Dense(
            output_dim, activation="softplus", name="action_dense2"
        )

        self.value_dense0 = Dense(128, activation="relu", name="value_dense0")
        self.value_dense1 = Dense(1, activation="linear", name="value_dense1")

    def call(self, inputs, **kwargs):
        """I/O dimensionality
        
        dim(input_state) = (num_frame, observation_dim)
        dim(input_value) = (num_frame, 1)
        dim(output_mu) = (num_frame, stock_dim)
        dim(output_sigma) = (num_frame, stock_dim)
        dim(output_value) = (num_value, 1)
        """
        # Parse inputs
        input_state, input_value = inputs

        # Shared feed-forward network
        x = self.bn_dense0(input_state)
        x = self.bn_dense1(x)

        # Actor pathway
        x0 = self.action_dense0(x)
        output_mu = self.action_dense1(x0)
        output_log_sigma = self.action_dense2(x0)

        # Critic pathway
        output_value = self.value_dense0(x)
        output_value = self.value_dense1(output_value)

        return [output_mu, output_log_sigma, output_value]
