import  numpy                   as  np

import  tensorflow              as  tf
import  keras.backend           as  K
from    keras                   import  initializers, layers
from    keras.regularizers      import  l2
from    keras.layers            import  Conv3D, MaxPool3D, Dense, LeakyReLU


def flatten(x):
    ''' Flatten feature
    '''
    return K.reshape(x, (-1, np.prod(x.shape[1:])))


class ConvLayer(tf.keras.layers.Layer):
    def __init__(self, features, kernel, stride,
                       kernel_initializer="he_normal",
                       kernel_regularizer=l2(1.e-6),
                       activation=LeakyReLU(), 
                       padding="same", 
                       **kwargs):
        super(ConvLayer, self).__init__(**kwargs)
        
        # Parameters
        self.features = features
        self.kernel = kernel
        self.stride = stride
        self.strides = (stride, stride, stride)

        # Operators
        self.conv3d = Conv3D(filters=self.features,
                             kernel_size=self.kernel,
                             kernel_initializer=kernel_initializer,
                             kernel_regularizer=kernel_regularizer,
                             padding=padding,
                             strides=self.strides)
        self.activation = activation

    def call(self, x, **kwargs):
        return self.activation(self.conv3d(x))

    def compute_output_shape(self, shape):
        return (shape[0],) + self.strides + (self.features,)


class Encoder(tf.keras.Model):
    def __init__(self):
        super(Encoder, self).__init__()

        # Zeroth convolution block
        self.conv0 = ConvLayer(features=32, 
                               kernel=5, 
                               stride=1, 
                               name="conv0")
        self.pool0 = MaxPool3D(pool_size=2,
                               name="pool0")

        # First convolution block
        self.conv1 = ConvLayer(features=32, 
                               kernel=5, 
                               stride=1, 
                               name="conv1")
        self.pool1 = MaxPool3D(pool_size=2,
                               name="pool1")

        # Second convolution block
        self.conv2 = ConvLayer(features=64, 
                               kernel=5, 
                               stride=1, 
                               name="conv2")
        self.pool2 = MaxPool3D(pool_size=2,
                               name="pool2")

        # Final convolution
        self.conv3 = ConvLayer(features=64, 
                               kernel=3, 
                               stride=1, 
                               name="conv3")

    def call(self, x, **kwargs):
        # Encoding
        c0 = self.conv0(x)
        p0 = self.pool0(c0)
    
        c1 = self.conv1(p0)
        p1 = self.pool1(c1)

        c2 = self.conv2(p1)
        p2 = self.pool2(c2)

        c3 = self.conv3(p2)

        # Flatten
        enc = flatten(c3)

        return enc


class DenseLayer(tf.keras.layers.Layer):
    def __init__(self, units,
                       kernel_initializer="he_normal",
                       kernel_regularizer=l2(1.e-6),
                       activation=LeakyReLU(),
                       name="dense-block",
                       **kwargs):
        super(DenseLayer, self).__init__(**kwargs)

        # Parameters
        self.units = units

        # Operators
        self.dense = Dense(units=self.units,
                            kernel_initializer=kernel_initializer,
                            kernel_regularizer=kernel_regularizer,
                            name=name)
        self.activation = activation

    def call(self, x, bActivation=True, **kwargs):
        # Transform next layer's dimensions
        if bActivation:
            return self.activation(self.dense(x))
        else:
            return self.dense(x)

    def compute_output_shape(self, shape):
        return (shape[0], self.units)


class Decoder(tf.keras.Model):
    def __init__(self, agents):
        super(Decoder, self).__init__()
 
        # Set paremeter
        self.agents = agents

        # Fully-connected layers
        self.denseBlock0 = [DenseLayer(units=512, name="dense-block0-%d"%i)
                            for i in range(agents)]
        self.denseBlock1 = [DenseLayer(units=256, name="dense-block1-%d"%i)
                            for i in range(agents)]
        self.denseBlock2 = [DenseLayer(units=128, name="dense-block2-%d"%i)
                            for i in range(agents)]
        self.denseBlock3 = [DenseLayer(units=6, name="dense-block3-%d"%i)
                            for i in range(agents)]

    def call(self, xs, **kwargs):
        ''' (numBatch, numAgents, features)
            (numBatch, numAgents, 6)
        '''       
        # Zero-th communication block
        comm0 = tf.reduce_mean(xs, axis=1)
        d0 = [self.denseBlock0[i](tf.concat((xs[:, i], comm0), axis=-1)) 
              for i in range(self.agents)]
        d0 = tf.stack(d0, axis=1)

        # First communication block
        comm1 = tf.reduce_mean(d0, axis=1)
        d1 = [self.denseBlock1[i](tf.concat((d0[:, i], comm1), axis=-1)) 
              for i in range(self.agents)]
        d1 = tf.stack(d1, axis=1)   
     
        # Second communication block
        comm2 = tf.reduce_mean(d1, axis=1)
        d2 = [self.denseBlock2[i](tf.concat((d1[:, i], comm2), axis=-1)) 
              for i in range(self.agents)]
        d2 = tf.stack(d2, axis=1)

        # Zero-th communication block
        comm3 = tf.reduce_mean(d2, axis=1)
        d3 = [self.denseBlock3[i](tf.concat((d2[:, i], comm3), axis=-1), 
                                  bActivation=False) 
              for i in range(self.agents)]
        d3 = tf.stack(d3, axis=1)

        return d3


class Network(tf.keras.Model):
    def __init__(self, agents, bTrain=True):
        super(Network, self).__init__()

        # Set parameter
        self.bTrain = bTrain

        # Operators
        self.encoder = Encoder()
        self.decoder = Decoder(agents=agents)

    def call(self, inputs, **kwargs):
        ''' dim(states) = (numBatch, 
                           numAgent, 
                           height, width, depth, 
                           numFrame)
        '''
        # Get states and actions
        states, actions = inputs["states"], inputs["actions"]

        # Normalization
        states = states / 255.
        
        # Encoding
        h0 = [self.encoder(states[:, i]) for i in range(states.shape[1])]
        h0 = tf.stack(h0, axis=1)    # (batchs, agents, features)

        # Decoding
        out = self.decoder(h0) 
      
        # Find expectation of maximum q-values 
        if self.bTrain and K.all(actions >= 0):
            onehot = tf.one_hot(K.flatten(actions), out.shape[-1], 1., 0.)
            out = K.reshape(out, (-1, out.shape[-1]))
            out = tf.reduce_sum(out * onehot, axis=1)    
            out = K.reshape(out, (-1, actions.shape[-1])) 

        return out
