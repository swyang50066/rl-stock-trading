import  keras.backend       as  K
from    keras.models        import  Model
from    keras.layers        import  Input, Dense, Flatten


def build_feed_forward_network(input_dim):
    ''' Build feed-forward network model
    '''
    inputs = Input((input_dim))
    x = Flatten()(inputs)
    x = Dense(64, activation="relu")(x)
    outputs = Dense(128, activation="relu")(x)

    return Model(inpupts=inputs, outputs=outputs)
