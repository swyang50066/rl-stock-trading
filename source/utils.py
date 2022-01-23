import  os
import  json
from    collections     import  defaultdict


## Features
CONFIG_TRAIN_IO_FEATURES = {
    "ticker_list": [
        "AAPL"
    ],
}
DEFAULT_TRAIN_CONFIG_FEATURES = defaultdict(bool, {
    **CONFIG_TRAIN_IO_FEATURES,
})


def decoder(obj):
    ''' decode object recursively
    '''
    for key, value in obj.items():
        if isinstance(value, dict):
            for subkey, subvalue in decoder(value):
                yield (subkey, subvalue)
        else:
            yield (key,value)


def getConfig(obj, EXEC_MODE):
    def _update(key, value):
        ''' Update value of default dictionary
        '''
        # ====> Some condition to verify value
        return value

    # Build feature dictionary
    DEFAULT_CONFIG_FEATURES = (
        DEFAULT_TRAIN_CONFIG_FEATURES if EXEC_MODE == "train"
        else DEFAULT_TEST_CONFIG_FEATURES
    )     

    # Build data base info dictionary
    for key, value in decoder(obj):
        DEFAULT_CONFIG_FEATURES[key] = _update(key, value)
   
    return DEFAULT_CONFIG_FEATURES


def setConfig(EXEC_MODE="train"):
    ''' Parsing configuration file (.json), one sets model environment
    '''
    # Set path of configuration file
    JSON_HOME_PATH = (
        os.path.dirname(os.path.realpath(__file__))
    if EXEC_MODE == "train":
        path = JSON_HOME_PATH + "/train_config.json"
    elif EXEC_MODE == "test":
        path = JSON_HOME_PATH + "/test_config.json"
    
    # Parse configuration from .json
    with open(path) as f:
        obj = json.load(f)

    # Update configuration
    config = getConfig(obj, EXEC_MODE)

    return config
