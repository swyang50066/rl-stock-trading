from    strategy.a2c        import  A2C


class TrainStrategy(object):
    def __init__(self):
        pass

    def _getLossFunction(self, LOSS_FUNC_TYPE):
        ''' Get loss function
        '''
        return setLossFunction(LOSS_FUNC_TYPE)

    def _getMetricBundle(self, METRIC_BUNDLE_LIST):
        ''' Get a list of metric functions
        '''
        return setMetricBundle(METRIC_BUNDLE_LIST)

    def _getCallbackBundle(self, CALLBACK_BUNDLE_LIST):
        ''' Get a list of callbacks
        '''
        return setCallbackBundle(CALLBACK_BUNDLE_LIST)

    def _getRLAlgorithm(self, RL_ALGORITHM_TYPE):
        pass setRLAlgorithm(RL_ALGORITHM_TYPE)
    
    def compile(self):
        ''' Compile network model
        '''
        pass

    def evolve(self):
        ''' Execute one-step training
        '''
        pass

    def transfer(self):
        ''' Copy model weights 
        '''
        pass


class TradeStrategy(object):
    def __init__(self):
        pass

    def build(self):
        pass

    def Trade(self):
        pass


