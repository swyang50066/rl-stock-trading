from    strategy.DDPG       import  DDPG
from    strategy.A2C        import  A2C
from    strategy.A3C        import  A3C
from    strategy.TD3        import  TD3
from    strategy.PPO        import  PPO
from    strategy.SAC        import  SAC
from    strategy.TRPO       import  TRPO


class Strategy(object):
    def __init__(self):
        pass

    def getLossFunction(self):
        pass

    def getMetricBundle(self):
        pass

    def getCallbackBundle(self):
        pass

    def getNetworkModel(self):
        pass

    def compile(self):
        pass

    def evolve(self):
        pass

    def transfer(self):
        pass

