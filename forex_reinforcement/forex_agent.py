import numpy as np
import math
from forex_environment import ForexEnvironment 


class ForexAgent:
    def __init__(self,env):
        self.env = env;
    

    def mainLoop(self):
        #print(self.env.get_action_sample());
        self.env.step(self.env.get_action_sample());
        #print(self.env.get_action_sample());
        
