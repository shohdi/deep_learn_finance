import numpy as np
import math
from forex_environment import ForexEnvironment 


class ForexAgent:
    def __init__(self,env):
        self.env = env;
        self._started = False;
    

    def mainLoop(self):
        while True:
            #print(self.env.get_action_sample());
            s_t,r_t,g_m,im_r = self.env.step(self.env.get_action_sample());
            if(not self._started and im_r != 0):
                self._started = True;
                print("state : ",s_t," reward ",r_t," game over ",g_m," im reward ",im_r);
            #print(self.env.get_action_sample());
        
