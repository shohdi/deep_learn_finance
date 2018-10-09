import numpy as np;
import math;
import collections

class ForexEnvironment:
    def __init__(self):
        self._states_coll = collections.deque(maxlen=4);
        self.initVars();
    
    def initVars(self):
        self._step_started = False;
        self._step_ended = False;
        self._last_action = 0;
        self._last_state = None;
        self._last_reward = 0;
        self._last_game_over = False;
        self._myRandomFrame = 0;

        
        
      
    

    def reset(self):
        self.initVars();
        self._step_started = True;
        
        
        i = 0;
        while(not self._step_ended):
            i= i+1;

        self._states_coll.append(self._last_state);
        return self._last_state,self._last_reward,self._last_game_over,None;
    

    def step(self,action):
        self.initVars();
        #print("we will start step ",self._step_started);
        self._step_started = True;
        #print("step started  ",self._step_started);
        self._last_action = action;
        
        i = 0;
        while(not self._step_ended):
            i= i+1;

        #print("step ended  ",self._step_started , self._step_ended);
        self._states_coll.append(self._last_state);

        return self._last_state,self._last_reward,self._last_game_over,None;
    

    def get_action_sample(self):
        ret  = int( math.floor( (np.random.random() * 4)));
        if(len(self._states_coll) > 0):
            if(self._last_state[0,5] > 0 or self._last_state[0,4] > 0):
                self._myRandomFrame = self._myRandomFrame + 1;
                ret = int( math.floor( (np.random.random() * 3000)));
                '''
                lastClose = self._last_state[0,3];
                up = self._last_state[0,4];
                down = self._last_state[0,5];

                if(up > 0 and (lastClose - up) >= 0.0005):
                    ret = 3;
                elif(down > 0 and (down - lastClose) >= 0.0005):
                    ret = 3;
                '''
                if(self._myRandomFrame > 2000):
                    if(self._last_state[0,4] > 0 and self._last_state[0,3] > self._last_state[0,4]):
                        ret = 3;
                    elif(self._last_state[0,5] > 0 and self._last_state[0,3] < self._last_state[0,5]):
                        ret = 3;
                

            else:
                self._myRandomFrame = 0;
        
        if(ret > 3):
            ret = 0;

        return ret;


