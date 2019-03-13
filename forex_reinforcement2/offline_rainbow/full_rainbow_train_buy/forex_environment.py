import numpy as np;
import math;
import collections
import gym
import gym.spaces
from gym.utils import seeding
import ptan
from enum import Enum


COMPRESS_LEVEL = 60 * 4

class OneDataPrices:
    def __init__(self):
        self.high = [];
        self.low = [];
        self.open = [];
        self.close = [];
        self.avgm = [];
        self.avgh = [];
        self.avgd = [];



    

class MyDataPos(Enum):
    HIGH = 0,
    LOW = 1,
    OPEN = 2,
    CLOSE = 3,
    AVGM = 4,
    AVGH = 5,
    AVGD = 6,
    MONTH= 7,
    DAY_M = 8,
    DAY_W = 9,
    HOUR = 10,
    MIN = 11,
    ASK = 12,
    BID = 13


class Actions(Enum):
    Skip = 0
    Buy = 1
    Close = 2

class ForexEnvironment(gym.Env):
    def __init__(self):
        self.bars_count = 50;
        self.return_1_d = False;
        self.action_space = gym.spaces.Discrete(n=len(Actions))

        self.action_space.sample = (lambda : self.mySample())
        
        
        
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32)
        self._states_coll = collections.deque(maxlen=4);
        self.initVars();
    
    @property
    def shape(self):
        if(self.return_1_d):
            return self.shape1d()
        # [h, l, c] * bars + position_flag + rel_profit (since open)
        return (8*self.bars_count + 1 + 1, )
    
    def shape1d(self):
        return (10, self.bars_count)
    
    def initVars(self):
        self._step_started = False;
        self._step_ended = False;
        self._last_action = 0;
        self._last_state = None;
        self._last_reward = 0;
        self._last_game_over = False;
        self._myRandomFrame = 0;
        self._last_im_reward = 0;
        self._last_pos = 0;

        
    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]

    def mySample(self):
        ret = int( math.floor( (np.random.random() * 30)))
        if(ret > 2):
            ret = 0
        return ret        
      
    

    def reset(self):
        return self.step(0)
        self.initVars();
        self._step_started = True;
        
        
        i = 0;
        while(not self._step_ended):
            i= i+1;

        self._states_coll.append(self._last_state);
        state = self.encode(self._last_state)
        return  state,self._last_reward,self._last_game_over,self._last_pos;
    

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
        state = []
        if(self._last_pos == 0):
            state = self.encode(self._last_state)
        return  state,self._last_reward,self._last_game_over,self._last_pos;
    

    def getPrices(self,arr):
        prices = OneDataPrices()
        prices.high =  arr[:,MyDataPos.HIGH.value[0]]
        prices.low = arr[:,MyDataPos.LOW.value[0]]
        prices.close = arr[:,MyDataPos.CLOSE.value[0]]
        prices.open = arr[:,MyDataPos.OPEN.value[0]]
        prices.avgd = arr[:,MyDataPos.AVGD.value[0]]
        prices.avgm = arr[:,MyDataPos.AVGM.value[0]]
        prices.avgh = arr[:,MyDataPos.AVGH.value[0]]
        return prices;




    def encode(self,arr):
        prices = self.getPrices(arr);


        min,max,minAvg,maxAvg = self.getMaxMin(prices)
        deviaAvg = maxAvg - minAvg
        devia = max-min
        """
        Convert current state into numpy array.
        """

        res = np.ndarray(shape=self.shape, dtype=np.float32)
        ofs = (self.bars_count*COMPRESS_LEVEL)-1


        high = np.array(prices.high,dtype=np.float32).reshape((-1,COMPRESS_LEVEL))
        low = np.array(prices.low,dtype=np.float32).reshape((-1,COMPRESS_LEVEL))
        open = np.array(prices.open,dtype=np.float32).reshape((-1,COMPRESS_LEVEL))
        close  = np.array(prices.close,dtype=np.float32).reshape((-1,COMPRESS_LEVEL))
        avgm  = np.array(prices.avgm,dtype=np.float32).reshape((-1,COMPRESS_LEVEL))
        avgh  = np.array(prices.avgh,dtype=np.float32).reshape((-1,COMPRESS_LEVEL))
        avgd  = np.array(prices.avgd,dtype=np.float32).reshape((-1,COMPRESS_LEVEL))

        high = high.max(1)
        low = low.min(1)
        close = close[:,-1]
        open = open[:,0]
        avgm = avgm[:,-1]
        avgh = avgh.mean(1)
        avgd = avgd.mean(1)




        shift = 0
        for bar_idx in range(0,len(high)):
            #'high','low','open','close','avgm','avgh','avgd','month','dayofmonth','dayofweek','hour','minute','ask','bid','volume'
            res[shift] = (high[bar_idx] - min)/devia
            shift += 1
            res[shift] = (low[bar_idx] - min)/devia
            shift += 1
            res[shift] = (open[bar_idx] - min)/devia
            shift += 1
            res[shift] = (close[bar_idx] - min)/devia
            shift += 1
            res[shift] = (avgm[bar_idx] - minAvg)/deviaAvg
            shift += 1
            res[shift] = (avgh[bar_idx] - minAvg)/deviaAvg
            shift += 1
            res[shift] = (avgd[bar_idx] - minAvg)/deviaAvg
            shift += 1
            res[shift] = (close[bar_idx] - minAvg)/deviaAvg
            shift += 1
            #res[shift] = self._prices.month[self._offset + bar_idx]
            #shift += 1
            #res[shift] = self._prices.dayofmonth[self._offset + bar_idx]
            #shift += 1
            #res[shift] = self._prices.dayofweek[self._offset + bar_idx]
            #shift += 1
            #res[shift] = self._prices.hour[self._offset + bar_idx]
            #shift += 1
            #res[shift] = self._prices.minute[self._offset + bar_idx]
            #shift += 1
            
        res[shift] = float(self._last_pos) 
        shift += 1
        res[shift] = self._last_reward
        
        
        return res        
        
        
    def getMaxMin(self,prices):
        max = 0.0;
        min = 9999999.0;
        maxAvg = 0.0;
        minAvg = 9999999.0;
        offset = np.array(prices.high).shape[0]-1
        for bar_idx in range(-(self.bars_count * COMPRESS_LEVEL)+1 , 1):
            val = prices.high[offset + bar_idx]
            if(val > 0 and val < min):
                min = val
            if(val > max):
                max = val;
            val = prices.low[offset + bar_idx]
            if(val > 0 and val < min):
                min = val
            if(val > max):
                max = val;
            
            val = prices.avgd[offset + bar_idx]
            if(val > 0 and val < minAvg):
                minAvg = val
            if(val > maxAvg):
                maxAvg = val;
            
            val = prices.avgh[offset + bar_idx]
            if(val > 0 and val < minAvg):
                minAvg = val
            if(val > maxAvg):
                maxAvg = val;
            
            val = prices.avgm[offset + bar_idx]
            if(val > 0 and val < minAvg):
                minAvg = val
            if(val > maxAvg):
                maxAvg = val;

            val = prices.close[offset + bar_idx]
            if(val > 0 and val < minAvg):
                minAvg = val
            if(val > maxAvg):
                maxAvg = val;
            
        return min,max,minAvg,maxAvg


    def normArray(self,arr):
        arr = np.array(arr);
        max = np.amax(arr);
        
        noZero = np.array( [i if i > 0.0 else max for i in arr.flatten()]);
        min = np.amin(noZero);
        arr[arr == 0] = min;
        ret = np.array((arr-min)/(max-min),dtype=np.float32);
        return ret;
    

    def normCustomArray(self,arr):
        arr = np.array(arr);
        max = np.amax(arr[:,0:8]);
        
        noZero = np.array( [i if i > 0.0 else max for i in arr[:,0:8].flatten()]);
        min = np.amin(noZero);
        for i in range(len(arr)):
            for j in range(8):
                lnum = arr[i][j];
                if(lnum == 0):
                    lnum = min;
                arr[i][j] = ((lnum-min)/(max-min));
            arr[i][8] = 0;
        


        ret = np.array(arr,dtype=np.float32);
        ret = np.reshape(ret,(140,));
        return ret;






    def get_action_sample(self):
        ret  = int( math.floor( (np.random.random() * 4)));
        
        if(len(self._states_coll) > 0):
            if(self._last_state[5] > 0 or self._last_state[4] > 0):
                self._myRandomFrame = self._myRandomFrame + 1;
                ret = int( math.floor( (np.random.random() * 15)));
                '''
                lastClose = self._last_state[0,3];
                up = self._last_state[0,4];
                down = self._last_state[0,5];

                if(up > 0 and (lastClose - up) >= 0.0005):
                    ret = 3;
                elif(down > 0 and (down - lastClose) >= 0.0005):
                    ret = 3;
                '''
                if(self._last_im_reward > 1):
                    ret = 3;
                elif(self._myRandomFrame > 10):
                    if(self._last_state[4] > 0 and self._last_state[3] > self._last_state[4]):
                        ret = 3;
                    elif(self._last_state[5] > 0 and self._last_state[3] < self._last_state[5]):
                        ret = 3;
                

            else:
                self._myRandomFrame = 0;
        
        if(ret > 3):
            ret = 0;

        return ret;


