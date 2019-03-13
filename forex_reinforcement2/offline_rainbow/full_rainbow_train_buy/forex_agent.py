import numpy as np
import math
import os
import collections
from forex_environment import ForexEnvironment 
from keras import backend as K
from keras.models import Sequential
from keras.layers.convolutional import Conv2D,Conv1D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import RepeatVector
from keras.layers.core import Reshape
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import SGD,RMSprop,Adam
from keras.layers.recurrent import LSTM
from keras.layers import ConvLSTM2D
from keras.layers.cudnn_recurrent import CuDNNLSTM
from keras.losses import mean_squared_error
import pickle
import torch
from lib import model

from lib import environ
import ptan

IS_TEST_RUN = True
#initialize parameters
DATA_DIR= os.path.join(".","data")
NUM_ACTIONS = 4 #number of valid actions (0 do nothing , 1 trade down , 2 trade up , 3 close trade)
GAMMA = 0.99 # decay rate of past observations
INITIAL_EPSILON = 1 # starting value of epsilon
FINAL_EPSILON = 0.1 # final value of epsilon
MEMORY_SIZE = 5000000 # number of previous transitions to remember
NUM_EPOCHS_OBSERVE = 100
NUM_EPOCHS = 259200

BATCH_SIZE = 100

ITERATE_COPY_Q = 1000;








class ForexAgent:
    def __init__(self,env):
        self.env = env;
        self._started = False;
        self.shape = self.env.shape
        


        self.num_games,self.num_wins = 0,0



    def buildModel(self):
        net = model.RainbowDQN(self.env.observation_space.shape, self.env.action_space.n).to('cpu')

        
        modelSave = os.path.join(DATA_DIR,"model.data")
        modelExists = os.path.isfile(modelSave);
        if modelExists:
            print('found model file , loading ...');
            net.load_state_dict(torch.load(modelSave))
            net.eval()
            print('end found model file , loading ...');
        
        ret = ptan.agent.DQNAgent(lambda x: net.qvals(x), ptan.actions.ArgmaxActionSelector(), device='cpu')
        return  ret;
        

    def createModels(self):
        self.model = self.buildModel();
       
        
    def mainLoop(self):
        self.createModels();
        e = 0;
        state_count = 0;
        
        while True:
            e = e+1 ;


            #get first state
            a_0 = 0 # (0= left , 1 = buy , 2 = close)
            s_t , r_0 , game_over,info = self.env.step(a_0)
            
            s_tm1 = s_t
            a_t,r_t = None,None
            
            while not game_over:
                s_tm1 = s_t
                pos = info
                if pos == 0:
                    obs_v = [s_tm1]
                    out_v,_ = self.model(obs_v)
                    action_idx = out_v[0]
                    a_t = action_idx;
                else:
                    a_t = 0

 


                #apply action , get reward
                s_t , r_t , game_over ,info= self.env.step(a_t)
                state_count+=1
                #if reward , increment num_wins
                if r_t > 0 and game_over :
                    self.num_wins +=1


            
            print("Epoch {:04d}/{:d} | loss {:.5f} | Win count {:d} | current epsilon {:.5f} | last reward {:.5f}".format(e + 1,NUM_EPOCHS,0,self.num_wins,0,r_t))

            


        
