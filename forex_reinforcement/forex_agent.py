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
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import SGD,RMSprop,Adam
from keras.layers.recurrent import LSTM
from keras.layers import ConvLSTM2D
from keras.layers.cudnn_recurrent import CuDNNLSTM
from keras.losses import mean_squared_error

#initialize parameters
DATA_DIR= os.path.join(".","data")
NUM_ACTIONS = 4 #number of valid actions (left , stay , right)
GAMMA = 0.99 # decay rate of past observations
INITIAL_EPSILON = 1 # starting value of epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
MEMORY_SIZE = 750000 # number of previous transitions to remember
NUM_EPOCHS_OBSERVE = 100
NUM_EPOCHS = 100000

BATCH_SIZE = 32

ITERATE_COPY_Q = 2000;








class ForexAgent:
    def __init__(self,env):
        self.env = env;
        self._started = False;
        
        self.createModels();
        self.experience  = collections.deque(maxlen=MEMORY_SIZE)
        self.last_ex = collections.deque(maxlen=MEMORY_SIZE);
        self.fout = open(os.path.join(DATA_DIR,"rl-network-results.tsv"),"wb")
        self.num_games,self.num_wins = 0,0
        self.epsilon = INITIAL_EPSILON


    def buildModel(self):
        shape = (100,6);
        
        
        print ('print model input shape : ',shape);
        model = Sequential()
        model.add(CuDNNLSTM(600,input_shape=shape ,return_sequences=True ))#, dropout=0.2, recurrent_dropout=0.2))
        model.add(CuDNNLSTM(600,return_sequences=True ))
        model.add(CuDNNLSTM(600 ,return_sequences=True ))
        model.add(CuDNNLSTM(600 ))
       
        
        model.add(Dense(4,kernel_initializer="normal"))
        model.compile(optimizer=Adam(lr=1e-6),loss="mse")
        model.summary();
        return model;


    def createModels(self):
        self.model = self.buildModel();
        self.model1 = self.buildModel();
        self.copyModelWeights(self.model,self.model1);
        test = np.zeros((100,6));
        
        print("input ",test)
        out = self.model.predict(np.expand_dims(test, axis=0));
        print("out of model like : ",out)
        out = self.model.predict(np.expand_dims(test, axis=0));
        print("out of model like : ",out)

    def get_next_batch(self,experience,model,num_actions,gamma,batch_size):
        model = self.model;
        model1 = self.model1;
        batch_indices = np.random.randint(low=0,high=len(experience),size=batch_size)
        batch = [experience[i] for i in batch_indices]
        X = np.zeros((batch_size,100,6))
        Y = np.zeros((batch_size,num_actions))
        for i in range(len(batch)):
            s_t,a_t,r_t,s_tp1,game_over = batch[i]
            X[i] = s_t
            Y[i] = model.predict(np.expand_dims(s_t, axis=0))[0]
            Q_sa = np.max(model1.predict(np.expand_dims(s_tp1, axis=0))[0])
            if game_over:
                Y[i,a_t] = r_t
            else:
                Y[i,a_t] = r_t + gamma * Q_sa
        return X,Y

    def copyModelWeights(self,modelSource,model1Target):
        weights = modelSource.get_weights();
        model1Target.set_weights(weights);






    







    

    def mainLoop(self):
        e = 0;
        while True:
            e = e+1 ;
            #print(self.env.get_action_sample());
            '''
            if(not self._started and im_r != 0):
                self._started = True;
                print("state : ",s_t," reward ",r_t," game over ",g_m," im reward ",im_r);
            #print(self.env.get_action_sample());
            '''
            if((e % ITERATE_COPY_Q) == 0):
                self.copyModelWeights(self.model,self.model1);
            self.env.step(0);

            loss=0.0
            #get first state
            a_0 = 0 # (0= left , 1 = stay , 2 = right)
            s_t , r_0 , game_over,_ = self.env.step(a_0)
            
            s_tm1 = s_t
            a_t,r_t = None,None
            while not game_over:
                s_tm1 = s_t

                #next action
                if len(self.last_ex) < NUM_EPOCHS_OBSERVE:
                    a_t = self.env.get_action_sample();
                else :
                    if np.random.rand() <= self.epsilon:
                        a_t = self.env.get_action_sample();
                    else:
                        q = self.model.predict(np.expand_dims(s_t, axis=0))[0]
                        a_t = np.argmax(q)

                #apply action , get reward
                s_t , r_t , game_over ,_= self.env.step(a_t)
                
                #if reward , increment num_wins
                if r_t > 0 and game_over :
                    self.num_wins +=1
                #store experience
                self.experience.append((s_tm1,a_t,r_t,s_t,game_over))
                if(r_t > 0 and game_over):
                    self.last_ex.append((s_tm1,a_t,r_t,s_t,game_over))

                if len(self.last_ex) > NUM_EPOCHS_OBSERVE :
                    # finished observing , now start training
                    # get next batch
                    X,Y = self.get_next_batch(self.experience,self.model,NUM_ACTIONS,GAMMA,BATCH_SIZE)
                    loss += self.model.train_on_batch(X,Y)
                    X,Y = self.get_next_batch(self.last_ex,self.model,NUM_ACTIONS,GAMMA,BATCH_SIZE)
                    loss += self.model.train_on_batch(X,Y)
                    self.model1.train_on_batch(X,Y)
            
            
            #reduce epsilon gradually
            if self.epsilon > FINAL_EPSILON :
                self.epsilon -= ((INITIAL_EPSILON - FINAL_EPSILON)/NUM_EPOCHS)
            
            print("Epoch {:04d}/{:d} | loss {:.5f} | Win count {:d}".format(e + 1,NUM_EPOCHS,loss,self.num_wins))
            if e % 100 == 0 :
                self.model.save(os.path.join(DATA_DIR,"rl-network.h5"),overwrite=True)
                self.model.save_weights(os.path.join(DATA_DIR,"rl-network_w.h5"),overwrite=True)
            

        self.fout.close()
        self.model.save(os.path.join(DATA_DIR,"rl-network.h5"),overwrite=True)
        self.model.save_weights(os.path.join(DATA_DIR,"rl-network_w.h5"),overwrite=True)

        
