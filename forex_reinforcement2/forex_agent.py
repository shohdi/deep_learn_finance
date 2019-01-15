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

IS_TEST_RUN = False
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
        self.shape = (140,)
        
        
        expExists = os.path.isfile(os.path.join(DATA_DIR,"rl-network_exp.h5"));
        self.experience = None;
        if(expExists):
            print('found experience file');
            myFin = open(os.path.join(DATA_DIR,"rl-network_exp.h5"),'rb');
            self.experience = pickle.load(myFin);
            myFin.close();
            print('end loading experience file');
            print('exp length ',len(self.experience));
        else:
            self.experience  = collections.deque(maxlen=MEMORY_SIZE)

        self.last_ex = collections.deque(maxlen=(NUM_EPOCHS_OBSERVE*3));
        self.fout = open(os.path.join(DATA_DIR,"rl-network-results.tsv"),"wb")
        self.num_games,self.num_wins = 0,0
        if(not IS_TEST_RUN):
            self.epsilon = INITIAL_EPSILON
        else:
            self.epsilon = 0.0;


    def buildModel(self):
        
        print ('print model input shape : ',self.shape);

        model = Sequential()
        model.add(Dense(256,input_shape=self.shape));
        model.add(Activation('relu'));
        model.add(Dense(512));
        model.add(Activation('relu'));
        model.add(Dense(256));
        model.add(Activation('relu'));
        
        model.add(Dense(NUM_ACTIONS,kernel_initializer="normal"))
        
        
        #model = Sequential()
        #model.add(CuDNNLSTM(600,input_shape=shape ,return_sequences=True ))#, dropout=0.2, recurrent_dropout=0.2))
        #model.add(CuDNNLSTM(600 ))
        #model.add(CuDNNLSTM(600 ,return_sequences=True ))
        #model.add(CuDNNLSTM(600 ))
       
        
        #model.add(Dense(NUM_ACTIONS,kernel_initializer="normal"))
        model.compile(optimizer=Adam(lr=1e-6),loss="mse")
        model.summary();
        modelExists = os.path.isfile(os.path.join(DATA_DIR,"rl-network_w.h5"));
        if modelExists:
            print('found model file , loading ...');
            model.load_weights(os.path.join(DATA_DIR,"rl-network_w.h5"))
            print('end found model file , loading ...');
        return model;
        

    def createModels(self):
        self.model = self.buildModel();
        self.model1 = self.buildModel();
        
        self.copyModelWeights(self.model,self.model1);
        test = np.zeros(self.shape);
        
        
        out = self.model.predict(np.expand_dims(test, axis=0));
        print("out of model like : ",out)
        out = self.model.predict(np.expand_dims(test, axis=0));
        print("out of model like : ",out)
        out1 = self.model1.predict(np.expand_dims(test, axis=0));
        print("out of model1 like :  " , out1)

    def get_next_batch(self,experience,num_actions,gamma,batch_size):
        
        batch_indices = np.random.randint(low=0,high=len(experience),size=batch_size)
        batch = [experience[i] for i in batch_indices]
        X = np.zeros((batch_size,self.shape[0],))
        Y = np.zeros((batch_size,num_actions))
        for i in range(len(batch)):
            s_t,a_t,r_t,s_tp1,game_over = batch[i]
            X[i] = s_t
            Y[i] = self.model.predict(np.expand_dims(s_t, axis=0))[0]
            q_next = self.model.predict(np.expand_dims(s_tp1, axis=0))[0];
            a_next = np.argmax(q_next);

            
            Q_sa = self.model1.predict(np.expand_dims(s_tp1, axis=0))[0,a_next];
            if game_over:
                Y[i,a_t] = r_t
            else:
                Y[i,a_t] = r_t + gamma * Q_sa
        return X,Y

    def copyModelWeights(self,modelSource,model1Target):
        weights = modelSource.get_weights();
        model1Target.set_weights(weights);
        
    
  
        







    







    

    def mainLoop(self):
        self.createModels();
        e = 0;
        state_count = 0;
        
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
            #no need for reset
            #self.env.step(0);

            loss=0.0
            #get first state
            a_0 = 0 # (0= left , 1 = stay , 2 = right)
            s_t , r_0 , game_over,_ = self.env.step(a_0)
            
            s_tm1 = s_t
            a_t,r_t = None,None
            
            while not game_over:
                s_tm1 = s_t

                #next action
                randValue = np.random.rand()
                if  randValue <= self.epsilon:
                    a_t = self.env.get_action_sample();
                    #print("random action ",a_t)
                else:
                    q = self.model.predict(np.expand_dims(s_t, axis=0))[0]
                    a_t = np.argmax(q)
                    #print("predicted action ",a_t)

                #apply action , get reward
                s_t , r_t , game_over ,_= self.env.step(a_t)
                state_count+=1
                #if reward , increment num_wins
                if r_t > 0 and game_over :
                    self.num_wins +=1
                #store experience
                self.experience.append((s_tm1,a_t,r_t,s_t,game_over))
                if(r_t > 0 and game_over ):
                    self.last_ex.append((s_tm1,a_t,r_t,s_t,game_over))
                


                if (not IS_TEST_RUN) and   (len(self.last_ex) >= NUM_EPOCHS_OBSERVE or len(self.experience) >= (NUM_EPOCHS_OBSERVE * 10 * 30) ):
                    #print("entering training")
                    # finished observing , now start training
                    # get next batch
                    if (state_count % (BATCH_SIZE // 2)) == 0:
                        X,Y = self.get_next_batch(self.experience,NUM_ACTIONS,GAMMA,BATCH_SIZE)
                        #print("getting batch , y ",Y)
                        loss += self.model.train_on_batch(X,Y)
                    #print("trained first model loss ",loss)
                    #X,Y = self.get_next_batch(self.last_ex,NUM_ACTIONS,GAMMA,BATCH_SIZE)
                    #print("get win batch y ",Y)
                    #loss += self.model.train_on_batch(X,Y)
                    #print("trained first model on win ",loss)
                    #self.model1.train_on_batch(X,Y)
                    #print("trained second model on win ",loss)
                    #print("finished training ok");
            
            
            #reduce epsilon gradually
                
                if self.epsilon > FINAL_EPSILON and len(self.last_ex) >= (NUM_EPOCHS_OBSERVE * 2)  :
                    self.epsilon -= ((INITIAL_EPSILON - FINAL_EPSILON)/NUM_EPOCHS)
            
            print("Epoch {:04d}/{:d} | loss {:.5f} | Win count {:d} | current epsilon {:.5f} | last reward {:.5f}".format(e + 1,NUM_EPOCHS,loss,self.num_wins,self.epsilon,r_t))
            if  (not IS_TEST_RUN) and ((e % 100 == 0) and e > 0) :
                print("saving ... ",e)
                self.model.save(os.path.join(DATA_DIR,"rl-network.h5"),overwrite=True)
                self.model.save_weights(os.path.join(DATA_DIR,"rl-network_w.h5"),overwrite=True)
                print("saving experience !!!")
                myFout = open(os.path.join(DATA_DIR,"rl-network_exp.h5"),'wb');
                pickle.dump(self.experience,myFout);
                myFout.close();
                print("finished saving experience")
            

        self.fout.close()
        if(not IS_TEST_RUN):
            self.model.save(os.path.join(DATA_DIR,"rl-network.h5"),overwrite=True)
            self.model.save_weights(os.path.join(DATA_DIR,"rl-network_w.h5"),overwrite=True)

        
