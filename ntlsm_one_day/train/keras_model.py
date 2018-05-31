import numpy as np
import tensorflow as tf

import os as os;


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
import matplotlib
matplotlib.use('Agg');
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint

from train.my_flags import MyFlags

class KerasModel:
    def __init__(self):
        self.myFlags = MyFlags();
        self.INPUT_SIZE = self.myFlags.INPUT_SIZE // self.myFlags.HOW_MANY_MINUTES ;
    

    def buildModel(self):
        shape = (self.INPUT_SIZE,3);
        if(self.myFlags.noOfShots > 1):
            shape = (self.myFlags.noOfShots+1,(self.myFlags.INPUT_SIZE - self.myFlags.noOfShots)*3);
        
        print ('print model input shape : ',shape);
        model = Sequential()
        model.add(CuDNNLSTM(self.myFlags.hiddenUnits,input_shape=shape  ))#, dropout=0.2, recurrent_dropout=0.2))
        for i in range(self.myFlags.hiddenLayers-1):
            model.add(RepeatVector(1));
            model.add(CuDNNLSTM(self.myFlags.hiddenUnits ))#, dropout=0.2, recurrent_dropout=0.2))
        
        model.add(Dropout(0.2));
        model.add(Dense(1))

        model.add(Activation("sigmoid"));
        model.summary();
        return model;
    



    def trainModel(self,model,x,y,xVal,yVal,xTest,yTest):
        
        #model.compile(loss="mean_squared_error", optimizer="RMSprop",   metrics=["mae"]);
        model.compile(loss="binary_crossentropy", optimizer="RMSprop",   metrics=["acc"]);
        filePath = os.path.join(self.myFlags.outputDir,"my-model-{epoch:06d}.h5");
        checkpoint = ModelCheckpoint(filepath=filePath,save_best_only=True);
        history = model.fit(x, y, batch_size=self.myFlags.batchSize, epochs=self.myFlags.npEpoch, validation_data=(xVal, yVal),callbacks=[checkpoint]);
        score = model.evaluate(xTest,yTest,verbose=1);
        print("Test score:",score[0]);
        print("Test accuracy:",score[1]);
        print(history.history.keys());
       

        #plt.gcf().clear();
        plt.plot(history.history['loss']);
        plt.plot(history.history['val_loss']);
    
        plt.title('model loss');
        plt.ylabel('loss');
        plt.xlabel('epoch');
        plt.legend(['train','test'],loc='upper left');
        #plt.show();
        plt.savefig(os.path.join(self.myFlags.outputDir,'loss.png'));



