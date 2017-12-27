import numpy as np
from keras import backend as K
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import SGD,RMSprop,Adam
import matplotlib
matplotlib.use('Agg');
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
import os as os



class KerasHelper:
    def convNetBuild(self,input_shape):
        model = Sequential();
        model.add(Dense(3,input_shape=input_shape));
        model.add(Activation('relu'));
        model.add(Activation('softmax'));

        return model;


    def convNetTrain(self,xTrain,yTrain,xTest,yTest,nbEpoch,batchSize,valSplit,outputDir,inputTrainData):
        inputShape = np.shape(xTrain[0]);
        print(inputShape);
        #inputShape = (inputShape[0],inputShape[1],1);
        print(inputShape);
        K.set_image_dim_ordering("tf");
        xTrain = xTrain.astype('float32');
        xTest = xTest.astype('float32');
        yTrain = yTrain.astype('float32');
        yTest = yTest.astype('float32');
        #xTrain /= 255.0;
        #xTest /= 255.0;
        #xTrain = xTrain[:,:,:,np.newaxis];
        #xTest = xTest[:,:,:,np.newaxis];
        #print(xTrain.shape[0],'train samples');
        #print(xTest.shape[0],'test samples');
        model = self.convNetBuild(inputShape);
        if(inputTrainData != ''):
            print('train file',inputTrainData);
            oldTrainData = os.path.join(outputDir,inputTrainData);
            print('full path train file',oldTrainData);
            model.load_weights(oldTrainData);
        model.compile(loss="categorical_crossentropy",optimizer=Adam(),metrics=["accuracy"]);
        filePath = os.path.join(outputDir,"my-model-{epoch:06d}.h5");
        checkpoint = ModelCheckpoint(filepath=filePath,save_best_only=True);
        history = model.fit(xTrain,yTrain,batch_size=batchSize,epochs=nbEpoch,verbose=1,validation_split=valSplit,callbacks=[checkpoint]);
        score = model.evaluate(xTest,yTest,verbose=1);
        print("Test score:",score[0]);
        print("Test accuracy:",score[1]);
        print(history.history.keys());
        plt.plot(history.history['acc']);
        plt.plot(history.history['val_acc']);
        plt.title('model accuracy');
        plt.ylabel('accuracy');
        plt.xlabel('epoch');
        plt.legend(['train','test'],loc='upper left');
        #plt.show();
        plt.savefig(os.path.join(outputDir,'acc.png'));
        plt.plot(history.history['loss']);
        plt.plot(history.history['val_loss']);
        plt.title('model loss');
        plt.ylabel('loss');
        plt.xlabel('epoch');
        plt.legend(['train','test'],loc='upper left');
        #plt.show();
        plt.savefig(os.path.join(outputDir,'loss.png'));


        
    def convNetOpTrain(self,xTrain,yTrain,xTest,yTest,nbEpoch,batchSize,valSplit,outputDir,inputTrainData):
            inputShape = np.shape(xTrain[0]);
            print(inputShape);
            inputShape = (inputShape[0],inputShape[1],1);
            print(inputShape);
            K.set_image_dim_ordering("tf");
            xTrain = xTrain.astype('float32');
            xTest = xTest.astype('float32');
            yTrain = yTrain.astype('float32');
            yTest = yTest.astype('float32');
            xTrain /= 255.0;
            xTest /= 255.0;
            xTrain = xTrain[:,:,:,np.newaxis];
            xTest = xTest[:,:,:,np.newaxis];
            print(xTrain.shape[0],'train samples');
            print(xTest.shape[0],'test samples');
            model = self.convNetBuild(inputShape);
            if(inputTrainData != ''):
                print('train file',inputTrainData);
                oldTrainData = os.path.join(outputDir,inputTrainData);
                print('full path train file',oldTrainData);
                model.load_weights(oldTrainData);
            model.compile(loss="binary_crossentropy",optimizer=Adam(),metrics=["accuracy"]);
            filePath = os.path.join(outputDir,"operation-model-{epoch:06d}.h5");
            checkpoint = ModelCheckpoint(filepath=filePath,save_best_only=True);
            history = model.fit(xTrain,yTrain,batch_size=batchSize,epochs=nbEpoch,verbose=1,validation_split=valSplit,callbacks=[checkpoint]);
            score = model.evaluate(xTest,yTest,verbose=1);
            print("Test score:",score[0]);
            print("Test accuracy:",score[1]);
            print(history.history.keys());
            plt.plot(history.history['acc']);
            plt.plot(history.history['val_acc']);
            plt.title('model accuracy');
            plt.ylabel('accuracy');
            plt.xlabel('epoch');
            plt.legend(['train','test'],loc='upper left');
            plt.show();
            plt.plot(history.history['loss']);
            plt.plot(history.history['val_loss']);
            plt.title('model loss');
            plt.ylabel('loss');
            plt.xlabel('epoch');
            plt.legend(['train','test'],loc='upper left');
            plt.show();

