
import numpy as np
import tensorflow as tf

import os as os;

from train.deep_input_ret import DeepInputRet

from train.keras_helper import KerasHelper

from train.join_input import JoinInput
from train.read_file import ReadFile
from keras import backend as K
from keras.models import Sequential
from keras.layers.convolutional import Conv2D,Conv1D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import SGD,RMSprop,Adam
from keras.layers.recurrent import LSTM
from keras.losses import mean_squared_error
import matplotlib
matplotlib.use('Agg');
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
import os as os
import tensorflow as tf

flags = tf.app.flags;
FLAGS = flags.FLAGS;

flags.DEFINE_integer('INPUT_SIZE',30,'INPUT_SIZE');
flags.DEFINE_integer('OUTPUT_SIZE',1,'OUTPUT_SIZE');

flags.DEFINE_string('INPUT_FOLDER','input','INPUT_FOLDER');


flags.DEFINE_integer('npEpoch',20,'npEpoch');
flags.DEFINE_integer('numberOfLayers',1,'npEpoch');
flags.DEFINE_integer('hiddenCels',128,'npEpoch');


flags.DEFINE_integer('batchSize',60,'batchSize');

flags.DEFINE_float('valSplit',0.2,'valSplit');

flags.DEFINE_string('outputDir','output','outputDir');

flags.DEFINE_string('inputTrainData','','inputTrainData');
flags.DEFINE_string('trainFiles','myOldData.csv','trainFiles');
flags.DEFINE_string('testFiles','myOldData.csv','testFiles');

flags.DEFINE_bool('isOperation',True,'isOperation');


#1/1 - 1/50 - 1/100 - 1/500 - 1/1000


def main(_):
    
    joinFilesClass = JoinInput();
    trainFileNames = joinFilesClass.joinInput(FLAGS.INPUT_FOLDER,FLAGS.trainFiles);
    testFileNames = joinFilesClass.joinInput(FLAGS.INPUT_FOLDER,FLAGS.testFiles);

    print('train files ',trainFileNames);
    print('test files ',testFileNames);

    reader = ReadFile(trainFileNames);

    arr = reader.readMultiFiles();

    

    arr = np.array(arr);
    arr = np.reshape(arr,(-1,6));

    arrLen = len(arr);
    testLen = int(arrLen * FLAGS.valSplit);
    testStart = arrLen - testLen;

    trainData = arr[0:testStart];
    testData = arr[testStart:];

    trainData = trainData[:,3];
    testData = testData[:,3];

    #print(testData.shape[0]);
    length = len(trainData)  - (FLAGS.INPUT_SIZE + FLAGS.OUTPUT_SIZE);
    inputSize = FLAGS.INPUT_SIZE;
    outputSize = FLAGS.OUTPUT_SIZE;
    train = list();
    y_ = list();
    testLength = len(testData) - (FLAGS.INPUT_SIZE + FLAGS.OUTPUT_SIZE);
    test = list();
    y_test = list();
    for i in range(length):
        if(i%100 == 0):
            print(i);
        train.append(   trainData[i: (i+inputSize)]);
        y_.append(  trainData[(i+inputSize) : (i+inputSize+outputSize)]);
        myMax = np.amax(train[i]);
        myMin = np.amin(train[i]);
        myMean = myMax - myMin;
        train[i] = (train[i] - myMin)/myMean;
        y_[i] = (y_[i] - myMin)/myMean;
    
    train = np.array(train);
    y_ = np.array(y_);


    for i in range(testLength):
        if(i%100 == 0):
            print(i);
        test.append(   testData[i: (i+inputSize)]);
        y_test.append(  testData[(i+inputSize) : (i+inputSize+outputSize)]);
        myMax = np.amax(test[i]);
        myMin = np.amin(test[i]);
        myMean = myMax - myMin;
        test[i] = (test[i] - myMin)/myMean;
        y_test[i] = (y_test[i] - myMin)/myMean;
    
    test = np.array(test);
    y_test = np.array(y_test);


    print("train ",train[0],"result ",y_[0]);


    #build model
    model = Sequential()
    model.add(LSTM(FLAGS.hiddenCels,input_shape=(inputSize,1) , dropout=0.2, recurrent_dropout=0.2))
    for i in range(FLAGS.numberOfLayers-1):
        model.add(LSTM(FLAGS.hiddenCels , dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))

    model_file = os.path.join(FLAGS.outputDir,'lstm.h5');
    model.save(model_file);

    model.compile(loss="mean_squared_error", optimizer="adam",   metrics=["mean_squared_error"])
    filePath = os.path.join(FLAGS.outputDir,"my-model-{epoch:06d}.h5");
    checkpoint = ModelCheckpoint(filepath=filePath,save_best_only=True);

    #train
    train = train[:,:,np.newaxis];
    test = test[:,:,np.newaxis];

    history = model.fit(train, y_, batch_size=FLAGS.batchSize, epochs=FLAGS.npEpoch, validation_data=(test, y_test),callbacks=[checkpoint]);
    score = model.evaluate(test,y_test,verbose=1);
    y_predicted = model.predict(test);
    print("Test score:",score[0]);
    print("Test accuracy:",score[1]);
    print(history.history.keys());
    plt.plot(y_test);
    plt.plot(y_predicted);
    
    plt.title('real vs predicted');
    plt.ylabel('price');
    plt.xlabel('index');
    plt.legend(['real','predicted'],loc='best');
    #plt.show();
    plt.savefig(os.path.join(FLAGS.outputDir,'acc.png'));
    plt.gcf().clear();
    plt.plot(history.history['loss']);
    plt.plot(history.history['val_loss']);
    
    plt.title('model loss');
    plt.ylabel('loss');
    plt.xlabel('epoch');
    plt.legend(['train','test'],loc='upper left');
    #plt.show();
    plt.savefig(os.path.join(FLAGS.outputDir,'loss.png'));


    
 



    
   
    

if __name__ == '__main__':
    tf.app.run()