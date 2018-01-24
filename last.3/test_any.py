
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


flags.DEFINE_integer('npEpoch',200,'npEpoch');

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

    print(np.shape(testData));

    
   
    

if __name__ == '__main__':
    tf.app.run()