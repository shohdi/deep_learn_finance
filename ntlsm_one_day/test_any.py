
import numpy as np
import tensorflow as tf

import os as os;

from train.deep_input_ret import DeepInputRet

from train.keras_helper import KerasHelper

from train.join_input import JoinInput

flags = tf.app.flags;
FLAGS = flags.FLAGS;
flags.DEFINE_string('shohdi_debug','False','shohdi_debug');
flags.DEFINE_integer('INPUT_SIZE',2400,'INPUT_SIZE');
flags.DEFINE_integer('OUTPUT_SIZE',80,'OUTPUT_SIZE');
flags.DEFINE_integer('HOW_MANY_MINUTES',80,'HOW_MANY_MINUTES');
flags.DEFINE_string('INPUT_FOLDER','input','INPUT_FOLDER');


flags.DEFINE_integer('npEpoch',20,'npEpoch');

flags.DEFINE_integer('batchSize',50,'batchSize');

flags.DEFINE_float('valSplit',0.05,'valSplit');

flags.DEFINE_string('outputDir','output','outputDir');

flags.DEFINE_string('inputTrainData','','inputTrainData');
flags.DEFINE_string('trainFiles','15_year.csv','trainFiles');
flags.DEFINE_string('testFiles','last_year.csv','testFiles');

flags.DEFINE_bool('isOperation',True,'isOperation');

#1/1 - 1/50 - 1/100 - 1/500 - 1/1000


def main(_):
    
    joinFilesClass = JoinInput();
    trainFileNames = joinFilesClass.joinInput(FLAGS.INPUT_FOLDER,FLAGS.trainFiles);
    testFileNames = joinFilesClass.joinInput(FLAGS.INPUT_FOLDER,FLAGS.testFiles);

    print('train files ',trainFileNames);
    print('test files ',testFileNames);

    
    
        

    inputClass = DeepInputRet(FLAGS.INPUT_SIZE,FLAGS.OUTPUT_SIZE,trainFileNames,FLAGS.HOW_MANY_MINUTES);
    
    xTrain,yTrain,xTest,yTest = inputClass.getAllResultsEqual(False,FLAGS.valSplit);
    

    


    helper = KerasHelper();

    helper.convNetTrain(xTrain,yTrain,xTest,yTest,FLAGS.npEpoch,FLAGS.batchSize,FLAGS.valSplit,FLAGS.outputDir,FLAGS.inputTrainData);
    
   
    

if __name__ == '__main__':
    tf.app.run()