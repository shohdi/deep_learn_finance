import numpy as np
import tensorflow as tf

import os as os;





from train.join_input import JoinInput
from train.read_file import ReadFile

flags = tf.app.flags;
FLAGS = flags.FLAGS;
flags.DEFINE_string('shohdi_debug','False','shohdi_debug');
flags.DEFINE_integer('INPUT_SIZE',1536 ,'INPUT_SIZE');
flags.DEFINE_integer('OUTPUT_SIZE',96,'OUTPUT_SIZE');
flags.DEFINE_integer('HOW_MANY_MINUTES',4,'HOW_MANY_MINUTES');
flags.DEFINE_string('INPUT_FOLDER','input','INPUT_FOLDER');


flags.DEFINE_integer('npEpoch',5,'npEpoch');

flags.DEFINE_integer('batchSize',128,'batchSize');

flags.DEFINE_float('valSplit',0.1,'valSplit');
flags.DEFINE_float('testSplit',0.1,'testSplit');

flags.DEFINE_string('outputDir','output','outputDir');

flags.DEFINE_string('inputTrainData','','inputTrainData');
flags.DEFINE_string('trainFiles','15_year.csv','trainFiles');
#flags.DEFINE_string('trainFiles','last_year.csv','trainFiles');

flags.DEFINE_string('testFiles','last_year.csv','testFiles');

#flags.DEFINE_string('trainFiles','trainYear1.csv;trainYear2.csv;trainYear3.csv','trainFiles');
#flags.DEFINE_string('testFiles','testYear.csv','testFiles');

flags.DEFINE_bool('isOperation',True,'isOperation');
flags.DEFINE_integer('candleSize',6,'candleSize');
flags.DEFINE_integer('hiddenUnits',128,'hiddenUnits');
flags.DEFINE_integer('hiddenLayers',1,'hiddenLayers');
flags.DEFINE_integer('noOfShots',0,'noOfShots');


class MyFlags:
    def __init__(self):
        self.shohdi_debug = FLAGS.shohdi_debug;
        self.INPUT_SIZE = FLAGS.INPUT_SIZE;
        self.OUTPUT_SIZE = FLAGS.OUTPUT_SIZE;
        self.HOW_MANY_MINUTES = FLAGS.HOW_MANY_MINUTES;
        self.INPUT_FOLDER = FLAGS.INPUT_FOLDER;
        self.npEpoch = FLAGS.npEpoch;
        self.batchSize = FLAGS.batchSize;
        self.valSplit = FLAGS.valSplit;
        self.outputDir = FLAGS.outputDir;
        self.inputTrainData = FLAGS.inputTrainData;
        self.trainFiles = FLAGS.trainFiles;
        self.testFiles = FLAGS.testFiles;
        self.isOperation = FLAGS.isOperation;
        self.candleSize = FLAGS.candleSize;
        self.hiddenUnits = FLAGS.hiddenUnits;
        self.hiddenLayers = FLAGS.hiddenLayers;
        self.testSplit = FLAGS.testSplit;
        self.noOfShots = FLAGS.noOfShots;
        