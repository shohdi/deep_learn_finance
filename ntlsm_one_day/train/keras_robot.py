
import numpy as np
import tensorflow as tf

import os as os;





from train.join_input import JoinInput
from train.read_file import ReadFile

flags = tf.app.flags;
FLAGS = flags.FLAGS;
flags.DEFINE_string('shohdi_debug','False','shohdi_debug');
flags.DEFINE_integer('INPUT_SIZE',2880,'INPUT_SIZE');
flags.DEFINE_integer('OUTPUT_SIZE',96,'OUTPUT_SIZE');
flags.DEFINE_integer('HOW_MANY_MINUTES',1,'HOW_MANY_MINUTES');
flags.DEFINE_string('INPUT_FOLDER','input','INPUT_FOLDER');


flags.DEFINE_integer('npEpoch',5,'npEpoch');

flags.DEFINE_integer('batchSize',50,'batchSize');

flags.DEFINE_float('valSplit',0.05,'valSplit');

flags.DEFINE_string('outputDir','output','outputDir');

flags.DEFINE_string('inputTrainData','','inputTrainData');
#flags.DEFINE_string('trainFiles','15_year.csv','trainFiles');
flags.DEFINE_string('trainFiles','last_year.csv','trainFiles');

flags.DEFINE_string('testFiles','last_year.csv','testFiles');

flags.DEFINE_bool('isOperation',True,'isOperation');

#1/1 - 1/50 - 1/100 - 1/500 - 1/1000


def main(_):
    
    joinInput = JoinInput();
    trainFileNames = joinInput.joinInput(FLAGS.INPUT_FOLDER,FLAGS.trainFiles);
    testFileNames = joinInput.joinInput(FLAGS.INPUT_FOLDER,FLAGS.testFiles);


    readFile = ReadFile();
    trainArr = readFile.readMultiFiles(trainFileNames);
    testArr = readFile.readMultiFiles(testFileNames);

    
    #loop on train
    end = FLAGS.INPUT_SIZE + FLAGS.OUTPUT_SIZE + 0 ;
    arr=np.zeros((end,));
    for i in range(len(trainArr)- end) : 
        start = i;
        end = FLAGS.INPUT_SIZE + FLAGS.OUTPUT_SIZE + i ;
        oneInputOutput = trainArr[start:end];
        oneInputOutput = oneInputOutput.copy();
        oneInput = oneInputOutput[0: FLAGS.INPUT_SIZE];
        npArr = np.array(oneInputOutput);
        arr = npArr;
    
    print (arr.shape,arr);



    return ;
    
        

    
    
   
    

if __name__ == '__main__':
    tf.app.run();