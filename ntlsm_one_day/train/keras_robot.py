
import numpy as np
import tensorflow as tf

import os as os;





from train.join_input import JoinInput
from train.read_file import ReadFile
from train.my_flags import MyFlags



#1/1 - 1/50 - 1/100 - 1/500 - 1/1000


def main(_):
    myFlags = MyFlags();
    joinInput = JoinInput();
    trainFileNames = joinInput.joinInput(myFlags.INPUT_FOLDER,myFlags.trainFiles);
    testFileNames = joinInput.joinInput(myFlags.INPUT_FOLDER,myFlags.testFiles);


    readFile = ReadFile();
    trainArr = readFile.readMultiFiles(trainFileNames);
    testArr = readFile.readMultiFiles(testFileNames);

    
    #loop on train
    end = myFlags.INPUT_SIZE + myFlags.OUTPUT_SIZE + 0 ;
    arr=np.zeros((end,));
    for i in range(len(trainArr)- end) : 
        start = i;
        end = myFlags.INPUT_SIZE + myFlags.OUTPUT_SIZE + i ;
        oneInputOutput = trainArr[start:end];
        oneInputOutput = oneInputOutput.copy();
        oneInput = oneInputOutput[0: myFlags.INPUT_SIZE];
        npArr = np.array(oneInputOutput);
        arr = npArr;
    
    print (arr.shape,arr);



    return ;
    
        

    
    
   
    

if __name__ == '__main__':
    tf.app.run();