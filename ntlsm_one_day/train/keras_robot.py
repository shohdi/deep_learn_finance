
import numpy as np
import tensorflow as tf

import os as os;





from train.join_input import JoinInput
from train.read_file import ReadFile
from train.my_flags import MyFlags
from train.normalize_input import NormalizeInput;



#1/1 - 1/50 - 1/100 - 1/500 - 1/1000


def main(_):
    myFlags = MyFlags();
    joinInput = JoinInput();
    trainFileNames = joinInput.joinInput(myFlags.INPUT_FOLDER,myFlags.trainFiles);
    testFileNames = joinInput.joinInput(myFlags.INPUT_FOLDER,myFlags.testFiles);


    readFile = ReadFile();
    trainArr = np.array( readFile.readMultiFiles(trainFileNames));
    testArr = np.array( readFile.readMultiFiles(testFileNames));
    trainArr = trainArr.reshape(-1,6);
    testArr = testArr.reshape(-1,6);
    print(trainArr.shape);
    print(testArr.shape);
    normalizeInput = NormalizeInput();
    #loop on train
    end = (myFlags.INPUT_SIZE) + (myFlags.OUTPUT_SIZE) + 0 ;
    arr=np.zeros((end,));
    for i in range(len(trainArr)- end) : 
        start = i;
        end = myFlags.INPUT_SIZE + myFlags.OUTPUT_SIZE + i ;
        oneInputOutput = trainArr[start:end];
        oneInputOutput = oneInputOutput.copy();
        oneInput = oneInputOutput[0: myFlags.INPUT_SIZE];
        npArr = np.array(oneInputOutput);
        
        npArr = npArr[:,2:];
        if(i%1000 == 0):
            print(npArr);
            print(normalizeInput.getHighLowClose(npArr))
        arr = npArr;
    
    print (arr.shape,arr);



    return ;
    
        

    
    
   
    

if __name__ == '__main__':
    tf.app.run();