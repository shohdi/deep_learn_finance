



import numpy as np
import tensorflow as tf

import os as os;


from train.my_flags import MyFlags
from train.normalize_input import NormalizeInput;


class OneInOutPrep:
    def __init__(self):
        self.myFlags = MyFlags();
        self.normalizeInput = NormalizeInput();
    
    def fixArrayWithMean(self,arr,high,low):
        arr = (arr - low)/(high - low);
        arr = arr.astype('float32');
        return arr;

    def fixInput(self,inputArr):
        noOfShots = self.myFlags.noOfShots;
        inputArr = inputArr[:,3:];
        close,high,low = self.normalizeInput.getHighLowClose(inputArr);
        if(high == 0):
            high = 0.0001;
        inputArr = self.fixArrayWithMean(inputArr,high,low);
        inputArr = np.array(inputArr,dtype='float32');
        oneShot = self.myFlags.INPUT_SIZE - noOfShots ;
        ret = np.zeros((noOfShots+1,oneShot,3),dtype='float32');
        for i in range(noOfShots+1):
            ret[i] = inputArr[i:(i+oneShot)]
        return ret , high,low;




    def fixOneInputOutput(self,mainArr,index):
        oneArr = self.getInOut(mainArr,index);
        inputArr = oneArr[:self.myFlags.INPUT_SIZE];
        outputArr = oneArr[self.myFlags.INPUT_SIZE:self.myFlags.INPUT_SIZE + self.myFlags.OUTPUT_SIZE];
        inputArr,high,low = self.fixInput(inputArr);
        outputArr = outputArr[:,3:];
        
        
        
        outputArr =  self.fixArrayWithMean(outputArr,high,low);

        resultClose = outputArr[len(outputArr)-1][0];
        result = np.zeros((1,),dtype='float32');
        oneShot = self.myFlags.INPUT_SIZE - self.myFlags.noOfShots;
        close = inputArr[self.myFlags.noOfShots,oneShot-1,0];
        if(resultClose > close):
            result[0] = 1.0;
        else:
            result[0] = 0.0;

        return inputArr,result[0];
    


    def getEnd(self,index):
        return (self.myFlags.INPUT_SIZE * self.myFlags.candleSize) + (self.myFlags.OUTPUT_SIZE * self.myFlags.candleSize) + index;


    def getInOut(self,mainArr,index):
        
        start = index;
        end = self.getEnd(index) ;
        ret = mainArr[start:end];
        ret = ret.copy();
        ret = ret.reshape((-1,self.myFlags.candleSize));

        return ret;
