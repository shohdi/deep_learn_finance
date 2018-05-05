



import numpy as np
import tensorflow as tf

import os as os;


from train.my_flags import MyFlags
from train.normalize_input import NormalizeInput;


class OneInOutPrep:
    def __init__(self):
        self.myFlags = MyFlags();
        self.normalizeInput = NormalizeInput();
    

    def fixOneInputOutput(self,mainArr,index):
        oneArr = self.getInOut(mainArr,index);
        return oneArr;
    


    def getEnd(self,index):
        return (self.myFlags.INPUT_SIZE * self.myFlags.candleSize) + (self.myFlags.OUTPUT_SIZE * self.myFlags.candleSize) + index;


    def getInOut(self,mainArr,index):
        
        start = index;
        end = self.getEnd(index) ;
        ret = mainArr[start:end];
        ret = ret.copy();
        ret = ret.reshape((-1,self.myFlags.candleSize));

        return ret;
