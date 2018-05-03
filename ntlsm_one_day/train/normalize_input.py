import numpy as np
import tensorflow as tf

import os as os;


from train.my_flags import MyFlags



class NormalizeInput:
    def __init__(self):
        self.myFlags = MyFlags();

    


    def getHighLowClose(self,inputArr):
        myMax = np.amax(inputArr);
        myMin = np.amin(inputArr);
        flatInput = inputArr.copy();
        flatInput = flatInput.flatten();
        close = flatInput[len(flatInput)-3];
        upDiff = myMax - close;
        downDiff = close - myMin;
        diff = 0.0;
        if(upDiff >= downDiff):
            diff = upDiff;
        else:
            diff = downDiff;
        
        return (close,close+diff,close-diff);

