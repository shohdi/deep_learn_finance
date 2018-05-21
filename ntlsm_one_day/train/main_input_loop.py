import numpy as np
import tensorflow as tf

import os as os;


from train.my_flags import MyFlags
from train.normalize_input import NormalizeInput;

from train.one_in_out_prep import OneInOutPrep;
from train.join_input import JoinInput
from train.read_file import ReadFile


class MainInputLoop:
    def __init__(self):
        self.myFlags = MyFlags();
        self.oneInOutPrep = OneInOutPrep();
        self.joinInput = JoinInput();
        self.readFile = ReadFile();

    def normalizeInput(self,files):
        
        fileNames = self.joinInput.joinInput(self.myFlags.INPUT_FOLDER,files);
               
        arr = np.array( self.readFile.readMultiFiles(fileNames));
        size = int( len(arr)//self.myFlags.candleSize);
        samplesSize = size- (self.myFlags.INPUT_SIZE + self.myFlags.OUTPUT_SIZE);
        X = np.zeros((samplesSize,self.myFlags.INPUT_SIZE,3),dtype='float32');
        Y = np.zeros((samplesSize,1),dtype='float32');
        for i in range(samplesSize):
            index = i * self.myFlags.candleSize;

            oneX,oneY = self.oneInOutPrep.fixOneInputOutput(arr,index);
            oneX = np.array(oneX,dtype='float32');
            X[i] = oneX;
            oneY = np.array([oneY],dtype='float32');
            Y[i] = oneY;
            if(i % 1000 == 0):
                print ('input number : %d , output %f' % (i,oneY));
            

        

        print ('x shape %s , y shape %s' % (X.shape,Y.shape));
        return X,Y ;




        
        