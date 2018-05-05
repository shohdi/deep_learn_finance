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
        size = int( len(arr)/self.myFlags.candleSize);
        for i in range(size- (self.myFlags.INPUT_SIZE + self.myFlags.OUTPUT_SIZE)):
            index = i * self.myFlags.candleSize;

            oneInOut = self.oneInOutPrep.fixOneInputOutput(arr,index);
            if(i % 100):
                print(oneInOut.shape);




        
        