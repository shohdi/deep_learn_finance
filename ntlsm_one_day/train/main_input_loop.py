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

    def normalizeInputShots(self,files):
        
        fileNames = self.joinInput.joinInput(self.myFlags.INPUT_FOLDER,files);
               
        arr = np.array( self.readFile.readMultiFiles(fileNames));
        size = int( len(arr)//self.myFlags.candleSize);
        samplesSize = size- (self.myFlags.INPUT_SIZE + self.myFlags.OUTPUT_SIZE);
        X = np.zeros((samplesSize,self.myFlags.noOfShots+1,self.myFlags.INPUT_SIZE - self.myFlags.noOfShots,3),dtype='float32');

        xUp = np.zeros((samplesSize,self.myFlags.noOfShots+1,self.myFlags.INPUT_SIZE - self.myFlags.noOfShots,3),dtype='float32');
        xDown = np.zeros((samplesSize,self.myFlags.noOfShots+1,self.myFlags.INPUT_SIZE - self.myFlags.noOfShots,3),dtype='float32');
        upCount = 0;
        downCount = 0;
        Y = np.zeros((samplesSize,1),dtype='float32');
        for i in range(samplesSize):
            index = i * self.myFlags.candleSize;

            oneX,oneY = self.oneInOutPrep.fixOneInputOutputShots(arr,index);
            oneX = np.array(oneX,dtype='float32');
            if(oneY > 0.5):
                xUp[upCount] = oneX;
                upCount = upCount + 1 ;
            else :
                xDown[downCount] = oneX;
                downCount = downCount + 1;
        myCount = downCount;
        if(downCount > upCount):
            myCount = upCount;
        
        for i in range(myCount):
            index = i * 2;
            X[index] = xUp[i];
            Y[index] = np.array([1.0],dtype='float32');
            X[index+1] = xDown[i];
            Y[index+1] =  np.array([0.0],dtype='float32');
            
            if(i % 1000 == 0):
                print ('input number : %d , output %f' % (index,Y[index]));
            

        X = X[:(myCount * 2)];
        Y = Y[:(myCount * 2)];
        

        print ('x shape %s , y shape %s' % (X.shape,Y.shape));
        print ('first x ',X[0]);
        print ('first x shape ',X[0].shape);
        return X,Y ;

    def normalizeInput(self,files):
        
        fileNames = self.joinInput.joinInput(self.myFlags.INPUT_FOLDER,files);
               
        arr = np.array( self.readFile.readMultiFiles(fileNames));
        size = int( len(arr)//self.myFlags.candleSize);
        samplesSize = size- (self.myFlags.INPUT_SIZE + self.myFlags.OUTPUT_SIZE);
        X = np.zeros((samplesSize,self.myFlags.INPUT_SIZE ,3),dtype='float32');

        xUp = np.zeros((samplesSize,self.myFlags.INPUT_SIZE ,3),dtype='float32');
        xDown = np.zeros((samplesSize,self.myFlags.INPUT_SIZE ,3),dtype='float32');
        upCount = 0;
        downCount = 0;
        Y = np.zeros((samplesSize,1),dtype='float32');
        for i in range(samplesSize):
            index = i * self.myFlags.candleSize;

            oneX,oneY = self.oneInOutPrep.fixOneInputOutput(arr,index);
            oneX = np.array(oneX,dtype='float32');
            if(oneY > 0.5):
                xUp[upCount] = oneX;
                upCount = upCount + 1 ;
            else :
                xDown[downCount] = oneX;
                downCount = downCount + 1;
        myCount = downCount;
        if(downCount > upCount):
            myCount = upCount;
        
        for i in range(myCount):
            index = i * 2;
            X[index] = xUp[i];
            Y[index] = np.array([1.0],dtype='float32');
            X[index+1] = xDown[i];
            Y[index+1] =  np.array([0.0],dtype='float32');
            
            if(i % 1000 == 0):
                print ('input number : %d , output %f' % (index,Y[index]));
            

        X = X[:(myCount * 2)];
        Y = Y[:(myCount * 2)];
        

        print ('x shape %s , y shape %s' % (X.shape,Y.shape));
        print ('first x ',X[0]);
        print ('first x shape ',X[0].shape);
        return X,Y ;




        
        