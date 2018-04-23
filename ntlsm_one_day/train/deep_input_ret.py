from train.forex_divide_input_output import ForexDivideInputOutput
from train.input_average import InputAverage
from train.output_calc import OutputCalc
from train.draw_input import DrawInput
import numpy as np
import tensorflow as tf
import random as random
import os as os

flags = tf.app.flags;
FLAGS = flags.FLAGS;

   
from train.compress_candles import CompressCandles


class DeepInputRet :
    def __init__(self,history,future,fileNames,compress):
        self.history = history;
        self.future = future;
        self.fileNames = fileNames;
        self.compress = compress;
        self.forexDivideInputOutput = ForexDivideInputOutput(history,future,fileNames);
        
        self.inputAverage = InputAverage();
        
        self.ouputCalc = OutputCalc();
        self.drawInput = DrawInput(self.compress);


    def getSuccessFailData(self):
        inputTuble,outputTuble,mainArr = self.forexDivideInputOutput.getInputOutput();
        average = self.inputAverage.getInputAverage(self.future,inputTuble,mainArr);
        outputArr,outputFirst,outputSecond = self.ouputCalc.calcOutput(mainArr,average,outputTuble);
        #print("output " , outputFirst);
        inputImgs = self.drawInput.drawAllInputs(inputTuble,mainArr);
        
        if(FLAGS.shohdi_debug == 'False'):
            import scipy.misc as smp
            img = smp.toimage(inputImgs[0]);
            img.show();
        #print("input " , inputImgs);
        
        
        return inputImgs,outputFirst;


    def getOperationData(self):
        inputTuble,outputTuble,mainArr = self.forexDivideInputOutput.getInputOutput();
        average = self.inputAverage.getInputAverage(self.future,inputTuble,mainArr);
        outputArr,outputFirst,outputSecond = self.ouputCalc.calcOutput(mainArr,average,outputTuble);
        #print("output " , outputFirst);
        inputImgs = self.drawInput.drawAllInputs(inputTuble,mainArr);
        
        if(FLAGS.shohdi_debug == 'False'):
            import scipy.misc as smp
            img = smp.toimage(inputImgs[0]);
            img.show();
        #print("input " , inputImgs);

        input = [];
        output = [];

        for i in range(len(outputSecond)):
            if(outputSecond[i][0] >= 0):
                input.append(inputImgs[i]);
                output.append(outputSecond[i]);
        
        
        return np.array(input),np.array(output);


    
    def getInputImgs(self,inputTuble,mainArr):
        inputImgs=[];
        for i in range(len(inputTuble)):
            oneArrayIndexes = inputTuble[i];
            oneArray = mainArr[oneArrayIndexes[0]:oneArrayIndexes[1]];
            oneArray = oneArray[:,2:];
            oneArrayMax = np.amax(oneArray);
            oneArrayMin = np.amin(oneArray);
            oneArray = np.array(oneArray);
            oneArray = (oneArray - oneArrayMin)/(oneArrayMax-oneArrayMin);
            oneArray = oneArray.flatten();
            inputImgs.append(oneArray);
        inputImgs = np.array(inputImgs);
        return inputImgs;
        
    
    def getSplitResult(self,valSplit,inputImgs,inputTuble,outputTuble,average,outputArr,outputFirst,outputSecond):
        if(valSplit != None):
            myLen = len(inputImgs);
            valLen = int( myLen * valSplit);
            valStart = myLen - valLen;
            xTest = inputImgs[valStart:];
            yTest = outputArr[valStart:];
            inputImgs = inputImgs[0:valStart];
            outputArr = outputArr[0:valStart];
            outputFirst = outputFirst[0:valStart];
            outputSecond = outputSecond[0:valStart];
            inputTuble = inputTuble[0:valStart];
            outputTuble = outputTuble[0:valStart];
        return xTest,yTest,inputImgs,outputArr,outputFirst,outputSecond,inputTuble,outputTuble;
        
    def convertInputImagesToTwoDim (self,inputImgs):
        
        
        print(np.shape(inputImgs));
        inputImgs = np.lib.pad(inputImgs, ((0,0),(8,8)), 'constant', constant_values=(255));
        print(np.shape(inputImgs));
        inputImgs = inputImgs.reshape((-1,16,16));

        print(np.shape(inputImgs));
        return inputImgs;

    def compressMyImage(self,inputImgs):
        ret = [];
        import scipy.ndimage;

        for i in range(len(inputImgs)):
            oneItem = scipy.ndimage.zoom(inputImgs[i],0.125,order=0);
            ret.append(oneItem);
            



        return np.array(ret);

    def getAllResultsEqual(self,isTest,valSplit):
        inputTuble,outputTuble,mainArr = self.forexDivideInputOutput.getInputOutput();
        average = self.inputAverage.getInputAverage(self.future,inputTuble,mainArr);
        outputArr,outputFirst,outputSecond = self.ouputCalc.calcOutput(mainArr,average,outputTuble);
        #print("output " , outputFirst);
        #inputImgs = self.getInputImgs(inputTuble,mainArr);
        #inputImgs = self.convertInputImagesToTwoDim(inputImgs) * 255;

        inputImgs = self.drawInput.drawAllInputs(inputTuble,mainArr);
        #inputImgs = self.compressMyImage(inputImgs);

        xTest,yTest,inputImgs,outputArr,outputFirst,outputSecond,inputTuble,outputTuble = self.getSplitResult(valSplit,inputImgs,inputTuble,outputTuble,average,outputArr,outputFirst,outputSecond);

        
        if(FLAGS.shohdi_debug == 'False'):
            import scipy.misc as smp
            img = smp.toimage(inputImgs[0]);
            smp.imsave(os.path.join(FLAGS.outputDir,'img1.png') ,img);
            img = smp.toimage(inputImgs[1]);
            smp.imsave(os.path.join(FLAGS.outputDir,'img2.png') ,img);
            img = smp.toimage(inputImgs[2]);
            smp.imsave(os.path.join(FLAGS.outputDir,'img3.png') ,img);
        
        #print("input " , inputImgs);
        if(isTest):
            if(valSplit != None):
                return inputImgs,np.array(outputArr),xTest,yTest;
            else:
                return inputImgs,np.array(outputArr);
        upArr = [];
        downArr = [];
        noMove = [];

        for i in range(len(outputSecond)):
            val = outputSecond[i][0];
            image = inputImgs[i];
            if(val < 0):
                noMove.append(image);
            else :
                if(val == 0):
                    downArr.append(image);
                else:
                    upArr.append(image);
        

        countsArr = np.array( [len(upArr),len(downArr),len(noMove)]);
        amin = np.amin(countsArr);
        ret = [];
        retOut = [];
        for i in range(amin):
            
            threeArr = [];
            threeArr.append((upArr[i],[1.0,0.0,0.0]));
            threeArr.append((downArr[i],[0.0,0.0,1.0]));
            threeArr.append((noMove[i],[0.0,1.0,0.0])); 

            index = random.randint(0,2);

            ret.append(threeArr[index][0]);
            retOut.append(threeArr[index][1]);

            del threeArr[index];



            index = random.randint(0,1);

            ret.append(threeArr[index][0]);
            retOut.append(threeArr[index][1]);

            del threeArr[index];

            ret.append(threeArr[0][0]);
            retOut.append(threeArr[0][1]);



        if(valSplit != None):
            return np.array(ret),np.array(retOut),xTest,yTest;
        else:
            return np.array(ret),np.array(retOut);


        

        

        

        


