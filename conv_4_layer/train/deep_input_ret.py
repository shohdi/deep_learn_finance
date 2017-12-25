from train.forex_divide_input_output import ForexDivideInputOutput
from train.input_average import InputAverage
from train.output_calc import OutputCalc
from train.draw_input import DrawInput
import numpy as np
import tensorflow as tf
import random as random
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



    def getAllResultsEqual(self):
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



        
        return np.array(ret),np.array(retOut);


        

        

        

        


