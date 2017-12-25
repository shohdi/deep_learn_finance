from train.forex_divide_input_output import ForexDivideInputOutput
from train.input_average import InputAverage
from train.output_calc import OutputCalc
from train.draw_input import DrawInput
import numpy as np
import tensorflow as tf
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

        

        


