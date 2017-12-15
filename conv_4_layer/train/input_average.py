from train.forex_divide_input_output import ForexDivideInputOutput
import numpy as np;

class InputAverage:
    def __init__(self):
        return;
        
    

    def getInputAverage (self,outputSize,inputTuble,arr):
        ret = list();
        for i in range(len(inputTuble)):
            inputArr = arr[inputTuble[i][0]:inputTuble[i][1]];
            inputArr = inputArr[:,2:];
            inputAverage = 0.0;
            count = 0.0;
            for index in range(len(inputArr)):
                if((index+outputSize) <= len(inputArr)):
                    partArr = inputArr[index:(index+outputSize)];
                    maxVal = np.amax(partArr);
                    minVal = np.amin(partArr);
                    inputAverage = inputAverage + (maxVal-minVal);
                    count = count + 1;
            
        
            inputAverage = inputAverage/count;
            inputAverage = inputAverage * 0.7;
            ret.append(inputAverage);
        return np.array(ret);

