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
            valueToWin = 0.0;
            valueToTrade = 0.0;
            multiplyValue = 0.0;

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
            valueToWin,valueToTrade,multiplyValue = self.calculateValueToTrade(inputAverage);
            ret.append((inputAverage,multiplyValue,valueToTrade,valueToWin));
        return np.array(ret);

    def calculateValueToTrade(self,inputAverage):
        #1/1 - 1/50 - 1/100 - 1/500 - 1/1000
        xArr = [1/1000.0,1/500.0,1/100.0,1/50.0,1/1.0];
        for i in range(len(xArr)):
            tradeX = xArr[i];
            if(inputAverage <= tradeX):
                valueToTrade = tradeX/inputAverage;
                valueToWin = 1/valueToTrade;
                valueToTrade = round(valueToTrade,2);
                valueToWin = round(valueToWin,2);
                return valueToWin,valueToTrade,tradeX;

        return 0.0,0.0,0.0;

