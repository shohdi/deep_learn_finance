import numpy as np

class OutputCalc:


    def calcOutput(self,arr,average,outputTuble):
        outputRet = [];
        outputFirst = [];
        outputSecond = [];
        for i in range(len(outputTuble)):
            loc = outputTuble[i];
            outputArr = arr[loc[0]:loc[1]];
            startVal = outputArr[0][2];
            oneAverage = average[i][0];
            halfAverage = oneAverage/2.0;
            upVal = startVal + oneAverage;
            downVal = startVal - oneAverage;
            
            upFail = startVal - halfAverage;
            downFail = startVal + halfAverage;

            upIndex = self.calcIndexOfVal(outputArr,upVal,True);
            upFailIndex = self.calcIndexOfVal(outputArr,upFail,False);
            downIndex = self.calcIndexOfVal(outputArr,downVal,False);
            downFailIndex = self.calcIndexOfVal(outputArr,downFail,True);

            foundSignal = False;
            if(self.checkIndexes(upIndex,upFailIndex)):
                outputRet.append([1,0,0]);
                outputFirst.append([1.0]);
                outputSecond.append([1.0]);
                foundSignal = True;
            
            
            if(self.checkIndexes(downIndex,downFailIndex)):
                outputRet.append([0,0,1]);
                outputFirst.append([1.0]);
                outputSecond.append([0.0]);
                foundSignal = True;

            
            if(not foundSignal):
                outputRet.append([0,1,0]);
                outputFirst.append([0.0]);
                outputSecond.append([-1.0]);
        

        return np.array(outputRet),np.array(outputFirst),np.array(outputSecond);

            




        


    def checkIndexes(self,index,failIndex):
        if(index < 0):
            return False;

        if(failIndex < 0):
            return True;

        if(index < failIndex):
            return True;
        else :
            return False;

    def calcIndexOfVal(self,arr,val,isUp):
        for i in range(len(arr)):
            if(isUp):
                if(val <= arr[i][4]):
                    return i;
            else :
                if(val >= arr[i][5]):
                    return i;
            
        return -1;


    






