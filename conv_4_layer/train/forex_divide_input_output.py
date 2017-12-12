from train.read_file import ReadFile
import numpy as np


class ForexDivideInputOutput:
    def __init__(self,history,future):
        self.history = history
        self.future = future
        self.readFile = ReadFile("/home/shohdi/projects/learning-tensorflow/projects/deep_learn_finance/conv_4_layer/input/myOldData.csv")
    


    def getInputOutput(self):
        mainArr = self.readFile.readMultiFiles()
        #print(mainArr)
        mainArr  = np.array(mainArr)
        mainArr = np.reshape(mainArr,(-1,6))
        

        for mainCycle in range(len(mainArr)):
            inputIn = []
            outputIn = []
            last = mainCycle + self.history + self.future;
            if(last < len(mainArr)):
                inputIn = mainArr[mainCycle : (mainCycle + self.history)]
                valid = self.checkArrInSameDay(inputIn)
                if valid :
                    outputIn = self.getOutput(mainArr[(mainCycle + self.history):],inputIn[0][0])
                



    def getOutput (self,arr,day):
        ret = list()
        return ret


    def checkArrInSameDay (self,arr):
        if(len(arr) == 0):
            return False
        
        day = arr[0][0]
        for i in range(len(arr)):
            if(arr[i][0] != day):
                return False


        return True
        
                    
                
            








