from train.read_file import ReadFile
import numpy as np


class ForexDivideInputOutput:
    def __init__(self,history,future,files):
        self.history = history
        self.future = future
        self.files = files
        self.readFile = ReadFile(self.files)
    


    def getInputOutput(self):
        mainArr = self.readFile.readMultiFiles()
        #print(mainArr)
        mainArr  = np.array(mainArr)
        mainArr = np.reshape(mainArr,(-1,6))
        inputTuble = []
        outputTuble = []
               

        for mainCycle in range(len(mainArr)):
            inputIn = []
            outputIn = []
            last = mainCycle + self.history + self.future;
            if(last < len(mainArr)):
                inputStartIndex = mainCycle
                inputEndIndex = (mainCycle + self.history)
                inputIn = mainArr[inputStartIndex : inputEndIndex ]
                valid = self.checkArrInSameDay(inputIn)
                if valid :
                    outputStartIndex = inputEndIndex

                    outputIn = mainArr[outputStartIndex:]
                    #print(outputIn.shape)
                    outputIn = self.getOutput(outputIn,inputIn[0][0])
                    #print(outputIn.shape)
                    outputEndIndex = outputStartIndex + len(outputIn)
                    if(outputEndIndex >= (outputStartIndex + self.future)):
                        inputTubleItem = (inputStartIndex,inputEndIndex)
                        outputTubleItem = (outputStartIndex,outputEndIndex)
                        inputArrItem = mainArr[inputTubleItem[0]:inputTubleItem[1]];
                        inputArrItem = inputArrItem[:,2:];
                        amax = np.amax(inputArrItem);
                        amin = np.amin(inputArrItem);
                        
                        if(amin > 0 and (amax - amin) > 0):
                            inputTuble.append(inputTubleItem);
                            outputTuble.append(outputTubleItem);
        return inputTuble,outputTuble,mainArr
                



    def getOutput (self,arr,day):
        ret = []

        index = 0

        while ( index < len(arr) and  arr[index:][0][0] == day):
            index +=1
        
        

        ret = arr[0:index]
            


        return ret


    def checkArrInSameDay (self,arr):
        if(len(arr) == 0):
            return False
        
        day = arr[0][0]
        for i in range(len(arr)):
            if(arr[i][0] != day):
                return False


        return True
        
                    
                
            








