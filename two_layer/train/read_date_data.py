
from train.read_file import ReadFile
import numpy as np
from train.candle_reader import CandleReader


class ReadDateFile:
    
    def __init__(self):
        self.current = 0

    def next_batch(self,num, data, labels):
        
        idx = np.arange(0 , len(data))
        np.random.shuffle(idx)
        idx = idx[:num]
        data_shuffle = [data[ i] for i in idx]
        labels_shuffle = [labels[ i] for i in idx]

        return np.asarray(data_shuffle), np.asarray(labels_shuffle)


    def next_norm_batch(self,num,data,labels):
        if(self.current == None):
            self.current = 0
        if((self.current + num) >= len(data)):
            self.current = 0
        
        idx = range(len(data))
        idx = idx[self.current:(self.current+num)]
        self.current = self.current + num
        data_shuffle = [data[i] for i in idx]
        labels_shuffle = [labels[i] for i in idx]
        return np.asarray(data_shuffle), np.asarray(labels_shuffle)
    
    
    def checkArraySameDay(self,inputInnerArr,outputInnerArr):
        day = inputInnerArr[0]
        
        inputInnerArrNp = np.array(inputInnerArr)
        outputInnerArrNp = np.array(outputInnerArr)
        inputInnerArrNp = np.reshape(inputInnerArrNp,(-1,6))
        outputInnerArrNp  = np.reshape(outputInnerArrNp,(-1,6))
        for i in range(len(inputInnerArrNp)):
            if(inputInnerArrNp[i][0] != day):
                #print("day : ",day , "new day : ",inputInnerArrNp[i][0])
                return False


        for i in range(len(outputInnerArrNp)):
            newDay = outputInnerArrNp[i][0]
            if(newDay != day):
                #print("day : ",day,"new day : ",newDay)
                return False
               
        


        return True

    def checkCandleDir(self,candle):
        openC = candle[0]
        closeC = candle[1]
        if(closeC < openC):
            return -1.0
        else :
            if(closeC > openC):
                return 1.0
            else :
                return 0.0
        return 0.0

    def check5CandleDir(self,outputInnerNpArr):
        openC = outputInnerNpArr[0][0]
        closeC = outputInnerNpArr[len(outputInnerNpArr)-1][1]
        if(closeC > openC):
            return 1.0
        else:
            if(closeC < openC):
                return -1.0
            else:
                return 0.0
        return 0.0

    def check3CandleDir(self,outputInnerNpArr):
        openC = outputInnerNpArr[0][0]
        closeC = outputInnerNpArr[2][1]
        if(closeC > openC):
            return 1.0
        else:
            if(closeC < openC):
                return -1.0
            else:
                return 0.0
        return 0.0

    def checkCandleSegnal(self,outputInnerNpArr):
        firstCandle = self.checkCandleDir(outputInnerNpArr[0])
        secondCandle = self.checkCandleDir(outputInnerNpArr[1])
        thirdCandle = self.checkCandleDir(outputInnerNpArr[2])

        if(firstCandle == secondCandle == thirdCandle == -1.0):
            return -1.0
        else :
            if(firstCandle == secondCandle == thirdCandle == 1.0):
                return 1.0
            else :
                return 0.0
        
        return 0.0


    def readSysFile(self,fileName):
        readMyFile = ReadFile(fileName)
        arr = readMyFile.readMultiFiles()
        #print("array : ",arr)
        npArr = np.array(arr)
        
        return npArr

    def doNormalizeToInputAndOutput(self,inputInnerArr,outputInnerArr):
        inputInnerNpArr = np.array(inputInnerArr)
        outputInnerNpArr = np.array(outputInnerArr)
        inputInnerNpArr = np.reshape(inputInnerNpArr,(-1,6))
        outputInnerNpArr = np.reshape(outputInnerNpArr,(-1,6))
        inputInnerNpArr = [x[2:] for x in inputInnerNpArr ]
        outputInnerNpArr = [x[2:] for x in outputInnerNpArr]
        outputResult = list()
        #outputResult.append(self.check5CandleDir(outputInnerNpArr))
        #isSignal = self.checkCandleSegnal(outputInnerNpArr)
        isSignal = self.check5CandleDir(outputInnerNpArr)
        #print("output arr : ",outputInnerNpArr)
        #print("open : ",candleOpen)
        #print("close : ",candleClose)
        outputResult = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        upCount = 0
        downCount = 0
        for i in range(len(outputInnerNpArr)):
            if(outputInnerNpArr[i][1] > outputInnerNpArr[i][0]):
                upCount  = upCount +1
            else:
                if(outputInnerNpArr[i][1] < outputInnerNpArr[i][0]):
                    downCount = downCount+1
        
        if(isSignal == 1.0):
            #up
            
            outputResult[4+upCount] = 1.0
        else :
            if(isSignal == -1.0):
                outputResult[5-downCount] = 1.0
           
        outputResult = [0.0,1.0,0.0]
        openC = outputInnerNpArr[0][0]
        closeC = outputInnerNpArr[len(outputInnerNpArr)-1][1]
        if(closeC > openC):
            outputResult = [1.0,0.0,0.0]
        else :
            if(closeC < openC):
                outputResult = [0.0,0.0,1.0]
        
        
        #print("output result : ",outputResult)
        
        inputInnerNpArr = np.reshape(inputInnerNpArr,(-1))
        return inputInnerNpArr,outputResult,outputInnerNpArr

    def readArr(self,npArr):
        inputArr = list()
        outputArr = list()
        print("len of all data ",len(npArr))
        i=-6
        while (i< len(npArr)):
            i = i+6
            inputLen = 60
            outputLen = 30
            if((i + inputLen + outputLen) <= len(npArr) ):
                
                inputInnerArr = np.copy(npArr[i:(i+inputLen)])
                outputInnerArr = np.copy(npArr[(i+inputLen):(i+inputLen+outputLen)])
                isOkInput = self.checkArraySameDay(inputInnerArr,outputInnerArr)
                #print("is ok ",isOkInput)
                if(isOkInput):
                    
                    
                    inputInnerNpArr,outputResult,outputInnerNpArr = self.doNormalizeToInputAndOutput(inputInnerArr,outputInnerArr)
                    fixImageObj = CandleReader()
                    minOne,maxOne,retBool = fixImageObj.fixOneImage(inputInnerNpArr)
                    #lastClose = inputInnerNpArr[len(inputInnerNpArr)-3]
                    

                    
                    #inputInnerNpArr = np.exp(inputInnerNpArr)  /np.max(np.exp([lastClose]))
                    #inputInnerNpArr = self.softmaxNorm(inputInnerNpArr)
                    if(retBool):
                        inputArr.append(inputInnerNpArr)
                    openC = outputInnerNpArr[0][0]
                    closeC = outputInnerNpArr[len(outputInnerNpArr)-1][1]
                    #openC = (openC - minOne)/(maxOne-minOne)
                    #closeC = (closeC-minOne)/(maxOne-minOne)
                    if(retBool):
                        outputArr.append(outputResult)
                    #outputArr.append([closeC,openC])
                
                    
        self.myDataImages = inputArr
        self.myDataLabels = outputArr
        print ("len input ",len(self.myDataImages))
        print("len output ",len(self.myDataLabels))
        #print("output ",outputArr)
        #print("input ",inputArr)

    def readFile(self,fileName):
        npArr = self.readSysFile(fileName)
        self.readArr(npArr)
        

    def softmaxNorm(self,arr):
        return np.exp(arr)/np.sum(np.exp(arr),axis=0)

#test = ReadDateFile()
#test.readFile("/home/shohdi/Documents/data_with_date/myOldData.csv")

