from train.read_file import ReadFile
import numpy as np




class CandleReader :
    def __init__ (self):
        self.trainFiles = list()
        self.trainResultFiles = list()
        self.testFiles = list()
        self.testResultFiles = list()
        self.trainData = list()
        self.trainResult = list()
        self.testData = list()
        self.testResult = list()
    

    def addTrainFile(self,fileName):
        self.trainFiles.append(fileName)
    
    def addTestFile(self,fileName):
        self.testFiles.append(fileName)
    

    def addTrainResultFile(self,fileName):
        self.trainResultFiles.append(fileName)

    def addTestResultFile(self,fileName):
        self.testResultFiles.append(fileName)
    


    def readAllFiles(self):
        self.trainData = self.readOneArrayFromFiles(self.trainFiles,240,self.trainData)
        self.fixMeanForAllImages(self.trainData)
        self.testData = self.readOneArrayFromFiles(self.testFiles,240,self.testData)
        self.fixMeanForAllImages(self.testData)
        self.trainResult = self.readOneArrayFromFiles(self.trainResultFiles,20,self.trainResult)
        self.testResult = self.readOneArrayFromFiles(self.testResultFiles,20,self.testResult)
        self.trainResult = self.normalizeResult(self.trainResult)
        self.testResult = self.normalizeResult(self.testResult)
        

    def readOneArrayFromFiles(self,fileArr,sizeOfReshape,resultArr):
        ret = resultArr
        for i in range(len(fileArr)):
            readFile = ReadFile(fileArr[i])
            arr = readFile.read_file()
            arr = np.array(arr)
            arr = np.reshape(arr,(-1,sizeOfReshape))
            if(len(ret) == 0):
                ret = arr
            else :
                ret = np.concatenate([ret,arr])
        return  np.copy(ret)

    def fixMeanForAllImages(self,data):
        for i in range(len(data)):
            self.fixOneImage(data[i])
        

    def fixOneImage(self,trainSet):
        
        #get max
        images = trainSet
        
        minOne = 99999.9
        maxOne = 0.0
        for j in range(len(images)):
            
            if(images[j] > maxOne):
                maxOne = images[j]
            if(images[j] < minOne):
                minOne = images[j]
        
        if(minOne == 99999.9):
            minOne = 0.0
        if(maxOne == 0.0):
            maxOne = 0.0
        #print("max : " , maxOne)
        #print("min : " , minOne)
        retBool = True
        rangePrice = maxOne - minOne
        if(rangePrice == 0):
            rangePrice = 1
            #print('Wrong Array ',images)
            retBool = False
        for k in range(len(images)):
            images[k] = (images[k] - minOne)/rangePrice
        
        return minOne,maxOne,retBool


    def normalizeResult(self,resultArr):
        ret = list()
        for i in range(len(resultArr)):
            resultData = resultArr[i]
            resultNormalized = [[0.0,0.0]]

            resultNormalized = np.array(resultNormalized)
            if(resultData[17] > resultData[0]):
                resultNormalized[0][0] = 1.0
            else :
                if(resultData[17] < resultData[0]):
                    resultNormalized[0][1] = 1.0
            
            if(len(ret) == 0):
                ret = resultNormalized
            else :
                ret = np.concatenate ([ret,resultNormalized])
                     
                    
        return np.copy(ret)



