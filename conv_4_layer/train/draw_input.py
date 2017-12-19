import numpy as np
from train.compress_candles import CompressCandles




class DrawInput:
    def __init__(self,compressCandles,multi):
        self.compressCandles = compressCandles;
        self.multi = multi;


    def drawAllInputs(self,inputTuble,arr):
        ret = [];
        for i in range (len(inputTuble)):
            key = inputTuble[i];
            inputArr = arr[key[0]:key[1]];
            inputArr = self.compressCandles.compressCandles(inputArr,self.multi);
            inputArr = inputArr[:,2:];
            newInput = self.drawOneInput(inputArr);
            ret.append(newInput);
        

        return np.array(ret);


    def drawOneInput(self,arr):
        myMax = np.amax(arr);
        myMin = np.amin(arr);
        length = len(arr);
        widthPixels = 2 + (length*5);
        graphHight = myMax - myMin;

        xPlace = 2;
        drawPage = np.empty(shape=(widthPixels,widthPixels),dtype=float);
        drawPage.fill(255);

        for i in range(len(arr)):
            candle = arr[i];
            self.drawOneCandle(xPlace,drawPage,candle,myMin,myMax,widthPixels,graphHight);            
            xPlace = xPlace + 5;

    
        return drawPage;


    def drawOneCandle (self,xPlace,drawPage,candle,myMin,myMax,widthPixels,graphHight):
        dir = 0;
        if(candle[0] > candle[1]):
            dir = 128;
        
        #draw candle body
        down = 0;
        up = 0;
        if(dir == 0):
            down = candle[0];
            up = candle[1];
        else :
            down = candle[1];
            up = candle[0];
        
        yPlace = self.calculateY(down,myMin,widthPixels,graphHight);
        yMaxPlace = self.calculateY(up,myMin,widthPixels,graphHight);

        


        indexY = yMaxPlace;

        indexX = xPlace;

        drawPage = self.drawBox(dir,indexX,indexX+3,indexY,yPlace+1,drawPage);
        #draw high and low
        indexX = indexX + 1;
        high = candle[2];
        low = candle[3];

        yPlace = self.calculateY(low,myMin,widthPixels,graphHight);
        yMaxPlace = self.calculateY(high,myMin,widthPixels,graphHight);
       
        indexY = yMaxPlace;
        drawPage = self.drawBox(dir,indexX,indexX+1,indexY,yPlace+1,drawPage);

    def calculateY (self,val,myMin,widthPixels,graphHight):
        yPlace = val - myMin;
        yPlace = ((widthPixels-1) * yPlace)/graphHight;
        yPlace = int(round(yPlace,0));
        if(yPlace > (widthPixels-1)):
            yPlace = widthPixels-1;
        return (widthPixels - 1) - yPlace;
        

    def drawBox(self,dir,xStart,xEnd,yStart,yEnd,arr):
        for i in range(xEnd - xStart):
            for j in range(yEnd - yStart):
                xIndex = i+xStart;
                yIndex = j+yStart;
                arr[yIndex,xIndex] = dir;

        return arr;

        

