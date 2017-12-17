import numpy as np




class DrawInput:


    def drawAllInputs(self,inputTuble,arr):
        ret = [];
        for i in range (len(inputTuble)):
            key = inputTuble[i];
            inputArr = arr[key[0]:key[1]];
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
        
        for i in range(len(arr)):
            candle = arr[i];
            dir = 0;
            if(candle[0] > candle[1]):
                dir = 128;
            
            drawPage = np.array(shape=(widthPixels,widthPixels));
            down = 0;
            if(dir == 0):
                down = candle[0];
                up = candle[1];
            else :
                down = candle[1];
                up = candle[0];
            
            yPlace = down - myMin;
            yPlace = (widthPixels * yPlace)/graphHight;

            yPlace = int(round(yPlace,0));

            yMaxPlace  = up - myMin;
            yMaxPlace = (widthPixels * yMaxPlace)/graphHight;

            indexY = yPlace;

            indexX = xPlace;

            while (indexX < (xPlace+3)):
                while (indexY < (yMaxPlace+1)):
                    


                    indexY = indexY + 1;               


                indexX = indexX + 1;

        

