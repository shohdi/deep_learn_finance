import numpy as np


class CompressCandles:
    def compressAllCandles(self,arr,multiply):
        arrRet = [];
        lenth = len(arr);
        for i in range(lenth):
            oneArray = arr[i];
            newArr = self.compressCandles(oneArray,multiply);
            arrRet.append(newArr);
        return np.array(arrRet);



    def compressCandles(self,arr,multi):
        ret = [];
        lenth = len(arr);
        newLen = (lenth/multi);
        newLen = int(newLen);
        start =  lenth - (newLen * multi);
        newArr = arr[start:];
        newArr = newArr.reshape(newLen,multi,6);
        for i in range(newLen):
            multiArr = newArr[i];
            arr = self.compressOneMulti(multiArr);
            ret.append(arr);
    
        return np.array(ret);


    def compressOneMulti(self,arr):
        ret = [];
        ret.append(arr[0][0]);
        ret.append(arr[0][1]);

        length =len(arr);
        ret.append(arr[0][2]);
        ret.append(arr[length-1][3]);
        newArr = arr[:,2:];
        ret.append(np.amax(newArr));
        ret.append(np.amin(newArr));




        return ret;