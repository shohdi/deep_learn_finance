import os
import codecs
import re


import tensorflow as tf
import numpy as np

class ReadFile:
        
    
    def readMultiFiles(self,file_path):
        ret = list()
        files = file_path.split(";")
        for i in range(len(files)):
            arr = self.read_file(files[i])
            ret.extend(arr)
        

        return ret

    def read_file(self,fileStr):
        ret = list()
        with codecs.open(os.path.join(fileStr),'r','windows-1252' ) as f :
            
            for line in f :
                line = re.sub(r'[^0-9\.]','',line)
                if(line != ''):
                    ret.insert(len(ret),float(line))
        

        return ret
    


#fileObj = ReadFile('/home/shohdi/.wine32/drive_c/Program Files/MetaTrader 5/MQL5/Files/myData.csv')
#print(fileObj.read_file())

