import os
import codecs
import re


import tensorflow as tf
import numpy as np

class ReadFile:
    def __init__(self,_file_path):
        self.file_path = _file_path
        print(self.file_path)
    
    def readMultiFiles(self):
        ret = list()
        files = self.file_path.split(";")
        for i in range(len(files)):
            arr = self.read_file_str(files[i])
            ret.extend(arr)
        

        return ret

    def read_file(self):
        ret = list()
        with codecs.open(os.path.join( self.file_path),'r','windows-1252' ) as f :
            
            for line in f :
                line = re.sub(r'[^0-9\.]','',line)
                if(line != ''):
                    ret.insert(len(ret),float(line))
        

        return ret
    def read_file_str(self,fileStr):
        ret = list()
        with codecs.open(os.path.join( fileStr),'r','windows-1252' ) as f :
            
            for line in f :
                line = re.sub(r'[^0-9\.]','',line)
                if(line != ''):
                    ret.insert(len(ret),float(line))
        

        return ret


#fileObj = ReadFile('/home/shohdi/.wine32/drive_c/Program Files/MetaTrader 5/MQL5/Files/myData.csv')
#print(fileObj.read_file())

