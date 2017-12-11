from train.read_file import ReadFile
import numpy as np


class ForexFix:
    def __init__(self,history,future):
        self.history = history
        self.future = future
        self.readFile = ReadFile("/home/shohdi/projects/learning-tensorflow/projects/deep_learn_finance/conv_4_layer/input/myOldData.csv")
    


    def getInputOutput(self):
        mainArr = self.readFile.readMultiFiles()
        print(mainArr)

        





