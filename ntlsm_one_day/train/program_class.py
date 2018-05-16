

from train.my_flags import MyFlags
from train.main_input_loop import MainInputLoop;



#1/1 - 1/50 - 1/100 - 1/500 - 1/1000

class ProgramClass:
    def __init__(self):
        self.myFlags = MyFlags();
        self.mainInputLoop = MainInputLoop();
    

    def run(self,args):
        x,y = self.mainInputLoop.normalizeInput(self.myFlags.trainFiles);
        xTest,yTest =  self.mainInputLoop.normalizeInput(self.myFlags.testFiles);
        
