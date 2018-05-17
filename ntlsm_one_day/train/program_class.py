

from train.my_flags import MyFlags
from train.main_input_loop import MainInputLoop;
from train.keras_model import KerasModel;



#1/1 - 1/50 - 1/100 - 1/500 - 1/1000

class ProgramClass:
    def __init__(self):
        self.myFlags = MyFlags();
        self.mainInputLoop = MainInputLoop();
        self.kerasModel = KerasModel();
    

    def run(self,args):
        x,y = self.mainInputLoop.normalizeInput(self.myFlags.trainFiles);
        xTest,yTest =  self.mainInputLoop.normalizeInput(self.myFlags.testFiles);
        
        valPerc = self.myFlags.valSplit;
        valLength = int(len(x) * valPerc);
        xVal = x[-valLength:];
        yVal = y[-valLength:];
        x = x[:-valLength];
        y = y[:-valLength];
        print ('x length : %d , valsplit %f , valLength %d' % (len(x),valPerc,len(xVal)));
        model = self.kerasModel.buildModel();
        self.kerasModel.trainModel(model,x,y,xVal,yVal,xTest,yTest);
        
