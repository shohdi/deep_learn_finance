

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
        #xTest,yTest =  self.mainInputLoop.normalizeInput(self.myFlags.testFiles);
        
        valPerc = self.myFlags.valSplit;
        valLength = int(len(x) * valPerc);
        testPerc = self.myFlags.testSplit;
        testLengh = int(len(x) * testPerc);
        xTest =  x[-testLengh:];
        yTest = y[-testLengh:];
        x = x[:(-testLengh-self.myFlags.INPUT_SIZE)];
        y = y[:(-testLengh-self.myFlags.INPUT_SIZE)];
        xVal = x[-valLength:];
        yVal = y[-valLength:];
        x = x[:(-valLength-self.myFlags.INPUT_SIZE)];
        y = y[:(-valLength-self.myFlags.INPUT_SIZE)];
        
        print('the shape of x %s , the shape of y %s , the shape of xTest %s the shape of yTest %s the shape of xVal %s , the shape of yVal %s' %(x.shape,y.shape,xTest.shape,yTest.shape,xVal.shape,yVal.shape));

        model = self.kerasModel.buildModel();
        self.kerasModel.trainModel(model,x,y,xVal,yVal,xTest,yTest);
        
