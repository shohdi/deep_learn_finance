
import numpy as np
import tensorflow as tf

import os as os;






from train.my_flags import MyFlags
from train.main_input_loop import MainInputLoop;



#1/1 - 1/50 - 1/100 - 1/500 - 1/1000


def main(_):
    myFlags = MyFlags();
    
    mainInputLoop = MainInputLoop();

    mainInputLoop.normalizeInput(myFlags.trainFiles);
    



    return ;
    
        

    
    
   
    

if __name__ == '__main__':
    tf.app.run();