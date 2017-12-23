
import numpy as np
import tensorflow as tf

import os as os;

from train.deep_input_ret import DeepInputRet

flags = tf.app.flags;
FLAGS = flags.FLAGS;
flags.DEFINE_string('shohdi_debug','False','shohdi_debug');


INPUT_SIZE = 60;
OUTPUT_SIZE = 15;
HOW_MANY_MINUTES = 10;
FOLDER = "input";


#1/1 - 1/50 - 1/100 - 1/500 - 1/1000


def main(_):
    
    FILE_NAMES = os.path.join(FOLDER,'myOldData.csv');

    
    
    inputClass = DeepInputRet(INPUT_SIZE,OUTPUT_SIZE,FILE_NAMES,HOW_MANY_MINUTES);
    train_x,train_y = inputClass.getSuccessFailData();
    

if __name__ == '__main__':
    tf.app.run()