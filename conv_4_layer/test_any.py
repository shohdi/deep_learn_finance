
import numpy as np
import scipy.misc as smp

from train.deep_input_ret import DeepInputRet





#1/1 - 1/50 - 1/100 - 1/500 - 1/1000


def main(_):
    INPUT_SIZE = 60;
    OUTPUT_SIZE = 15;
    HOW_MANY_MINUTES = 10;
    FILE_NAMES = "/home/shohdi/projects/learning-tensorflow/projects/deep_learn_finance/conv_4_layer/input/myOldData.csv";
    
    inputClass = DeepInputRet(INPUT_SIZE,OUTPUT_SIZE,FILE_NAMES,HOW_MANY_MINUTES);
    train_x,train_y = inputClass.getSuccessFailData();
    

main('')