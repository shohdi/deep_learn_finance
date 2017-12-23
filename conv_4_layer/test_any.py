
import numpy as np
import tensorflow as tf

import os as os;

from train.deep_input_ret import DeepInputRet

flags = tf.app.flags;
FLAGS = flags.FLAGS;
flags.DEFINE_string('shohdi_debug','True','shohdi_debug');
flags.DEFINE_integer('INPUT_SIZE',60,'INPUT_SIZE');
flags.DEFINE_integer('OUTPUT_SIZE',15,'OUTPUT_SIZE');
flags.DEFINE_integer('HOW_MANY_MINUTES',10,'HOW_MANY_MINUTES');
flags.DEFINE_string('INPUT_FOLDER','input','INPUT_FOLDER');




#1/1 - 1/50 - 1/100 - 1/500 - 1/1000


def main(_):
    
    FILE_NAMES = os.path.join(FLAGS.INPUT_FOLDER,'myOldData.csv');

    
    
    inputClass = DeepInputRet(FLAGS.INPUT_SIZE,FLAGS.OUTPUT_SIZE,FILE_NAMES,FLAGS.HOW_MANY_MINUTES);
    train_x,train_y = inputClass.getSuccessFailData();
    

if __name__ == '__main__':
    tf.app.run()