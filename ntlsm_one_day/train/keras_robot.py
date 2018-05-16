import tensorflow as tf
from train.program_class import ProgramClass



#1/1 - 1/50 - 1/100 - 1/500 - 1/1000


def main(_):
    programClass = ProgramClass();
    programClass.run(_);

    return ;
    
        

    
    
   
    

if __name__ == '__main__':
    tf.app.run();