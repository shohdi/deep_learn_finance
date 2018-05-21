import tensorflow as tf;
from test.lstm_test_program import LstmTestProgram



#1/1 - 1/50 - 1/100 - 1/500 - 1/1000


def main(_):
    me = LstmTestProgram();
    me.run(_);
    return ;
    
        

    
    
   
    

if __name__ == '__main__':
    tf.app.run();