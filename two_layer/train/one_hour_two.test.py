#import liberaries
import tensorflow as tf
import numpy as np

from tensorflow_helper import TensorflowHelper
from read_date_data import ReadDateFile
#from tensorflow.examples.tutorials.mnist import input_data
#mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)



data = ReadDateFile()
test = ReadDateFile()
newTest = ReadDateFile()
data.readFile("/home/shohdi/Documents/data_with_date/oneMonthTrain.csv")
test.readFile("/home/shohdi/Documents/data_with_date/testYear.csv")



model = TensorflowHelper()

model.twoBoltzYearLayer(40,3)



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    
    result_val =0
    saver.restore(sess,'/home/shohdi/projects/learning-tensorflow/one_layer_robot_data_create_seven/shohdi-14000000')
    print('before train accuracy for  test : %g' % model.accuracy.eval(feed_dict={model.x:test.myDataImages,model.y_:test.myDataLabels,model.keep:1.0}))
    
    '''
    success = 0.0
    fail = 0.0
    for i in range(len(test.myDataImages)):
        images = [test.myDataImages[i]]
        labels = [test.myDataLabels[i]]
        
        result = sess.run(model.yOut,feed_dict={model.x:images,model.y_:labels,model.keep:1.0})
        #print("result before modify ",result)
        resArr = result[0]
        resArr = np.exp(resArr)/np.sum(np.exp(resArr),axis=0)
        if(resArr[0] >= 0.995 and labels[0][0] == 1.0):
            success = success + 1
        else :
            if(resArr[2] >= 0.995 and labels[0][2] == 1.0):
                success = success + 1
            else:
                if(resArr[0] >= 0.995 or resArr[2] >= 0.995):
                    fail  = fail + 1
        
        print("result ",resArr)
    print("success ",success,"fail ",fail)
    '''
    
    
    
             
            
    
        

            
    
    








