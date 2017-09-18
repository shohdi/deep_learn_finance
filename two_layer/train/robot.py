#import liberaries
import os
import tensorflow as tf
import numpy as np

from train.tensorflow_helper import TensorflowHelper
from train.read_date_data import ReadDateFile

#create flags
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('input_dir','input','Input Directory')
flags.DEFINE_string('output_dir','output','Output Directory')


#from tensorflow.examples.tutorials.mnist import input_data
#mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)

def run_training():
    print("start , will open session now")
    #get inputs
    with tf.Session() as sess:
        print("start the session")
        model = TensorflowHelper()
        year1 = model.read_tensor_flow_file('trainYear1.csv',sess,FLAGS)
        year2 = model.read_tensor_flow_file('trainYear2.csv',sess,FLAGS)
        year3 = model.read_tensor_flow_file('trainYear3.csv',sess,FLAGS)
        test_data = model.read_tensor_flow_file('testYear.csv',sess,FLAGS)
        data = ReadDateFile()
        test = ReadDateFile()
        years = list()
        years.extend(year1)
        years.extend(year2)
        years.extend(year3)
        data.readArr(np.array(years))
        test.readArr(np.array(test_data))
        model.threeBoltzYearLayer(40,3)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        #saver.restore(sess,os.path.join(FLAGS.output_dir,"checkout-1000000"))
        i = 0
        while(i < (32000000)):
            i = i + 1
            batch = data.next_batch(50,data.myDataImages,data.myDataLabels)
            #batch = data.next_batch(50,mnist.train.images,mnist.train.labels)
            
            if(i%1000 == 0 or i == 1):
                train_accuracy = model.accuracy.eval(feed_dict= {model.x:batch[0],model.y_:batch[1],model.keep:1.0})    
                print('train accuricy for step %d is %g' % (i,train_accuracy))
            if(i%1000000 == 0 or i == 1000):
                saver.save(sess, os.path.join(FLAGS.output_dir,"checkout"),global_step=i)
            model.trainStep.run(feed_dict= {model.x:batch[0],model.y_:batch[1],model.keep:0.5})
       
        saver.save(sess, os.path.join(FLAGS.output_dir,"checkout"),global_step=32000000)


    #test result
        print('train accuracy for  test : %g' % model.accuracy.eval(feed_dict={model.x:test.myDataImages,model.y_:test.myDataLabels,model.keep:1.0}))




def main(_):
    run_training()

if __name__ == '__main__':
    tf.app.run()

