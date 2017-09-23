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
    #config = tf.ConfigProto(inter_op_parallelism_threads=1)
    with tf.Session() as sess:
        model = TensorflowHelper()
        model.conv4LayerModel(40,3)
        sess.run(tf.global_variables_initializer())
        
        print("start the session")
        saver = tf.train.Saver()
        images,labels = model.read_tensor_flow_file('trainData.csv',FLAGS)
        trainBatchTF = model.getBatch(50,images,labels)
        testImagesTf,testLabelsTf = model.read_tensor_flow_file('testData.csv',FLAGS)
        testBatchTf = model.getBatch(500,testImagesTf,testLabelsTf)
        
       
        #testImages,testLabels = sess.run([images,labels])
        
        
        #saver.restore(sess,os.path.join(FLAGS.output_dir,"checkout-1000000"))
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        

        i = 0
        while(i < (32000000)):
            i = i + 1
            batch = sess.run( trainBatchTF )

            #batch = data.next_batch(50,mnist.train.images,mnist.train.labels)
            
            if(i%100 == 0 or i == 1):
                train_accuracy = model.accuracy.eval(feed_dict= {model.x:batch[0],model.y_:batch[1],model.keep:1.0})    
                print('train accuricy for step %d is %g' % (i,train_accuracy))
            if(i%20000 == 0):
                saver.save(sess, os.path.join(FLAGS.output_dir,"checkout"),global_step=i)
            model.trainStep.run(feed_dict= {model.x:batch[0],model.y_:batch[1],model.keep:0.5})
       
        saver.save(sess, os.path.join(FLAGS.output_dir,"checkout"),global_step=32000000)


    #test result
        for i in range(20):
            testBatch = sess.run(testBatchTf)
            print('train accuracy for  test : %g' % model.accuracy.eval(feed_dict={model.x:testBatch[0],model.y_:testBatch[1],model.keep:1.0}))

        


def main(_):
    run_training()

if __name__ == '__main__':
    tf.app.run()

