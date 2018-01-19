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
flags.DEFINE_string('batch_size','4','Batch Size')



#from tensorflow.examples.tutorials.mnist import input_data
#mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)

def run_training():
    print("start , will open session now")
    print("batch size " ,FLAGS.batch_size)
    batchSizeInt = int(FLAGS.batch_size)
    print("Batch size int",batchSizeInt)
    #get inputs
    #config = tf.ConfigProto(inter_op_parallelism_threads=1)
    with tf.Session() as sess:
        model = TensorflowHelper()
        #fileNames = os.path.join(FLAGS.input_dir,'trainYear1.csv') +";" + os.path.join(FLAGS.input_dir,'trainYear2.csv') + ";" +os.path.join(FLAGS.input_dir,'trainYear3.csv')
        fileNames = os.path.join(FLAGS.input_dir,'trainYear3.csv')
        #fileNames = os.path.join(FLAGS.input_dir,'testYear.csv')
        print("file names ",fileNames)
        
        
        data = ReadDateFile()
        test = ReadDateFile()
       
        data.readFile(fileNames)
        test.readFile(os.path.join(FLAGS.input_dir,'testYear.csv'))
        model.conv4LayerModel(32,4,3)
        sess.run(tf.global_variables_initializer())
        
        print("start the session")
        saver = tf.train.Saver()
        #images,labels = model.read_tensor_flow_file('trainData.csv',FLAGS)
        #trainBatchTF = model.getBatch(int(FLAGS.batch_size),images,labels)
        #testImagesTf,testLabelsTf = model.read_tensor_flow_file('testData.csv',FLAGS)
        #testBatchTf = model.getBatch(500,testImagesTf,testLabelsTf)
        
       
        #testImages,testLabels = sess.run([images,labels])
        
        
        #saver.restore(sess,os.path.join(FLAGS.output_dir,"checkout-1000000"))
        #coord = tf.train.Coordinator()
        #threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        

        i = 0
        while(i < (32000000)):
            i = i + 1
            #batch = sess.run( trainBatchTF )

            batch = data.next_batch(batchSizeInt,data.myDataImages,data.myDataLabels)
            
            if(i%1000 == 0 or i == 1):
                batchForValidate = data.next_batch(213,data.myDataImages,data.myDataLabels)
                train_accuracy = model.accuracy.eval(feed_dict= {model.x:batchForValidate[0],model.y_:batchForValidate[1],model.keep:1.0})    
                print('train accuricy for step %d is %g' % (i,train_accuracy))
            if(i%1000000 == 0):
                saver.save(sess, os.path.join(FLAGS.output_dir,"checkout"),global_step=i)
            model.trainStep.run(feed_dict= {model.x:batch[0],model.y_:batch[1],model.keep:0.5})
       
        saver.save(sess, os.path.join(FLAGS.output_dir,"checkout"),global_step=32000000)


    #test result
        for i in range(20):
            testBatch = test.next_batch(500,test.myDataImages,test.myDataLabels)
            print('train accuracy for  test : %g' % model.accuracy.eval(feed_dict={model.x:testBatch[0],model.y_:testBatch[1],model.keep:1.0}))

        


def main(_):
    run_training()

if __name__ == '__main__':
    tf.app.run()

