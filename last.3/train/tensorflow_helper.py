import tensorflow as tf
import numpy as np
import os
import re




class TensorflowHelper :
    def __init__(self):
        self.trainStep = None
        self.accuracy = None
        self.yOut = None
        self.y_ = None
        self.x = None
        self.keep = None
        

    
    



    def read_tensor_flow_file(self,file_name,FLAGS):
        csv_file1 = os.path.join(FLAGS.input_dir,file_name)
        csv_path = tf.train.string_input_producer([csv_file1])
        textReader = tf.TextLineReader()
        key,value = textReader.read(csv_path)
        imageRow = list()
        for i in range(43):
            imageRow.append([0.0])
        
        batch_size  = 1
        min_after_dequeue = 10000
        capacity = min_after_dequeue + 3 * batch_size
        tfRow = tf.decode_csv(value, record_defaults=imageRow)
        images = tfRow[0:40]
        labels = tfRow[40:44]
        return images,labels
        
        
    def getBatch(self,batch_size,fileImages,fileLabels):
        min_after_dequeue = batch_size * 20
        capacity = min_after_dequeue + 3 * batch_size
        return tf.train.shuffle_batch([fileImages,fileLabels],batch_size=batch_size,capacity=capacity,min_after_dequeue=min_after_dequeue)
        

    def read_tensor_ready_file(self,file_name,sess,FLAGS):
        csv_file1 = os.path.join(FLAGS.input_dir,file_name)
        

        strRet = sess.run( tf.read_file(tf.constant(csv_file1)))
        data_string = strRet.decode("windows-1252")
        
        stringList = [float(re.sub(r'[^0-9\.]','',s)) for s in data_string.splitlines() if re.sub(r'[^0-9\.]','',s) != '']
        print("list len for file ",file_name,len(stringList))
        return stringList
        

            
    #helper methods
    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape,stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        initial=tf.constant(0.1,shape=shape)
        return tf.Variable(initial)

    #helper methods to initialize convolution and pool
    def conv2d(self,x,W):
        return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

    def max_pool_2x2(self,x):
        return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


    #convolution model
    def conv2LayerModel(self,inputSize1,inputSize2,outputSize):
        #create input  array
        inputSize = inputSize1 * inputSize2
        self.x = tf.placeholder(tf.float32,[None,inputSize])
        #create output expected array
        self.y_ = tf.placeholder(tf.float32,[None,outputSize])

        

        xEncoded = self.x


        W2 = self.weight_variable([5,5,1,32])
        b2 = self.bias_variable([32])

        xEncReshape = tf.reshape(xEncoded,[-1,inputSize1,inputSize2,1])
        h2 = tf.nn.relu(self.conv2d(xEncReshape,W2)+b2)
        h2Pool = self.max_pool_2x2(h2)

        W3 = self.weight_variable([5,5,32,64])
        b3 = self.bias_variable([64])

        h3 = tf.nn.relu(self.conv2d(h2Pool,W3)+b3)
        h3Pool = self.max_pool_2x2(h3)
        lastSize1 = int(inputSize1/(2.0*2.0))
        lastSize2 = int(inputSize2/(2.0*2.0))
        W4 = self.weight_variable([ lastSize1 * lastSize2 *64,lastSize1 * lastSize2])
        b4 = self.bias_variable([lastSize1 * lastSize2])


        h3PoolReshape = tf.reshape(h3Pool,[-1,lastSize1 * lastSize2 *64])

        h4 = tf.nn.relu(tf.matmul(h3PoolReshape,W4)+b4)

        self.keep = tf.placeholder(tf.float32)

        h4Drop = tf.nn.dropout(h4,self.keep)

        outputShape = [lastSize1 * lastSize2,outputSize]
        W5 = self.weight_variable(outputShape)
        b5 = self.bias_variable([outputSize])

        self.yOut = tf.matmul(h4Drop,W5) + b5

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_,logits=self.yOut))

        self.trainStep = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


        #define prediction way
        correct_prediction = tf.equal(tf.argmax(self.yOut,1),tf.argmax(self.y_,1))

        #accuracy

        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


    def oneLayer(self,inputSize,outputSize):
    
        #create input  array
        self.x = tf.placeholder(tf.float32,[None,inputSize])
        #create output expected array
        self.y_ = tf.placeholder(tf.float32,[None,outputSize])

        #encoding layer
        inputShape = [inputSize,outputSize]
        W1 = self.weight_variable(inputShape)
        b1 = self.bias_variable([outputSize])

        self.yOut = tf.matmul(self.x , W1)+b1

        
        self.keep = tf.placeholder(tf.float32)

        #h1Drop = tf.nn.dropout(h1,self.keep)

        #outputShape = [128,outputSize]
        #W2 = self.weight_variable(outputShape)
        #b2 = self.bias_variable([outputSize])

        #self.yOut = tf.matmul(h1Drop,W2) + b2

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_,logits=self.yOut))

        #self.trainStep = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        self.trainStep = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

        #define prediction way
        correct_prediction = tf.equal(tf.argmax(self.yOut,1),tf.argmax(self.y_,1))

        #accuracy

        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    def stackEncoder(self,inputSize,outputSize):
        #create input  array
        self.x = tf.placeholder(tf.float32,[None,inputSize])
        #create output expected array
        self.y_ = tf.placeholder(tf.float32,[None,outputSize])

        #encoding layer

        bDemo = self.bias_variable([inputSize])

        inputShape = [inputSize,128]
        W1 = self.weight_variable(inputShape)
        b1 = self.bias_variable([128])

        h1 = tf.nn.relu(tf.matmul(self.x , W1)+b1)
        h1DemoDrop = tf.nn.dropout(h1,0.5)

        W1Demo = self.weight_variable([128,inputSize])
        

        h1Demo = tf.matmul(h1DemoDrop , W1Demo)+bDemo

        self.h1Cross = -1 * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.x,logits=h1Demo))
        self.h1UTrain = tf.train.AdamOptimizer(1e-4).minimize(self.h1Cross)

        W2 = self.weight_variable([128,64])
        b2 = self.bias_variable([64])

        h2 = tf.nn.relu(tf.matmul(h1,W2)+b2)
        h2DemoDrop = tf.nn.dropout(h2,0.5)
        W2Demo = self.weight_variable([64,inputSize])

        h2Demo =  tf.matmul(h2DemoDrop,W2Demo)+bDemo

        self.h2Cross = -1 * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.x,logits=h2Demo))
        self.h2UTrain = tf.train.AdamOptimizer(1e-4).minimize(self.h2Cross)

        W3 = self.weight_variable([64,32])
        b3 = self.bias_variable([32])

        h3 = tf.nn.relu(tf.matmul(h2,W3)+b3)
        h3DemoDrop = tf.nn.dropout(h3,0.5)
        W3Demo = self.weight_variable([32,inputSize])

        h3Demo = tf.matmul(h3DemoDrop,W3Demo)+bDemo

        self.h3Cross = -1 * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.x,logits=h3Demo))
        self.h3UTrain = tf.train.AdamOptimizer(1e-4).minimize(self.h3Cross)

        
        self.keep = tf.placeholder(tf.float32)

        hDrop = tf.nn.dropout(h3,self.keep)

        outputShape = [32,outputSize]
        Wf = self.weight_variable(outputShape)
        bf = self.bias_variable([outputSize])

        self.yOut =  tf.matmul(hDrop,Wf) + bf

        self.cross_entropy = -1 * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_,logits=self.yOut))

        self.trainStep = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)
        #self.trainStep = tf.train.GradientDescentOptimizer(0.05).minimize(self.cross_entropy)

        #define prediction way
        correct_prediction = tf.equal(tf.argmax(self.yOut,1),tf.argmax(self.y_,1))
        

        #accuracy

        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


    

    


    def twoCustomBoltzLayer(self,inputSize,outputSize):
        #create input  array
        self.x = tf.placeholder(tf.float32,[None,inputSize])
        #create output expected array
        self.y_ = tf.placeholder(tf.float32,[None,outputSize])

        #encoding layer
        inputShape = [inputSize,1024]
        W1 = self.weight_variable(inputShape)
        b1 = self.bias_variable([1024])

        h1 = tf.nn.tanh(tf.matmul(self.x , W1)+b1)

        
        self.keep = tf.placeholder(tf.float32)

        h1Drop = tf.nn.dropout(h1,self.keep)

        outputShape = [1024,outputSize]
        W2 = self.weight_variable(outputShape)
        b2 = self.bias_variable([outputSize])

        self.yOut = tf.matmul(h1Drop,W2) + b2

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_,logits=self.yOut))

        #self.trainStep = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        self.trainStep = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

        #define prediction way
        correct_prediction = tf.equal(tf.argmax(self.yOut,1),tf.argmax(self.y_,1))
        

        #accuracy

        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    def threeBoltzYearLayer(self,inputSize,outputSize):
        #create input  array
        self.x = tf.placeholder(tf.float32,[None,inputSize])
        #create output expected array
        self.y_ = tf.placeholder(tf.float32,[None,outputSize])

        #encoding layer
        inputShape = [inputSize,2048]
        W1 = self.weight_variable(inputShape)
        b1 = self.bias_variable([2048])

        h1 = tf.nn.relu(tf.matmul(self.x , W1)+b1)


        W2 = self.weight_variable([2048,60])
        b2 = self.bias_variable([60])

        h2 = tf.nn.relu(tf.matmul(h1 , W2)+b2)
        
        self.keep = tf.placeholder(tf.float32)

        hfDrop = tf.nn.dropout(h2,self.keep)

        outputShape = [60,outputSize]
        Wf = self.weight_variable(outputShape)
        bf = self.bias_variable([outputSize])

        self.yOut = tf.matmul(hfDrop,Wf) + bf

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_,logits=self.yOut))

        self.trainStep = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        #self.trainStep = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

        #define prediction way
        correct_prediction = tf.equal(tf.argmax(self.yOut,1),tf.argmax(self.y_,1))

        #accuracy

        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


    
    
    
    def twoBoltzYearLayer(self,inputSize,outputSize):
        #create input  array
        self.x = tf.placeholder(tf.float32,[None,inputSize])
        #create output expected array
        self.y_ = tf.placeholder(tf.float32,[None,outputSize])

        #encoding layer
        inputShape = [inputSize,2048]
        W1 = self.weight_variable(inputShape)
        b1 = self.bias_variable([2048])

        h1 = tf.nn.relu(tf.matmul(self.x , W1)+b1)

        
        self.keep = tf.placeholder(tf.float32)

        h1Drop = tf.nn.dropout(h1,self.keep)

        outputShape = [2048,outputSize]
        W2 = self.weight_variable(outputShape)
        b2 = self.bias_variable([outputSize])

        self.yOut = tf.matmul(h1Drop,W2) + b2

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_,logits=self.yOut))

        self.trainStep = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        #self.trainStep = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

        #define prediction way
        correct_prediction = tf.equal(tf.argmax(self.yOut,1),tf.argmax(self.y_,1))

        #accuracy

        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


    
    def twoBoltzLayer(self,inputSize,outputSize):
        #create input  array
        self.x = tf.placeholder(tf.float32,[None,inputSize])
        #create output expected array
        self.y_ = tf.placeholder(tf.float32,[None,outputSize])

        #encoding layer
        inputShape = [inputSize,128]
        W1 = self.weight_variable(inputShape)
        b1 = self.bias_variable([128])

        h1 = tf.nn.relu(tf.matmul(self.x , W1)+b1)

        
        self.keep = tf.placeholder(tf.float32)

        h1Drop = tf.nn.dropout(h1,self.keep)

        outputShape = [128,outputSize]
        W2 = self.weight_variable(outputShape)
        b2 = self.bias_variable([outputSize])

        self.yOut = tf.matmul(h1Drop,W2) + b2

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_,logits=self.yOut))

        self.trainStep = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        #self.trainStep = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

        #define prediction way
        correct_prediction = tf.equal(tf.argmax(self.yOut,1),tf.argmax(self.y_,1))

        #accuracy

        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


    def twoLayer(self,inputSize,outputSize):
        #create input  array
        self.x = tf.placeholder(tf.float32,[None,inputSize])
        #create output expected array
        self.y_ = tf.placeholder(tf.float32,[None,outputSize])

        #encoding layer
        inputShape = [inputSize,128]
        W1 = self.weight_variable(inputShape)
        b1 = self.bias_variable([128])

        h1 = tf.nn.relu(tf.matmul(self.x , W1)+b1)

        
        

       

        
        W2 = self.weight_variable([128,128])
        b2 = self.bias_variable([128])


        h2 = tf.nn.relu(tf.matmul(h1 , W2)+b2)

        outputShape = [128,outputSize]
        Wf = self.weight_variable(outputShape)
        bf = self.bias_variable([outputSize])

        self.keep = tf.placeholder(tf.float32)
        h2Drop = tf.nn.dropout(h2,self.keep)
        self.yOut = tf.matmul(h2Drop,Wf) + bf

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_,logits=self.yOut))

        self.trainStep = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


        #define prediction way
        correct_prediction = tf.equal(tf.argmax(self.yOut,1),tf.argmax(self.y_,1))

        #accuracy

        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


    def customTwoHidden(self,inputSize,outputSize):
        #create input  array
        self.x = tf.placeholder(tf.float32,[None,inputSize])
        #create output expected array
        self.y_ = tf.placeholder(tf.float32,[None,outputSize])

        #encoding layer
        inputShape = [inputSize,8192]
        W1 = self.weight_variable(inputShape)
        b1 = self.bias_variable([8192])

        h1 = tf.nn.relu(tf.matmul(self.x , W1)+b1)

        
        

       

        
        #W2 = self.weight_variable([inputSize*2,inputSize])
        #b2 = self.bias_variable([inputSize])


        #h2 = tf.nn.relu(tf.matmul(h1 , W2)+b2)

        outputShape = [8192,outputSize]
        Wf = self.weight_variable(outputShape)
        bf = self.bias_variable([outputSize])

        self.keep = tf.placeholder(tf.float32)
        h1Drop = tf.nn.dropout(h1,self.keep)
        self.yOut = tf.matmul(h1Drop,Wf) + bf

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_,logits=self.yOut))

        #self.trainStep = tf.train.AdagradOptimizer(0.05).minimize(cross_entropy)
        self.trainStep = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        #define prediction way
        correct_prediction = tf.equal(tf.argmax(self.yOut,1),tf.argmax(self.y_,1))

        #accuracy

        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))





    #convolution model
    def conv4LayerModel(self,inputSize1,inputSize2,outputSize):
        #create input  array
        inputSize = inputSize1 * inputSize2
        self.x = tf.placeholder(tf.float32,[None,inputSize])
        #create output expected array
        self.y_ = tf.placeholder(tf.float32,[None,outputSize])

        W1 = self.weight_variable([4,4,1,32])
        b1 = self.bias_variable([32])

        xReshaped = tf.reshape(self.x,[-1,inputSize1,inputSize2,1])
        h1 = tf.nn.relu(self.conv2d(xReshaped,W1)+b1)



        


        W2 = self.weight_variable([4,4,32,64])
        b2 = self.bias_variable([64])

        
        h2 = tf.nn.relu(self.conv2d(h1,W2)+b2)

        W3 = self.weight_variable([4,4,64,128])
        b3 = self.bias_variable([128])

        h3 = tf.nn.relu(self.conv2d(h2,W3)+b3)
        
        h3Pool = self.max_pool_2x2(h3)

        W4 = self.weight_variable([4,4,128,256])
        b4 = self.bias_variable([256])
        h4 = tf.nn.relu(self.conv2d(h3Pool,W4)+b4)
        h4Pool = self.max_pool_2x2(h4)
        
        lastSize1 = int(inputSize1/4)
        lastSize2 = int(inputSize2/4)
        lastInputSize = lastSize1 * lastSize2 *256
        h4PoolReshape = tf.reshape(h4Pool,[-1,lastInputSize])
        
        Wf = self.weight_variable([ lastInputSize ,2048])
        bf = self.bias_variable([2048])

        hf = tf.nn.relu(tf.matmul(h4PoolReshape,Wf)+bf)

        self.keep = tf.placeholder(tf.float32)

        hfDrop = tf.nn.dropout(hf,self.keep)

        outputShape = [2048,outputSize]
        Wl = self.weight_variable(outputShape)
        bl = self.bias_variable([outputSize])

        self.yOut = tf.nn.relu(tf.matmul(hfDrop,Wl) + bl)

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_,logits=self.yOut))

        #self.trainStep = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        self.trainStep = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)


        #define prediction way
        correct_prediction = tf.equal(tf.argmax(self.yOut,1),tf.argmax(self.y_,1))

        #accuracy

        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

