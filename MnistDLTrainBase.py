#Tensorflow官方文档

import numpy as np
import tensorflow as tf
from TensorflowTest.mnist import input_data

#权重初始化
def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    #使用的是ReLU神经元，因此比较好的做法是用一个较小的正数来初始化偏置项，以避免神经元节点输出恒为0的问题（dead neurons）
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

#卷积和池化
#卷积使用1步长（stride size），0边距（padding size）的模板，保证输出和输入是同一个大小，池化用简单传统的2x2大小的模板做max pooling。
#x:[batch, in_height, in_width, in_channels]
#w:[filter_height, filter_width, in_channels, out_channels]
#strides:[N,H,W,C] dimensions,in general N=C=1
def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

mnist=input_data.read_data_sets('../../MNIST_data',one_hot=True)
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])
#第一层卷积
#第一层由一个卷积接一个max pooling完成。卷积在每个5x5的patch中算出32个特征。
#卷积的权重张量形状是[5, 5, 1, 32]，前两个维度是patch的大小，接着是输入的通道数目，最后是输出的通道数目。 而对于每一个输出通道都有一个对应的偏置量。
w_conv1=weight_variable([5,5,1,32])
b_conv1=bias_variable([32])
#为了用这一层,把x变成一个4d向量，其第2、第3维对应图片的宽、高，最后一维代表图片的颜色通道数(因为是灰度图所以这里的通道数为1，如果是rgb彩色图，则为3)
x_image=tf.reshape(x,[-1,28,28,1])
h_conv1=tf.nn.relu(conv2d(x_image,w_conv1)+b_conv1)
h_pool1=max_pool_2x2(h_conv1)
#第二层卷积
w_conv2=weight_variable([5,5,32,64])
b_conv2=bias_variable([64])
h_conv2=tf.nn.relu(conv2d(h_pool1,w_conv2)+b_conv2)
h_pool2=max_pool_2x2(h_conv2)
#密集连接层
#图片尺寸减小到7x7，我们加入一个有1024个神经元的全连接层，用于处理整个图片。我们把池化层输出的张量reshape成一些向量，乘上权重矩阵，加上偏置，然后对其使用ReLU
w_fc1=weight_variable([7*7*64,1024])
b_fc1=bias_variable([1024])
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)
#Dropout
keep_prob=tf.placeholder("float")
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)
#输出层
w_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop,w_fc2)+b_fc2)
#训练和评估模型
cross_entropy=-tf.reduce_sum(y_*tf.log(y_conv))
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))

with tf.Session() as sess:
    with tf.device("/cpu:0"):
        sess.run(tf.global_variables_initializer())
        for i in range(1000):
            batch=mnist.train.next_batch(100)
            sess.run(train_step,feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})
            if i%100==0:
                #train_accuracy=sess.run(accuracy,feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
                train_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
                print("step {0},training accuracy {1}".format(i,train_accuracy))
        print("test accuracy {0}".format(sess.run(accuracy,feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})))











