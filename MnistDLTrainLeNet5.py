#【Tensorflow】LeNet-5训练MNIST数据集

import numpy as np
import tensorflow as tf
from TensorflowTest.mnist import input_data

#1.transform mnist from 28*28 to 32*32
def mnist_reshape_32(_batch):
    batch=np.reshape(_batch,[-1,28,28])
    num=batch.shape[0]
    batch_32=np.array(np.random.rand(num,32,32),dtype=np.float32)
    for i in range(num):
        batch_32[i]=np.pad(batch[i],2,'constant',constant_values=0)
    return batch_32

# def weight_variable(shape,name):
#     return tf.Variable(tf.truncated_normal(shape,stddev=0.1),nam=name)
# def bias_variable(shape,name):
#     return tf.Variable(tf.constant(0.1,shape=shape),name=name)
# def conv2d(x,w,padding):
#     return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding=padding)
# def relu_bias(x,bias,name):
#     return tf.nn.relu(tf.nn.bias_add(x,bias),name=name)
# def max_pool_2x2(x):
#     return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

keep_prob=tf.placeholder(tf.float32)

def LeNetModel(x,regular):
    x_data=tf.reshape(x,[-1,32,32,1])
    #C1层：卷积层，卷积核为5*5，通道数/深度为6，不适用全0补充，步长为1
    #尺寸变化：32*32*2 to 28*28*6
    with tf.variable_scope('LayerC1'):
        c1weight=tf.Variable(tf.truncated_normal([5,5,1,6],stddev=0.1))
        tf.summary.histogram('c1/weight',c1weight)
        c1bias=tf.Variable(tf.constant(0.1,shape=[6]))
        tf.summary.histogram('c1/bias',c1bias)
        c1conv=tf.nn.conv2d(x_data,c1weight,strides=[1,1,1,1],padding='VALID')
        c1relu=tf.nn.relu(tf.nn.bias_add(c1conv,c1bias))
        tf.summary.histogram('c1/output',c1relu)
    #S2层：池化层，下采样为2*2，使用全0补充，步长为2
    #尺寸变化:28*28*6 to 14*14*6
    with tf.name_scope('LayerS2'):
        s2pool=tf.nn.max_pool(c1relu,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        tf.summary.histogram('s2/output',s2pool)
    #C3层：卷积层，卷积核为5*5，通道数/深度为16，不使用全0补充，步长为1
    #尺寸变化：14*14*6 to 10*10*6
    with tf.variable_scope('LayerC3'):
        c3weight=tf.Variable(tf.truncated_normal([5,5,6,16],stddev=0.1))
        tf.summary.histogram('c3/weight',c3weight)
        c3bias=tf.Variable(tf.constant(0.1,shape=[16]))
        tf.summary.histogram('c3/bias',c3bias)
        c3conv=tf.nn.conv2d(s2pool,c3weight,strides=[1,1,1,1],padding='VALID')
        c3relu=tf.nn.relu(tf.nn.bias_add(c3conv,c3bias))
        tf.summary.histogram('c3/output',c3relu)
    #S4层：池化层，下采样为2*2，使用全0补充，步长为2
    #尺寸变化:10*10*6 to 5*5*16
    with tf.variable_scope('LayerS4'):
        s4pool=tf.nn.max_pool(c3relu,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        tf.summary.histogram('s4/output',s4pool)
    #C5层：卷积层，卷积核为5*5，因为输入为5*5*16的map，所以也可视为全连接层
    #尺寸变化：5*5*16=1*400 to 1*120
    #训练时引入dropout随机将部分节点输出改为0，dropout可以避免过拟合，模型越简单越不容易过拟合
    #训练时引入正则化限制权重的大小，使得模型不能任意拟合训练数据中的随机早上，从而避免过拟合
    s4pool_shape=s4pool.get_shape().as_list()
    size=s4pool_shape[1]*s4pool_shape[2]*s4pool_shape[3]
    s4pool_reshape=tf.reshape(s4pool,[-1,size])
    with tf.variable_scope('LayerC5'):
        c5weight=tf.Variable(tf.truncated_normal([size,120],stddev=0.1))
        tf.summary.histogram('c5/weight',c5weight)
        c5bias=tf.Variable(tf.constant(0.1,shape=[120]))
        tf.summary.histogram('c5/bias',c5bias)
        if regular!=None:
            tf.add_to_collection('loss',regular(c5weight))
        c5=tf.matmul(s4pool_reshape,c5weight)
        c5relu=tf.nn.relu(tf.nn.bias_add(c5,c5bias))
        c5relu=tf.nn.dropout(c5relu,keep_prob)
        tf.summary.histogram('c5/output',c5relu)
    #F6层：全连接层
    #尺寸变化：1*120 to 1*84
    with tf.variable_scope('LayerF6'):
        f6weight=tf.Variable(tf.truncated_normal([120,84],stddev=0.1))
        tf.summary.histogram('f6/weight',f6weight)
        f6bias=tf.Variable(tf.constant(0.1,shape=[84]))
        tf.summary.histogram('f6/bias',f6bias)
        if regular!=None:
            tf.add_to_collection('loss',regular(f6weight))
        f6=tf.matmul(c5relu,f6weight)
        f6relu=tf.nn.relu(tf.nn.bias_add(f6,f6bias))
        f6relu=tf.nn.dropout(f6relu,keep_prob)
        tf.summary.histogram('f6/output',f6relu)
    #OUTPUT层：输出层，基于径向基，近似为全连接层,经过softmax得到分类结果
    #尺寸变化：1*84 to 1*10
    with tf.variable_scope('LayerF7'):
        f7weight=tf.Variable(tf.truncated_normal([84,10],stddev=0.1))
        tf.summary.histogram('f7/weight',f7weight)
        f7bias=tf.Variable(tf.constant(0.1,shape=[10]))
        tf.summary.histogram('f7/bias',f7bias)
        if regular!=None:
            tf.add_to_collection('loss',regular(f7weight))
        f7=tf.matmul(f6relu,f7weight)+f7bias
        tf.summary.histogram('f7/output',f7)
    return f7

mnist=input_data.read_data_sets('../../MNIST_data',one_hot=True)
xs=tf.placeholder(tf.float32,[None,32,32])
yLabels=tf.placeholder(tf.float32,[None,10])
regularizer = tf.contrib.layers.l2_regularizer(0.001)
ys=LeNetModel(xs,regularizer)

cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=ys,labels=tf.argmax(yLabels,1))
with tf.name_scope('lossValue'):
    loss=tf.reduce_mean(cross_entropy)+tf.add_n(tf.get_collection('loss'))
    tf.summary.scalar('lossValue',loss)
#with tf.name_scope('lossValue'):
    train_step=tf.train.GradientDescentOptimizer(0.01).minimize(loss)
correct_prediction=tf.equal(tf.argmax(ys,1),tf.argmax(yLabels,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

xs_test32=mnist_reshape_32(mnist.test.images)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    merged=tf.summary.merge_all()
    writer=tf.summary.FileWriter('logs/',sess.graph)
    for i in range(1000):
        batch_xs,batch_ys=mnist.train.next_batch(100)
        batch_xs32=mnist_reshape_32(batch_xs)
        sess.run(train_step,feed_dict={xs:batch_xs32,yLabels:batch_ys,keep_prob:1.0})
        if i%100==0:
            train_accuracy=sess.run(accuracy,feed_dict={xs:batch_xs32,yLabels:batch_ys,keep_prob:1.0})
            print("step {0},train_accuracy {1}".format(i,train_accuracy))
            rs=sess.run(merged,feed_dict={xs:batch_xs32,yLabels:batch_ys,keep_prob:1.0})
            writer.add_summary(rs,i)
        if i%100==0:
            test_accuracy=sess.run(accuracy,feed_dict={xs:xs_test32,yLabels:mnist.test.labels,keep_prob:1.0})
            print("step {0},test_accuracy {1}".format(i,test_accuracy))
    final_accuracy=sess.run(accuracy,feed_dict={xs:batch_xs32,yLabels:batch_ys,keep_prob:1.0})
    print("final_accuracy {0}".format(final_accuracy))





