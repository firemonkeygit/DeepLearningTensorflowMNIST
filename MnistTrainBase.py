import numpy as np
import tensorflow as tf
from TensorflowTest.mnist import input_data

###mnist基础-MNIST机器学习入门_习惯成就伟大-CSDN博客
###？？？问题：训练后的w值没有改变
#0获取数据集
mnist=input_data.read_data_sets('../../MNIST_data',one_hot=True)

# #1训练模型之矩阵转置版
# x=tf.placeholder('float',[784,None])#100*784矩阵
# w=tf.Variable(tf.zeros([10,784]))#
# #w = tf.Variable(tf.random_uniform([784, 10], -1.0, 1.0))
# #w=tf.Variable(tf.truncated_normal(([784,10]),stddev=0.1))
# b=tf.Variable(tf.zeros([10,1]))#1*10矩阵
#
# y=tf.nn.softmax(tf.matmul(w,x)+b)
# y_=tf.placeholder('float',[10,None])
# loss=-tf.reduce_sum(-y_*tf.log(y))#损失函数
# optimizer=tf.train.GradientDescentOptimizer(0.01)#优化器
# train=optimizer.minimize(loss)#训练
#
# init=tf.global_variables_initializer()
# with tf.Session() as sess:
#     with tf.device("/cpu:0"):
#         sess.run(init)
#         for i in range(1000):
#             batch_xs,batch_ys=mnist.train.next_batch(100)
#             xs=batch_xs.transpose()
#             ys=batch_ys.transpose()
#             sess.run(train,feed_dict={x:xs,y_:ys})
#         print("w训练值为：{0},b训练值为：{1}".format(sess.run(w),sess.run(b)))
#         #print("y值为：{0}".format(sess.run(y,feed_dict={x:batch_xs})))
#
# correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
# accuracy=tf.reduce_mean(tf.cast(correct_prediction,'float'))
# with tf.Session() as sess:
#     sess.run(init)
#     print("训练准确率为{0}".format(sess.run(accuracy,feed_dict={x:mnist.test.images.transpose(),y_:mnist.test.labels.transpose()})))

#1训练模型
x=tf.placeholder('float',[None,784])#100*784矩阵
w=tf.Variable(tf.zeros([784,10]))#
#w = tf.Variable(tf.random_uniform([784, 10], -1.0, 1.0))
#w=tf.Variable(tf.truncated_normal(([784,10]),stddev=0.1))
b=tf.Variable(tf.zeros([10]))#1*10矩阵

y=tf.nn.softmax(tf.matmul(x,w)+b)
yLabel=tf.placeholder('float',[None,10])
loss=-tf.reduce_sum(yLabel*tf.log(y+ 1e-10))#损失函数使用交叉商
# optimizer=tf.train.GradientDescentOptimizer(0.01)#优化器
# train=optimizer.minimize(cross_enpy)#训练
train_step=tf.train.GradientDescentOptimizer(0.01).minimize(loss)

correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(yLabel,1))#返回数据类型为bool的张量
accuracy=tf.reduce_mean(tf.cast(correct_prediction,'float'))#正确率
acc_=[]
init=tf.global_variables_initializer()
with tf.Session() as sess:
    with tf.device("/cpu:0"):
        sess.run(init)
        for i in range(1000):#训练次数n1（可将n2包括在其中，省略一次j循环）
            #for j in range(500):#一次完整样本训练的batch数量n2，50000/100，本循环也可省略
            batch_xs,batch_ys=mnist.train.next_batch(100)
            sess.run(train_step,feed_dict={x:batch_xs,yLabel:batch_ys})
            lossValue,accuracyValue=sess.run([loss, accuracy], feed_dict={x: mnist.test.images, yLabel: mnist.test.labels})
            acc_.append(accuracyValue)
            yValue=sess.run(y,feed_dict={x:batch_xs})
            prediction_result = sess.run(tf.argmax(y, 1), feed_dict={x: mnist.test.images})
            #print("loss is ：{0},acc is:{1},train sequeue is :{2} ".format(lossValue, accuracyValue,i))

        #print("w训练值为：{0},b训练值为：{1}".format(sess.run(w),sess.run(b)))
        #print("y值为：{0}".format(yValue))
        print("loss is ：{0},acc is:{1} ".format(lossValue,accuracyValue))


# with tf.Session() as sess:
#     sess.run(init)
#     print("训练准确率为{0}".format())
#入门版本-MNIST 手写数字识别【入门】 - Pam_sh - 博客园
import matplotlib.pyplot as plt
import numpy as np
def plot_images_labels_prediction(images,  # 图像列表
                                  labels,  # 标签列表
                                  prediction,  # 预测值列表
                                  index,  # 从第index个开始显示
                                  num=10):  # 缺省一次显示10幅
    fig = plt.gcf()  # 获取当前图表，
    fig.set_size_inches(10, 12)  # 一英寸为2.54cm
    if num > 25:
        num = 25  # 最多显示25个子图
    for i in range(0, num):
        ax = plt.subplot(5, 5, i + 1)  # 获取当前要处理的子图

        ax.imshow(np.reshape(images[index], (28, 28)),  # 显示第index个图像
                  cmap="binary")
        title = "label=" + str(np.argmax(labels[index]))  # 构建该图上要显示的
        if len(prediction) > 0:
            title += ",predict=" + str(prediction[index])
        ax.set_title(title, fontsize=10)  # 显示图上的title信息
        ax.set_xticks([]);  # 不显示坐标轴
        ax.set_yticks([])
        index += 1
    plt.show()



plot_images_labels_prediction(mnist.test.images,
                             mnist.test.labels,
                             prediction_result,0,10)

plt.plot(acc_)
plt.show()