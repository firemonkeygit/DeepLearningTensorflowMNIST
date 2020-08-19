
#官方训练示例1——基于最小均方误差的线性判别函数分类权值训练

import numpy as np
import tensorflow as tf

# 生成 2*100的 数据,100条(x1,x2)二维数据
x_data = np.float32(np.random.rand(2, 100))
print(x_data.shape)
# y = wx + b
y_data = np.dot([0.100, 0.200], x_data) + 0.300
print(y_data)

#设定初始参数值 b (1) w (1*2)
x = tf.placeholder(tf.float32, [2, 100], name="x")
y = tf.placeholder(tf.float32, [1, 100], name="y")
b = tf.Variable(tf.zeros([1]))
w = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
y_output = tf.matmul(w, x) + b
#定义优化目标函数
loss = tf.reduce_mean(tf.square(y_output-y))
#构建优化器
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)
#初始化,variable变量必须初始化
#用 tf.global_variables_initializer() 替代 tf.initialize_all_variables()，后者deprecated
#init = tf.initialize_all_variables()
init_new=tf.global_variables_initializer()
# #启动图
# sess = tf.Session()
# #会话运行
# sess.run(init_new)
# for step in range(0,201):
#     sess.run(train, feed_dict={x: x_data, y: y_data.reshape(1, -1)})
# if step % 20 == 0:
#     print("第%s迭代2，w为%s,b为%s" % (step, sess.run(w), sess.run(b)))
with tf.Session() as sess:
    sess.run(init_new)
    for step in range(0,201):
        sess.run(train,feed_dict={x:x_data,y:y_data.reshape(1,-1)})
    if step % 20 == 0:
        print("第{0}次迭代，w为{1},b为{2}".format(step, sess.run(w), sess.run(b)))

