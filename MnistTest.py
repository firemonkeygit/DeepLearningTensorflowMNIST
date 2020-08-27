# from tensorflow.examples.tutorials.mnist import input_data
#1 mnist初步使用代码
from TensorflowTest.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt
# MNIST_data指的是存放数据的文件夹路径，one_hot=True 为采用one_hot的编码方式编码标签
mnist = input_data.read_data_sets('../../MNIST_data/', one_hot=True)
# load data
train_X = mnist.train.images
train_Y = mnist.train.labels
print(train_X.shape, train_Y.shape)   # 输出训练集样本和标签的大小
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)
# 查看数据，例如训练集中第一个样本的内容和标签
print(train_X[0])       # 是一个包含784个元素且值在[0,1]之间的向量
print(train_Y[0])
# 可视化样本，下面是输出了训练集中前4个样本
fig, ax = plt.subplots(nrows=2, ncols=2, sharex='all', sharey='all')
ax = ax.flatten()
for i in range(4):
    img = train_X[i].reshape(28, 28)
    # ax[i].imshow(img,cmap='Greys')
    ax[i].imshow(img)
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()