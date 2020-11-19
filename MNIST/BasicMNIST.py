import tensorflow as tf
import numpy as np
import math

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


x_train = x_train.reshape(60000, 28*28)

w = tf.random.uniform(shape=[28*28])
tf.reduce_sum(x_train[12]*w)+0.3

sum((tf.reduce_sum(x_train[i]*w)+0.3-y_train[i])**2 for i in range(60))

y_train_extended = np.zeros(shape=(60000, 10), dtype=np.float32)
y_test_extendend = np.zeros(shape=(10000, 10), dtype=np.float32)

for i in range(10):
    y_train_extended[np.where(y_train == i), i] = 1
    y_test_extendend[np.where(y_test == i), i] = 1

# (tf.reduce_sum(x_train[12]*w[i]+b[i])-y_train[12][i])**2


def softmax(L):
    sum_exp = sum(math.exp(elem) for elem in L)
    return [math.exp(elem)/sum_exp for elem in L]


print(softmax([-2, 1.4, 0.4, 0.2, -0.1, 0.9, 0.3, 0.6, 0.8, 0.7]))

# softmax(tf.reduce_sum(x_train[12]*w[i]+b[i] for i in range(10))-y_train[12])**2

# Using tensorflow for machine learning

x = tf.Variable(initial_value=1.3, dtype=tf.float64)
print(x*x)

with tf.GradientTape() as g:
    g.watch(x)
    f = x*tf.math.exp(-x*x)+1/4
    print(g.gradient(f, x))


a = tf.Variable(5)
# Variables in tensorflow point to tensor
print(a*a)
print(a.numpy())


