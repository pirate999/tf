#coding=utf-8

import tensorflow as tf
import input_data

#load data
minist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#define model
x = tf.placeholder("float", [None, 784])
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, w) + b)

#real label
y_ = tf.placeholder("float", [None, 10])

#define model
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.global_variables_initializer()

print("y:",y_)

with tf.Session() as sess:
    #init
    sess.run(init)
    
    #train
    for i in range(1000):
        #raw data and label
        batch_xs, batch_ys = minist.train.next_batch(100)
        y_tmp = sess.run([train_step, y], feed_dict={x: batch_xs, y_: batch_ys})
        print(list(y_tmp[1]))
       
    
    #tf.argmax(A, 1)
    #if A is a vector return the index of max value in A
    #if A is a matrix, then return an vector of every row's max value in A
    #tf.argmax(y,1) 按行找出概率最大项的索引   
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
    
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    
    ll = sess.run([accuracy, correct_prediction], feed_dict={x: minist.test.images, y_: minist.test.labels})
    print(ll[1].shape)
    










