# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 10:34:44 2016

@author: chen
"""
import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf
import Myneural
import numpy as np
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

train_x=mnist.train.images
train_y=mnist.train.labels
test_x=mnist.test.images
test_y=mnist.test.labels
train_data=np.array([[x,y]for x,y in zip(train_x,train_y)])
test_data=np.array([[x,y]for x,y in zip(test_x,test_y)])
net=Myneural.Network([784,30,10])
net.SGD(train_data,10.0,100,20,test_data)


