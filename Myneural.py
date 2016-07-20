# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 20:14:12 2016

@author: chen

This is a python3 neural network chagned from the- 
online book <Neural networks and deep learning>
http://neuralnetworksanddeeplearning.com/
"""

#myneural
import numpy as np

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))
def sigmoid_prime(z):
    return np.multiply(sigmoid(z),1-sigmoid(z))
    
class Network(object):
    def __init__(self,sizes):
        self.size=sizes
        self.num_layer=len(sizes)
        self.biases=[np.random.randn(y,1) for y in sizes[1:]]
        self.weights=[np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]
        
    def feedforward(self,a):
        a=np.matrix(a).reshape(-1,1)
        for b,w in zip(self.biases,self.weights):
            z=np.dot(w,a)+b
            a=sigmoid(z)
        return a
            
    def update_mini_batch(self,mini_batch,eta):
        nabla_b=[np.zeros(b.shape) for b in self.biases]
        nabla_w=[np.zeros(weight.shape) for weight in self.weights]
        for x,y in mini_batch:
            delta_nabla_b,delta_nabla_w = self.backprop(x,y)
            nabla_b=[nb+dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]
            nabla_w=[nw+dnw for nw,dnw in zip(nabla_w,delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw for w,nw in zip(self.weights,nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b,nb in zip(self.biases,nabla_b)]   
        
    def evaluate(self,test_data):
        test_results=[(np.argmax(self.feedforward(x)),np.argmax(y)) for (x,y) in test_data]
        re=[int(x==y) for (x,y) in test_results]
        return sum(re)
        
    def cost_derivate(self,output_activations,y):
        y=np.matrix(y).reshape(-1,1)
        return (output_activations-y)
            
    def backprop(self,x,y):
        nabla_b=[np.zeros(b.shape) for b in self.biases]
        nabla_w=[np.zeros(w.shape) for w in self.weights]
        
        activation=x
        activations=[x]
        zs=[]
        
        for b,w in zip(self.biases,self.weights):
            activation=np.matrix(activation).reshape(-1,1)
            z=np.dot(w,activation)+b
            zs.append(z)
            activation=sigmoid(z)
            activations.append(activation)
        delta = np.array(self.cost_derivate(activations[-1],y))*np.array(sigmoid_prime(zs[-1]))
        nabla_b[-1]=delta
        nabla_w[-1]=np.dot(delta,activations[-2].transpose())
        
        for l in range(2,self.num_layer):
            z=zs[-l]
            sp=sigmoid_prime(z)
            delta=np.array(np.dot(self.weights[-l+1].transpose(),delta))*np.array(sp)
            nabla_b[-l]=delta
            nabla_w[-l]=np.dot(delta,np.matrix(activations[-l-1]).reshape(1,-1))
        return (nabla_b,nabla_w)
            
    def SGD(self,train_data,eta,epochs,mini_batch_size,test_data=None):
        
        n_test=len(test_data)
        n=len(train_data)
        for j in range(epochs):
            np.random.shuffle(train_data)
            mini_batches=[train_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,eta)
            if n_test:
                print("Epoch {0}: {1} / {2}".format(j,self.evaluate(test_data),n_test))
            else:
                print("Epoch {0} complete".format(j))      
        
