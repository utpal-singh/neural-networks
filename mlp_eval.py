# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 15:06:10 2020

@author: utpal
"""

import numpy as np
from random import random

class MLP(object):
      """
      A Simple Multilayer Perceptron class
      
      """
      
      def __init__(self, num_inputs=3, hidden_layers=[3,3], num_outputs=2):
            
            """
            Constructor for MLP. Takes a number of inputs, 
            number of hidden layers as list and num of outputs neurons
            
            Args:
                  num_inputs: number of input neurons
                  num_hidden: number of hidden neurons in a layer as a list
                  num_outputs: numner of output neurons
            
            """
            
            self.num_inputs = num_inputs
            self.hidden_layers = hidden_layers
            self.num_outputs = num_outputs
            
            layers = [num_inputs] + hidden_layers + [num_outputs]
            
            #initiate random weights
            
            weights = []
            for i in range(len(layers)-1):
                  w = np.random.rand(layers[i], layers[i+1])
                  weights.append(w)
                  
            self.weights = weights
            
            
            activations = []
            for i in range(len(layers)):
                  a = np.zeros(layers[i])
                  activations.append(a)
            self.activations = activations
            
            
            derivatives = []
            for i in range(len(layers)-1):
                  d = np.zeros((layers[i], layers[i+1]))
                  derivatives.append(d)
            self.derivatives = derivatives
            
                  
      def _sigmoid(self, x):
            return 1/(1+np.exp(-x))
                  
      def forward_propagate(self, inputs):
            
            """
            Computes forward propagation of layers based on input signals
            
            Args:
                  inputs (ndarray): (nd-signals)
                  
            Returns:
                  activations (ndarray): Ouput signals
            """
            
            
            activations = inputs
            self.activations[0] = inputs
            
            for i,w in enumerate(self.weights):
                  #calculate the net inputs for a given layer
                  
                  net_inputs = np.dot(activations, w)
                  
                  #calculate the activations
                  
                  activations = self._sigmoid(net_inputs)
                  self.activations[i+1] = activations
                  
            return activations
      
      
      def _sigmoid_derivative(self, x):
            return x * (1.0 - x)
      
      def back_propagate(self, error, verbose = False):
            
            #dE/dW_i = (y - a_[i+1])*(s'(h_[i+1]))*a_i
            #s'h_[i+1] = s(h_i+1)(1-s(h_[i+1]))
            #s_h[i+1] = a_[i+1]
            
            
            for i in reversed(range(len(self.derivatives))):
                  activations = self.activations[i+1]
                  
                  #delta = (y - a_[i+1])*(s'(h_[i+1]))
                  
                  delta = error * self._sigmoid_derivative(activations)
                  delta_reshaped = delta.reshape(delta.shape[0], -1).T
                  current_activations = self.activations[i]
                  current_activations_reshaped = current_activations.reshape(current_activations.shape[0], -1)
                  
                  #dE/dW_i = (y - a_[i+1])*(s'(h_[i+1]))*a_i
                  
                  self.derivatives[i] = np.dot(current_activations_reshaped, delta_reshaped)
                  
                  error = np.dot(delta, self.weights[i].T)
                  
                  if verbose:
                        print("Derivative for W{} is {}".format(i, self.derivatives[i]))
            return error
      
                  
      def gradient_descent(self, learning_rate):
            for i in range(len(self.weights)):
                  weights = self.weights[i]
#                  print("Original Weights W{} {}".format(i, weights))
                  derivatives = self.derivatives[i]
                  weights = weights + derivatives*learning_rate
#                  print('Updated Weights W{}{}'.format(i, weights))
                  
                  
      def _mse(self, target, output):
            return np.average((target - output)**2)
                  
                  
      def train(self, inputs, targets, epochs, learning_rate):
            for i in range(epochs):
                  sum_error = 0
                  for input, target in zip(inputs, targets):
                        output = self.forward_propagate(input)
                        error = target - output
                        self.back_propagate(error)
                        self.gradient_descent(learning_rate)
                        sum_error = sum_error + self._mse(target, output)
                  print("Error: {} at epoch: {}".format(sum_error / len(inputs), i))
      
#Run an example model
if __name__ == "__main__":
#      mlp = MLP(2, [5,5,5,5,5], 1)
#      
#      #Create some inputs
#      input = np.array([0.1, 0.2])
#      target = np.array([0.3])
#      
#      output = mlp.forward_propagate(input)
#      error = target - output
#      
#      mlp.back_propagate(error, verbose=True)
#      
#      mlp.gradient_descent(0.1)
      
#      inputs = np.array([[random() / 2 for _ in range(2)] for _ in range(1000)])
      inputs = np.random.rand(1000, 2)
      targets = np.array([[i[0] + i[1]] for i in inputs])
      mlp = MLP(2, [5], 1)
      
      
      mlp.train(inputs, targets, 50, 0.1)
      
      input = np.array([0.1, 0.2])
      target = np.array([0.3])
      
      output = mlp.forward_propagate(input)
      
      print(input[0], input[1], output[0])
      
