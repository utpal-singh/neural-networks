# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 15:06:10 2020

@author: utpal
"""

import numpy as np

#save activations and derivatives
#implement backpropagation
#implement gradient descent
#implement train
#train our net with some dummy dataset
#make some predictions

class MLP(object):
      """
      A Multilayer Perceptron class
      
      """
      
      def __init__(self, num_inputs=3, num_hidden=[3,5], num_outputs=2):
            
            """
            Constructor for MLP. Takes a number of inputs, 
            number of hidden layers as list and num of outputs neurons
            
            Args:
                  num_inputs: number of input neurons
                  num_hidden: number of hidden neurons in a layer as a list
                  num_outputs: numner of output neurons
            
            """
            
            self.num_inputs = num_inputs
            self.num_hidden = num_hidden
            self.num_outputs = num_outputs
            
            layers = [self.num_inputs] + self.num_hidden + [self.num_outputs]
            
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
      
      def back_propagate(self, error):
            
            #dE/dW_i = (y - a_[i+1])*(s'(h_[i+1]))*a_i
            #s'h_[i+1] = s(h_i+1)(1-s(h_[i+1]))
            #s_h[i+1] = a_[i+1]
            
            
            for i in reversed(range(len(self.derivatives))):
                  activations = self.activations[i+1]
                  
                  
      
      
if __name__ == "__main__":
      mlp = MLP()
      
      #Create some inputs
      inputs = np.random.rand(mlp.num_inputs)
      
      #Create forward propagations
      outputs = mlp.forward_propagate(inputs)
      
      #Print the results
      print("The network input is: {}".format(inputs))
      print("The nework output is: {}".format(outputs))
      