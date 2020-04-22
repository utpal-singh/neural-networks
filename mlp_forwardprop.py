# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 05:13:03 2020

@author: utpal
"""

import numpy as np

class multiANN():
      
      def __init__(self, num_inputs=3, num_hidden=[3,5], num_outputs=2):
            
            self.num_inputs = num_inputs
            self.num_hidden = num_hidden
            self.num_outputs = num_outputs
            
            layers = [self.num_inputs] + self.num_hidden + [self.num_outputs]
            
            #initiate random weights
            
            self.weights = []
            for i in range(len(layers)-1):
                  w = np.random.rand(layers[i], layers[i+1])
                  self.weights.append(w)
                  
      def _sigmoid(self, x):
            return 1/(1+np.exp(-x))
                  
      def forward_propagate(self, inputs):
            activations = inputs
            
            for w in self.weights:
                  #calculate the net inputs for a given layer
                  
                  net_inputs = np.dot(activations, w)
                  
                  #calculate the activations
                  
                  activations = self._sigmoid(net_inputs)
                  
            return activations
      
      
if __name__ == "__main__":
      
      #Create an MLP
      mlp = multiANN()
      
      #Create some inputs
      inputs = np.random.rand(mlp.num_inputs)
      
      #Create forward propagations
      outputs = mlp.forward_propagate(inputs)
      
      #Print the results
      print("The network input is: {}".format(inputs))
      print("The nework output is: {}".format(outputs))
            
            