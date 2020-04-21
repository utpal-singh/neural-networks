# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 04:00:03 2020

@author: utpal
"""

import math

def sigmoid(z):
      y = 1.0/(1.0 + math.exp(-z))
      return y

def activate(inputs, weights):
      a = 0
      for x,w in zip(inputs, weights):
            a = a + x*w
      return sigmoid(a)
      
if __name__ == "__main__":
      inputs = [0.5, 0.3, 0.2]
      weights = [0.4, 0.7, 0.2]
      
      output = activate(inputs, weights)
      print (output)