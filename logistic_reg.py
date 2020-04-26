# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 20:00:36 2020

@author: utpal
"""

import numpy as np

class logistic_reg():
      """
      
      A Logistic Regression Class
      
      """
      
      def __init__(self, x, y):
            self.x = x
            self.y = y
            
            
      def Grad_des(self):
            
            m = x.shape[1]
            n = x.shape[0]
            
            J = 0
            db = 0
            
            
            z = np.zeros(m)
            a = np.zeros(m)
            dz = np.zeros(m)
            dw = np.zeros((n,1))
            
            for i in range(m):
                  w = np.random.randn(n,1)
                  z[i] = np.dot(w.T, x[:, [i]])
                  a[i] = self._sigmoid(z[i])
                  
                  J+= -(y[i]*np.log(a[i]) + (1-y[i])*np.log(1-a[i]))
                  
                  dz[i] = a[i] - y[i]
                  
                  db += db + dz[i]
                  
                  for j in range(n):
                        dw[j] += np.dot(x[j], dz)
                  
      def _sigmoid(self, z):
            return 1/(1+np.exp(-z))
                  
if __name__ == "__main__":
      x = np.array([[.1,.2,.3,.4], [.5,.6,.7,.8], [.9,.10,.11,.12]])
      y = np.array([.13, .14, .15, .16])
      logre = logistic_reg(x,y)
      logre.Grad_des()
                  
                  
      