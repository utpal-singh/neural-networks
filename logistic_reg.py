# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 16:54:18 2020

@author: utpal
"""
import numpy as np

class LogisticReg():
      """
      A Logistic Regression Class
      
      """
      
      def __init__(self, x, y, alp):
            """
            Args:
                  m_test = Inputs a number of training examples
                  No of feature of a training example is currently set to 2
                  x: A feature matrix
                  alp: learning rate
                  y: a row matrix
            
            """
            
            self.alp = alp
            self.y = y
            
      def _sigmoid(self, x):
            return 1/(1+np.exp(-x))
      
      def GradDescent(self):
            J = 0
            dw_1 = 0
            dw_2 = 0
            db = 0
            b = np.random.randint(low = 0, high = 1)
            z = np.zeros(x.shape[0])
            
            m = x.shape[1]
            n = x.shape[0]
            
            
            for i in range(m):
                  w = np.random.randn(n, 1)
                  dz = np.random.randn(n, 1)
                  dw = np.random.randn(m, 1)
                  
                  
                  a = np.random.randn(m,1)
                  
                  z[i] = np.dot(w.T, x[:, [i]]) + b
#                        z[i] = w*x[:, [i]]
                  a[i] = self._sigmoid(z[i])
                  J += -(y[i]*(np.log(a[i]) + (1-y[i])*(np.log(1-a[i]))))
                  dz[i] = a[i] - y[i]
                  
                  for j in range(n):
                        dw[j] += x[j][i]*dz[i]
                  db += dz[i]
                  
            J = J/m
            print(J)
            
            
if __name__ == '__main__':
      x = np.array([[1,2,3,4],[2,3,4,5],[3,4,5,6]])
      y = np.array([4,5,6,7])
      alp = 2
      logreg = LogisticReg(x,y,alp)
      logreg.GradDescent()
      