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
      
      def __init__(m_test, alp):
            """
            Args:
                  m_test = Inputs a number of training examples
                  No of feature of a training example is currently set to 2
                  x: A feature matrix
                  alp: learning rate
            
            """
            
            self.m_test = m_test
            self.alp = alp
            
      def derivatives(self):
            J = 0, dw_1 = 0, dw_2 = 0, db = 0
            b = np.random.randint(low = 0, high = 1)
            w = np.random.randn(2, 1)
            
            for i in range(self.m):
                  z_sup_i = np.dot(w, x_sup_i) + b
                  a_sup_i = _sigmoid(z_sup_i)
                  J += -(y_sup_i*(np.log(a_sup_i) + (1-y_sup_i)*(np.log(1-a_sup_i))))
                  dz_sup_i = a_sup_i - y_sup_i
                  dw_1 += x_1_sup_i*dz_sup_i
                  dw_2 += x_1_sup_i*dz_sup_i
                  db += dz_sup_i
                  
            J = J/m
            dw_1 = dw_1/m
            dw_2 = dw_2/m
            db = db/m
            
            w_1 = w_1 - alp*dw_1
            w_2 = w_2 - alp*dw_2
            b = b - alp*db