#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np

class LogisticRegression:

    def __init__(self, alpha = 0.01, regLambda=0.01, epsilon=0.0001, maxNumIters = 50):
        '''
        Initializes Parameters of the  Logistic Regression Model
        '''
        self.alpha = alpha
        self.regLambda = regLambda
        self.epsilon = epsilon
        self.maxNumIters = maxNumIters
  
    

    
    
    def calculateGradient(self, weight, X, Y, regLambda):
        '''
        Computes the gradient of the objective function
        Arguments:
        
            X is a n-by-(d+1) numpy matrix
            Y is an n-by-1 dimensional numpy matrix
            weight is (d+1)-by-1 dimensional numpy matrix
            regLambda is the scalar regularization constant
        Returns:
            the gradient, an (d+1)-by-1 dimensional numpy matrix
        '''
        
        Gradient = np.zeros((X.shape[1],1))
        Z = np.dot(X,weight)
        diff = np.subtract(self.sigmoid(Z), Y)
        for counter in range(weight.shape[0]):           
            Gradient[counter,0] = np.dot(X[:,counter],diff) + regLambda*weight[counter,0]
            
        Gradient[0,0] -= regLambda*weight[0,0]
        
        return Gradient    

    def sigmoid(self, Z):
        '''
        Computes the Sigmoid Function  
        Arguments:
            A n-by-1 dimensional numpy matrix
        Returns:
            A n-by-1 dimensional numpy matrix
       
        '''
        
        return 1/(1+np.exp(-Z))

    def update_weight(self,X,Y,weight):
        '''
        Updates the weight vector.
        Arguments:
            X is a n-by-(d+1) numpy matrix
            Y is an n-by-1 dimensional numpy matrix
            weight is a d+1-by-1 dimensional numpy matrix
        Returns:
            updated weight vector : (d+1)-by-1 dimensional numpy matrix
        '''
        grad = self.calculateGradient(weight, X, Y, self.regLambda)
        diff = (self.alpha*grad)
        return weight - diff       
        
        
    
    def check_conv(self,weight,new_weight,epsilon):
        '''
        Convergence Based on Tolerance Values
        Arguments:
            weight is a (d+1)-by-1 dimensional numpy matrix
            new_weights is a (d+1)-by-1 dimensional numpy matrix
            epsilon is the Tolerance value we check against
        Return : 
            True if the weights have converged, otherwise False

        '''
        norm = np.sqrt(np.power((new_weight - weight),2).sum())
        if (norm< epsilon):
            return True
        else:
            return False
        
    def train(self,X,Y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy matrix
            Y is an n-by-1 dimensional numpy matrix
        Return:
            Updated Weights Vector: (d+1)-by-1 dimensional numpy matrix
        '''
        # Read Data
        n,d = X.shape
        
        #Add 1's column
        X = np.c_[np.ones((n,1)), X]
        self.weight = self.new_weight = np.zeros((d+1,1))
        
        for n in range(self.maxNumIters):
            self.new_weight = self.update_weight(X,Y,self.weight)
            if (self.check_conv(self.weight,self.new_weight,self.epsilon)):
                break
            self.weight = self.new_weight
                      
        return self.weight
        

    def predict_label(self, X,weight):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy matrix
            weight is a d+1-by-1 dimensional matrix
        Returns:
            an n-by-1 dimensional matrix of the predictions 0 or 1
        '''
        #data
        n=X.shape[0]
        #Add 1's column
        X = np.c_[np.ones((n,1)), X]
        result = np.zeros([X.shape[0],1])
        counter = 0
        for m in X[:,0]:
            preexp = np.dot(X[counter,:],weight)
            
            if (np.exp(preexp) >= 1):
                result[counter,0] = 1
            else:
                result[counter,0] = 0
            counter += 1
        
        return result
    
    def calculateAccuracy (self, Y_predict, Y_test):
        '''
        Computes the Accuracy of the model
        Arguments:
            Y_predict is a n-by-1 dimensional matrix (Predicted Labels)
            Y_test is a n-by-1 dimensional matrix (True Labels )
        Returns:
            Scalar value for accuracy in the range of 0 - 100 %
        '''
        counter = 0
        counterF = 0
        for m in Y_test[:,0]:
	    if (m == Y_predict[counter,0]):
	        counterF += 1
	    counter += 1
	Accuracy = float(counterF)/Y_test.shape[0] * 100.0
        return Accuracy
    
        
