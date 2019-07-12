from __future__ import division  # floating point division
import numpy as np
import math

import utilities as utils

class Regressor:
    """
    Generic regression interface; returns random regressor
    Random regressor randomly selects w from a Gaussian distribution
    """

    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}
        self.reset(parameters)

    def reset(self, parameters):
        """ Reset learner """
        self.weights = None
        self.resetparams(parameters)

    def resetparams(self, parameters):
        """ Can pass parameters to reset with new parameters """
        self.weights = None
        try:
            utils.update_dictionary_items(self.params,parameters)
        except AttributeError:
            # Variable self.params does not exist, so not updated
            # Create an empty set of params for future reference
            self.params = {}

    def getparams(self):
        return self.params

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.weights = np.random.rand(Xtrain.shape[1])

    def predict(self, Xtest):
        """ Most regressors return a dot product for the prediction """
        ytest = np.dot(Xtest, self.weights)
        return ytest

class RangePredictor(Regressor):
    """
    Random predictor randomly selects value between max and min in training set.
    """

    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.min = 0
        self.max = 1
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.min = np.amin(ytrain)
        self.max = np.amax(ytrain)

    def predict(self, Xtest):
        ytest = np.random.rand(Xtest.shape[0])*(self.max-self.min) + self.min
        return ytest

class MeanPredictor(Regressor):
    """
    Returns the average target value observed; a reasonable baseline
    """
    def __init__( self, parameters={} ):
        self.params = {}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.mean = None
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.mean = np.mean(ytrain)

    def predict(self, Xtest):
        return np.ones((Xtest.shape[0],))*self.mean


class FSLinearRegression(Regressor):
    """
    Linear Regression with feature selection, and ridge regularization
    """
    def __init__( self, parameters={} ):
        self.params = {'features': [1,2,3,4,5]}
        self.reset(parameters)

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Dividing by numsamples before adding ridge regularization
        # to make the regularization parameter not dependent on numsamples
        numsamples = Xtrain.shape[0]
        Xless = Xtrain[:,self.params['features']]
        self.weights = np.dot(np.dot(np.linalg.inv(np.dot(Xless.T,Xless)/numsamples), Xless.T),ytrain)/numsamples

    def predict(self, Xtest):
        Xless = Xtest[:,self.params['features']]
        ytest = np.dot(Xless, self.weights)
        return ytest

class RidgeLinearRegression(Regressor):
    """
    Linear Regression with ridge regularization (l2 regularization)
    TODO: currently not implemented, you must implement this method
    Stub is here to make this more clear
    Below you will also need to implement other classes for the other algorithms
    """
    def __init__( self, parameters={} ):
        # Default parameters, any of which can be overwritten by values passed to params
        self.params = {'regwgt': 0.5}
        self.reset(parameters)
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Dividing by numsamples before adding ridge regularization
        # to make the regularization parameter not dependent on numsamples
        numsamples = Xtrain.shape[0]
        Xless = Xtrain[:,:]
        X=np.dot(Xless.T,Xless)
        
        self.weights = np.dot(np.dot(np.linalg.inv(X/numsamples+self.params['regwgt']*np.identity(X.shape[0])), Xless.T),ytrain)/numsamples
    def predict(self, Xtest):
        Xless = Xtest[:,: ]
        ytest = np.dot(Xless, self.weights)
        return ytest
    
class Lasso(Regressor):
    def __init__( self, parameters={} ):
        self.params = {'regwgt': 0.3}
        self.reset(parameters)  
        
    def learn(self,Xtrain, ytrain):
        numsamples = Xtrain.shape[0]
        self.weights=np.zeros(Xtrain.shape[1])
        max_itr=100
        err=10000
        tolerance=0.001
        Xless = Xtrain[:,:]
        XX=np.dot(Xless.T,Xless)/numsamples
        Xy=np.dot(Xless.T,ytrain)/numsamples
        hita=1/(2*np.linalg.norm(np.dot(Xless.T,Xless)))
        cost=np.square(np.linalg.norm(np.dot(Xless,self.weights)-ytrain))+self.params['regwgt']*np.linalg.norm(self.weights,1)
        while(abs(cost-err)>tolerance and max_itr>0):
            err=cost
            w=self.weights-hita*np.dot(XX,self.weights)+hita*Xy
            for i in range(len(w)):
                prox=hita*self.params['regwgt']
                if w[i]>prox:
                    self.weights[i]=w[i]-prox
                elif w[i]<-prox:
                    self.weights[i]=w[i]+prox
                else:
                    self.weights[i]=0
            cost=np.square(np.linalg.norm(np.dot(Xless,self.weights)-ytrain))+self.params['regwgt']*np.linalg.norm(self.weights,1)
            max_itr-=1
            
                    
        
    def predict(self, Xtest):
        Xless = Xtest[:]
        ytest = np.dot(Xless, self.weights)
        return ytest
    
    
class SGD (Regressor):

    def __init__( self, parameters={} ):
        self.params = {'regwgt': 0.5}
        self.reset(parameters)  
        
    def learn(self,Xtrain, ytrain):
        numsamples = Xtrain.shape[0]
        #originally use np.zero which does match requirement in algorithm is text book
        self.weights=np.random.rand(Xtrain.shape[1])
        #modify from 1000 to 100, different from A2
        epochs=100
        hita=self.params['regwgt']
        for i in range(epochs):
            for j in range(numsamples):
                gradient=(np.dot(Xtrain[j],self.weights)-ytrain[j])*Xtrain[j]
                self.weights=self.weights-hita*gradient
                
    def predict(self, Xtest):
        Xless = Xtest[:]
        ytest = np.dot(Xless, self.weights)
        return ytest
    
class BGD(Regressor):
    def __init__( self, parameters={} ):
        self.params = {'regwgt': 0.5}
        self.reset(parameters)  
        
    def learn(self,Xtrain, ytrain):
        numsamples = Xtrain.shape[0]
        #originally use np.zero which does match requirement in algorithm is text book
        self.weights=np.random.rand(Xtrain.shape[1])
        max_itr=100
        err=10000
        tolerance=0.001
        Xless = Xtrain[:,:]
        cost=np.square(np.linalg.norm(np.dot(Xless,self.weights)-ytrain))/2*numsamples
        while(abs(cost-err)>tolerance and max_itr>0):
            err=cost
            gradient=np.dot(Xless.T,(np.dot(Xless,self.weights)-ytrain))/numsamples
            #find step-size by line-search
            hita=lineSearch (Xtrain, ytrain,self.weights,numsamples,self.params['regwgt'])
            self.weights=self.weights-hita*gradient
            cost=np.square(np.linalg.norm(np.dot(Xless,self.weights)-ytrain))/2*numsamples
            max_itr-=1
            
                    
        
    def predict(self, Xtest):
        Xless = Xtest[:]
        ytest = np.dot(Xless, self.weights)
        return ytest


def lineSearch (Xtrain, ytrain,weight,n,max_hita):
    i=0
    max_itr=100
    hita=max_hita
    scalar=0.7
    tolerance=0.001
    w=weight
    obj=np.square(np.linalg.norm(np.dot(Xtrain,weight)-ytrain))/2*n
    gradient=np.dot(Xtrain.T,(np.dot(Xtrain,weight)-ytrain))/n
    while(i<max_itr):
        w=weight-hita*gradient
        cost=np.square(np.linalg.norm(np.dot(Xtrain,w)-ytrain))/2*n
        if cost<(obj-tolerance):
            break
        else:
            hita=scalar*hita
            i+=1
    if i>=max_itr:
        return hita
    
    return hita