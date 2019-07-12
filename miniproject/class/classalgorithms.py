from __future__ import division  # floating point division
import numpy as np
import utilities as utils

#from scipy.stats import norm
import math
class Classifier:
    """
    Generic classifier interface; returns random classification
    Assumes y in {0,1}, rather than {-1, 1}
    """

    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}

    def reset(self, parameters):
        """ Reset learner """
        self.resetparams(parameters)

    def resetparams(self, parameters):
        """ Can pass parameters to reset with new parameters """
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

    def predict(self, Xtest):
        probs = np.random.rand(Xtest.shape[0])
        ytest = utils.threshold_probs(probs)
        return ytest

class LinearRegressionClass(Classifier):
    """
    Linear Regression with ridge regularization
    Simply solves (X.T X/t + lambda eye)^{-1} X.T y/t
    """
    def __init__( self, parameters={} ):
        self.params = {'regwgt': 0.01}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Ensure ytrain is {-1,1}
        yt = np.copy(ytrain)
        yt[yt == 0] = -1

        # Dividing by numsamples before adding ridge regularization
        # for additional stability; this also makes the
        # regularization parameter not dependent on numsamples
        # if want regularization disappear with more samples, must pass
        # such a regularization parameter lambda/t
        numsamples = Xtrain.shape[0]
        self.weights = np.dot(np.dot(np.linalg.pinv(np.add(np.dot(Xtrain.T,Xtrain)/numsamples,self.params['regwgt']*np.identity(Xtrain.shape[1]))), Xtrain.T),yt)/numsamples

    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        ytest[ytest > 0] = 1
        ytest[ytest < 0] = 0
        return ytest

class NaiveBayes(Classifier):
    """ Gaussian naive Bayes;  """

    def __init__(self, parameters={}):
        """ Params can contain any useful parameters for the algorithm """
        # Assumes that a bias unit has been added to feature vector as the last feature
        # If usecolumnones is False, it should ignore this last feature
        self.params = {'usecolumnones': True}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.means = []
        self.stds = []
        self.numfeatures = 0
        self.numclasses = 0
        

    def learn(self, Xtrain, ytrain):
        """
        In the first code block, you should set self.numclasses and
        self.numfeatures correctly based on the inputs and the given parameters
        (use the column of ones or not).

        In the second code block, you should compute the parameters for each
        feature. In this case, they're mean and std for Gaussian distribution.
        """

        ### YOUR CODE HERE
        
        class_set=set(ytrain)
        self.numclasses = len(class_set)
        #if self.params['usecolumnones']==False:
        self.numfeatures = Xtrain.shape[1]-1
        Xtrain=Xtrain[:,:self.numfeatures]
        #else:
        #   self.numfeatures = Xtrain.shape[1]

        ### END YOUR CODE

        origin_shape = (self.numclasses, self.numfeatures)
        self.means = np.zeros(origin_shape)
        self.stds = np.zeros(origin_shape)

        ### YOUR CODE HERE
        "create a list to contain prior"
        self.p_y=np.zeros(self.numclasses)
        self.class_list=[]

        "for each class of y, calculate the corresponding mean and var with respect to every feature "
        cla_num=0
        unique, counts = np.unique(ytrain, return_counts=True)
        dictionary = dict(zip(unique, counts))
        for cla in  unique:
            for j in range(self.numfeatures):
                "calculate the sum of all features in feature j whose class is cla"
                sum=0
                
                for i in range(Xtrain.shape[0]):
                    if ytrain[i]==cla:
                        sum+=Xtrain[i][j]
                        
                "calculate the mean"
                self.means[cla_num][j]=sum/dictionary[cla]
                
                square_error=0
                for i in range(Xtrain.shape[0]):
                    if ytrain[i]==cla:
                        square_error+=(Xtrain[i][j]-self.means[cla_num][j])**2
                "calculate the variance"
                self.stds[cla_num][j]=square_error/dictionary[cla]
            
            self.p_y[cla_num]=dictionary[cla]/len(ytrain)
            self.class_list.append(cla)
            "update the class num which is the index specifying different class"
            cla_num+=1
        ### END YOUR CODE
        
        assert self.means.shape == origin_shape
        assert self.stds.shape == origin_shape

    def predict(self, Xtest):
        """
        Use the parameters computed in self.learn to give predictions on new
        observations.
        """
        ytest = np.zeros(Xtest.shape[0], dtype=int)
        
        ### YOUR CODE HERE
        #if self.params['usecolumnones']==False:
        Xtest=Xtest[:,:self.numfeatures]

        
        "for every row in test matrix"
        for i in range(Xtest.shape[0]):
            "find most likely class with highest probability, among all available classes  "
            max_pos=0
            max_cla=0
            for cla_num in range(len(self.p_y)):
                
                likelyhood=1
                for j in range (len(Xtest[i])):
                    exp=math.exp(-(Xtest[i][j]-self.means[cla_num][j])**2/(2*self.stds[cla_num][j]))
                    norm_pdf=exp/math.sqrt(2*math.pi*self.stds[cla_num][j]) 
                    likelyhood*=norm_pdf
                posterior=likelyhood*self.p_y[cla_num]
                if posterior>max_pos:
                    max_pos=posterior
                    max_cla=self.class_list[cla_num]
            ytest[i]=max_cla

        ### END YOUR CODE

        assert len(ytest) == Xtest.shape[0]
        return ytest

class LogitReg(Classifier):

    def __init__(self, parameters={}):
        # Default: no regularization
        self.params = {'regwgt': 0.0, 'regularizer': 'None'}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None
        if self.params['regularizer'] is 'l2':
            self.regularizer = (utils.l2, utils.dl2)
        else:
            self.regularizer = (lambda w: 0, lambda w: np.zeros(w.shape,))

    def logit_cost(self, theta, X, y):
        """
        Compute cost for logistic regression using theta as the parameters.
        """

        cost = 0.0

        ### YOUR CODE HERE
        for i in range(X.shape[0]):
            "sigmoid is the transfer function (and also E(y|X)) of logit regress"
            sigmoid=1/(1+np.exp(-np.dot(theta,X[i])))
            cost+=-(y[i]*np.log(sigmoid)+(1-y[i])*np.log(1-sigmoid))
        
        ### END YOUR CODE

        return cost

    def logit_cost_grad(self, theta, X, y):
        """
        Compute gradients of the cost with respect to theta.
        """

        grad = np.zeros(len(theta))

        ### YOUR CODE HERE
        sigmoid=1/(1+np.exp(-np.dot(theta,X)))
        grad = (sigmoid-y)*X

        ### END YOUR CODE

        return grad

    def learn(self, Xtrain, ytrain):
        """
        Learn the weights using the training data
        """

        self.weights = np.zeros(Xtrain.shape[1],)
       
        ### YOUR CODE HERE
        stepSize=0.01
        epochs=10
        for k in range(epochs):
            for i in range(Xtrain.shape[0]):
                sigmoid=1/(1+np.exp(-np.dot(self.weights,Xtrain[i])))
                self.weights-=stepSize*(sigmoid-ytrain[i])*Xtrain[i]
        ### END YOUR CODE

    def predict(self, Xtest):
        """
        Use the parameters computed in self.learn to give predictions on new
        observations.
        """
        ytest = np.zeros(Xtest.shape[0], dtype=int)

        ### YOUR CODE HERE
        ytest = np.dot(Xtest, self.weights)
        for i in range(ytest.shape[0]):
            posterior=1/(1+np.exp(-ytest[i]))
            if  posterior >=0.5:
                ytest[i]=1
            else:
                ytest[i]=0
        ### END YOUR CODE

        assert len(ytest) == Xtest.shape[0]
        return ytest

class NeuralNet(Classifier):
    """ Implement a neural network with a single hidden layer. Cross entropy is
    used as the cost function.

    Parameters:
    nh -- number of hidden units
    transfer -- transfer function, in this case, sigmoid
    stepsize -- stepsize for gradient descent
    epochs -- learning epochs

    Note:
    1) feedforword will be useful! Make sure it can run properly.
    2) Implement the back-propagation algorithm with one layer in ``backprop`` without
    any other technique or trick or regularization. However, you can implement
    whatever you want outside ``backprob``.
    3) Set the best params you find as the default params. The performance with
    the default params will affect the points you get.
    """
    def __init__(self, parameters={}):
        self.params = {'nh': 16,
                    'transfer': 'sigmoid',
                    'stepsize': 0.1,
                    'epochs': 10}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        if self.params['transfer'] is 'sigmoid':
            self.transfer = utils.sigmoid
            self.dtransfer = utils.dsigmoid
        else:
            # For now, only allowing sigmoid transfer
            raise Exception('NeuralNet -> can only handle sigmoid transfer, must set option transfer to string sigmoid')
        self.w_input = None
        self.w_output = None

    def feedforward(self, inputs):
        """
        Returns the output of the current neural network for the given input
        """
        # hidden activations
        a_hidden = self.transfer(np.dot(self.w_input, inputs))

        # output activations
        a_output = self.transfer(np.dot(self.w_output, a_hidden))

        return (a_hidden, a_output)

    def backprop(self, x, y):
        """
        Return a tuple ``(nabla_input, nabla_output)`` representing the gradients
        for the cost function with respect to self.w_input and self.w_output.
        """

        ### YOUR CODE HERE
        h,y_hat = self.feedforward(x)
        
        nabla_output=np.outer((y_hat[0]-y),h)
        
        pre=np.dot(nabla_output,(y_hat[0]-y))
        med=np.multiply(pre[0],h)
        GD_W2=np.multiply(med,1-h)
                
                
        nabla_input=np.outer(GD_W2,x)
        ### END YOUR CODE

        assert nabla_input.shape == self.w_input.shape
        assert nabla_output.shape == self.w_output.shape
        return (nabla_input, nabla_output)

    # TODO: implement learn and predict functions
    def learn(self, Xtrain, ytrain):
        self.w_input = np.random.random((self.params['nh'],Xtrain.shape[1]))
        
        self.w_output=np.random.random((1,self.params['nh']))
        
        for i in range(self.params['epochs']):
            stepSize=self.params['stepsize']
            for j in range(Xtrain.shape[0]):
                h=self.transfer(np.dot(self.w_input,Xtrain[j]))
                
                #xvec=np.dot(h,self.w_output)
                #y_hat=1.0 / (1.0 + np.exp(np.negative(xvec)))
                y_hat=self.transfer(np.dot(self.w_output,h))
                
                GD_W1=np.outer((y_hat[0]-ytrain[j]),h)
                self.w_output-=stepSize*GD_W1
                
                
                
                pre=np.dot(self.w_output,(y_hat[0]-ytrain[j]))
               
                med=np.multiply(pre[0],h)
                GD_W2=np.multiply(med,1-h)
                
                
                self.w_input-=stepSize*np.outer(GD_W2,Xtrain[j])
    
    def predict(self, Xtest):

        ytest = np.zeros(Xtest.shape[0])

        ### YOUR CODE HERE


        for i in range(Xtest.shape[0]):
            #h=1/(1+np.exp(-np.dot(self.w_input,Xtest[i])))
            #y_hat=1/(1+np.exp(-np.dot(self.w_output,h)))
            h=self.transfer(np.dot(self.w_input,Xtest[i]))
            
            y_hat=self.transfer(np.dot(self.w_output,h))

            if  y_hat[0] >=0.5:
               
                ytest[i]=1
            else:
                ytest[i]=0
        ### END YOUR CODE

        assert len(ytest) == Xtest.shape[0]
        return ytest


class KernelLogitReg(LogitReg):
    """ Implement kernel logistic regression.

    This class should be quite similar to class LogitReg except one more parameter
    'kernel'. You should use this parameter to decide which kernel to use (None,
    linear or hamming).

    Note:
    1) Please use 'linear' and 'hamming' as the input of the paramteter
    'kernel'. For example, you can create a logistic regression classifier with
    linear kerenl with "KernelLogitReg({'kernel': 'linear'})".
    2) Please don't introduce any randomness when computing the kernel representation.
    """
    def __init__(self, parameters={}):
        # Default: no regularization
        self.params = {'regwgt': 0.0, 'regularizer': 'None', 'kernel': 'None'}
        self.reset(parameters)

    def learn(self, Xtrain, ytrain):
        """
        Learn the weights using the training data.

        Ktrain the is the kernel representation of the Xtrain.
        """
        Ktrain = None
        
        ### YOUR CODE HERE
        count=int(Xtrain.shape[0])
        self.centerList=[]
        while count>=1:
            index=count-1
            self.centerList.append(Xtrain[index])
            count=int(count/2)
         

        
        numOfCenter=len(self.centerList)
        #initialize Ktrain as n by k zero matrix 
        Ktrain = np.zeros((Xtrain.shape[0],numOfCenter))
        for i in range (Ktrain.shape[0]):
            for j in range(Ktrain.shape[1]):
                Ktrain[i][j]=np.dot(Xtrain[i],self.centerList[j])
        

        ### END YOUR CODE

        self.weights = np.zeros(Ktrain.shape[1],)
            

        ### YOUR CODE HERE
        stepSize=0.01
        epochs=10
        
        for k in range(epochs):
            for i in range(Ktrain.shape[0]):
                sigmoid=1/(1+np.exp(-np.dot(self.weights,Ktrain[i])))
                self.weights-=stepSize*(sigmoid-ytrain[i])*Ktrain[i]

        ### END YOUR CODE

        self.transformed = Ktrain # Don't delete this line. It's for evaluation.

    # TODO: implement necessary functions
    def predict(self, Xtest):
        """
        Use the parameters computed in self.learn to give predictions on new
        observations.
        """
        Ktest=np.zeros((Xtest.shape[0],len(self.centerList)))
        for i in range(Xtest.shape[0]):
            for j in range(len(self.centerList)):
                Ktest[i][j]=np.dot(Xtest[i],self.centerList[j])

        
        ytest = np.dot(Ktest, self.weights)
        
        for i in range(ytest.shape[0]):
            posterior=1/(1+np.exp(-ytest[i]))
            if  posterior >=0.5:
                ytest[i]=1
                
            else:
                ytest[i]=0
                

        assert len(ytest) == Xtest.shape[0]
        return ytest


# ======================================================================

def test_lr():
    print("Basic test for logistic regression...")
    clf = LogitReg()
    theta = np.array([0.])
    X = np.array([[1.]])
    y = np.array([0])

    try:
        cost = clf.logit_cost(theta, X, y)
    except:
        raise AssertionError("Incorrect input format for logit_cost!")
    assert isinstance(cost, float), "logit_cost should return a float!"

    try:
        grad = clf.logit_cost_grad(theta, X, y)
    except:
        raise AssertionError("Incorrect input format for logit_cost_grad!")
    assert isinstance(grad, np.ndarray), "logit_cost_grad should return a numpy array!"

    print("Test passed!")
    print("-" * 50)

def test_nn():
    print("Basic test for neural network...")
    clf = NeuralNet()
    X = np.array([[1., 2.], [2., 1.]])
    y = np.array([0, 1])
    clf.learn(X, y)

    assert isinstance(clf.w_input, np.ndarray), "w_input should be a numpy array!"
    assert isinstance(clf.w_output, np.ndarray), "w_output should be a numpy array!"

    try:
        res = clf.feedforward(X[0, :])
    except:
        raise AssertionError("feedforward doesn't work!")

    try:
        res = clf.backprop(X[0, :], y[0])
    except:
        raise AssertionError("backprob doesn't work!")

    print("Test passed!")
    print("-" * 50)

def main():
    test_lr()
    test_nn()

if __name__ == "__main__":
    main()
