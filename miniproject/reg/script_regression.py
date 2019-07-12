from __future__ import division  # floating point division
import csv
import random
import math
import numpy as np

import dataloader as dtl
import regressionalgorithms as algs

def l2err(prediction,ytest):
    """ l2 error (i.e., root-mean-squared-error) """
    return np.linalg.norm(np.subtract(prediction,ytest))

def l1err(prediction,ytest):
    """ l1 error """
    return np.linalg.norm(np.subtract(prediction,ytest),ord=1)

def l2err_squared(prediction,ytest):
    """ l2 error squared """
    return np.square(np.linalg.norm(np.subtract(prediction,ytest)))

def geterror(predictions, ytest):
    # Can change this to other error values
    return l2err(predictions,ytest)/ytest.shape[0]


if __name__ == '__main__':
    trainsize = 1000
    testsize = 2000
    numruns = 1
    #set name of file to be scan
    filename = 'datasets/Financial_Distress.csv'
    numOfFold=5

    regressionalgs = {'Random': algs.Regressor(),
                'Mean': algs.MeanPredictor(),
                #'FSLinearRegression5': algs.FSLinearRegression({'features': [1,2,3,4,5]}),
                #'FSLinearRegression50': algs.FSLinearRegression({'features': range(50)}),
                'RidgeLinearRegression': algs.RidgeLinearRegression(),
                #'Lasso':algs.Lasso(),
                'Stochastic Gradient Descent':algs.SGD(),
                'Batch Gradient Descent':algs.BGD(),
             }
    numalgs = len(regressionalgs)

    # Enable the best parameter to be selected, to enable comparison
    # between algorithms with their best parameter settings
    parameters = (
        {'regwgt': 0.1},
        {'regwgt': 0.01},
        {'regwgt': 1.0},
                      )
    numparams = len(parameters)
    
    errors = {}
    for learnername in regressionalgs:
        errors[learnername] = np.zeros((numparams,numOfFold))

    overall_R_squared={}
    for learnername in regressionalgs:
         overall_R_squared[learnername] = np.zeros((numparams,numOfFold))

    
    for r in range(numruns):
        dataset = dtl.loadcsv(filename)
        k_fold_indices=dtl.k_fold_cv(numOfFold,dataset)

        fold_count=0
        #trainset, testset = dtl.load_ctscan(trainsize,testsize)
        for train_index, test_index in k_fold_indices:
            trainset, testset = dtl.Normalize(dataset,train_index,test_index)
            #delete the lable on top of the table
            trainset=(trainset[0][1:],trainset[1][1:])
            testset=(testset[0][1:],testset[1][1:])
            print(('Running on train={0} and test={1} samples for run {2}').format(trainset[0].shape[0], testset[0].shape[0],r))
            
            
            #In each round of internal cross validation, train  one parameter in parameter set
           
            for p in range(numparams):
                
                # "iterate over different parameters
                params = parameters[p]
               
                #counter for iteration of internal cv in one parameter
                count_iter=0
                # tune hyperparameter by internal cross validation
                internal_cv_indices=dtl.k_fold_cv(numOfFold,trainset[0])
                
                for sub_train_index, sub_test_index in internal_cv_indices:
                    
                    
                    sub_trainset=(trainset[0][sub_train_index],trainset[1][sub_train_index])
                   

                    sub_testset=(trainset[0][sub_test_index],trainset[1][sub_test_index])

                    for learnername, learner in regressionalgs.items():
                        # Reset learner for new parameters
                        learner.reset(params)
                        #   print ('Running learner = ' + learnername + ' on parameters ' + str(learner.getparams()))
                        # Train model
                        #'sub_trainset[0] are trainsize of vectors of  features, sub_trainset[1] are their target value
                        learner.learn(sub_trainset[0], sub_trainset[1])
                        # Test model
                        predictions = learner.predict(sub_testset[0])
                        
                        
                        #calculate the adjusted R square
                        SS_res= l2err_squared(predictions,sub_testset[1])
                        SS_total = l2err_squared(sub_testset[1],np.average(sub_testset[1])) 
                        
                        adj_R_squared=1-(SS_res/SS_total)
                        #adj_R_squared=1-(SS_res/ SS_total)*(sub_testset[0].shape[0]-1)/(sub_testset[0].shape[0]-1-sub_testset[0].shape[1])
                        
                        #print ('Adjusted R-squared for ' + learnername + ': ' + str(adj_R_squared)+' on parameter '+ str(learner.getparams())+' in fold '+str(count_iter))

                        errors[learnername][p,count_iter] =adj_R_squared
                    
                    count_iter+=1

            #regressionalgs.items return a list of tuples of key-value pairs in the dictionary
            for learnername, learner in regressionalgs.items():
                #find best error and its corresponding params for each alg in regressionalgs
                best_R_squared = np.mean(errors[learnername][0,:])
                bestparams = 0
                for p in range(numparams):
                    ave_R_squared = np.mean(errors[learnername][p,:])
                    
                    #Larger R-squared is better
                    if ave_R_squared >best_R_squared:
                        best_R_squared=ave_R_squared
                        bestparams =p
                # Extract best parameters
                learner.reset(parameters[bestparams])
                
                
                
                #print ('Best parameters for ' + learnername + ': ' + str(learner.getparams()))
                #print ('Average error for ' + learnername + ': ' + str(best_R_squared))
                #se=np.std(errors[learnername][p,:])/np.sqrt(numruns)
                #print ('Standard error for ' + learnername + ': ' + str(se))
            
            for learnername, learner in regressionalgs.items():
                # Train model with tuned parameter

                
                learner.learn(trainset[0], trainset[1])
                # Test model
                predictions = learner.predict(testset[0])
                #calculate the adjusted R square
                

                SS_res= l2err_squared(predictions,testset[1])
                
                SS_total = l2err_squared(testset[1],np.average(testset[1]))

                adj_R_squared=1-(SS_res/SS_total)
                #adj_R_squared=1-(SS_res/ SS_total)*(testset[0].shape[0]-1)/(testset[0].shape[0]-1-testset[0].shape[1])
                
                print ('Adjusted R-squared for ' + learnername + ' on optimal parameters ' + str(learner.getparams())+' given fold '+str(fold_count)+' is : ' + str(adj_R_squared))

                #store the R-square of this learner with respect to optimal para in current fold, p is the index of optimal para in para list;
                #fold_count used to locate current fold
                for p in range(numparams):
                    if learner.getparams()==parameters[p]:
                        overall_R_squared[learnername][p,fold_count] =adj_R_squared
            
            fold_count+=1
        
        for learnername in regressionalgs:
            best_R_squared = np.mean(overall_R_squared[learnername][0,:])
            bestparams = 0
            for p in range(numparams):
                ave_R_squared = np.mean(overall_R_squared[learnername][0,:])
                if ave_R_squared >best_R_squared:
                    best_R_squared=ave_R_squared
                    bestparams =p
            
            print('Given this dataset, the optimal parameter for '+learnername+' is '+str(parameters[bestparams])+' with adjusted R-squared '+str(best_R_squared))
            
                
            

    

        
            






