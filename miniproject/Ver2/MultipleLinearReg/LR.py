"""from __future__ import division  # floating point division
    import csv
    import random
    import math"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

if __name__ == '__main__':

    #import dataset
    dataset=pd.read_csv('..\datasets\Financial_Distress.csv')
    dataset.fillna(dataset.mean())
    #First row is the custom index
    X=dataset.iloc[:,1:-1].values
    #Use bracket to transform Y as 2d array
    Y=dataset.iloc[:,[-1]].values

    """#Taking care of missing value
        from sklearn.preprocessing import Imputer,LabelEncoder,OneHotEncoder
        imputer=Imputer('NaN','mean' )
        imputer=imputer.fit(X[:,1:3])
        X[:,1:3]=imputer.transform(X[:,1:3])"""
    
    """ Another possible way to fill missing value:
        dataset.fillna(dataset.mean())
        """

    """#Encoding categorical data
        laberEncoder_X=LabelEncoder()
        X[:,0]=laberEncoder_X.fit_transform(X[:,0])
        #Dummy variable 虚拟编码
        oneHotEncoder=OneHotEncoder(categorical_features=[0])
        X=oneHotEncoder.fit_transform(X).toarray()
        labelEncoder_y=LabelEncoder()
        Y=labelEncoder_y.fit_transform(Y)"""

    #split the dataset into training set and test set
    X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

    #feature scaling
    x_sc=StandardScaler()
    X_train=x_sc.fit_transform(X_train)
    X_test=x_sc.transform(X_test)
    #Rescaling on Y is not necessary for classification problem
    y_sc=StandardScaler()
    y_train=y_sc.fit_transform(y_train)
    y_test=y_sc.transform(y_test)
    
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    
    y_pred = regressor.predict(X_test)
    
    regressor.score(X_train, y_train)
    regressor.score(X_test, y_test)
    
    print(r2_score(y_test,y_pred))


