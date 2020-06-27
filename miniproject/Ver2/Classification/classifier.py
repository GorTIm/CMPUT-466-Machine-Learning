"""
Data source and background: 
    https://www.kaggle.com/shebrahimi/financial-distress
    
First column: Company represents sample companies.

Second column: Time shows different time periods that data belongs to. 
Time series length varies between 1 to 14 for each company.

Third column: The target variable is denoted by "Financial Distress" 
if it is greater than -0.50 the company should be considered as healthy (0). 
Otherwise, it would be regarded as financially distressed (1).

Fourth column to the last column: The features denoted by x1 to x83, 
are some financial and non-financial characteristics of the sampled companies.
These features belong to the previous time period, which should be used to 
predict whether the company will be financially distressed or not (classification). 

Feature x80 is a categorical variable.

"""


"""from __future__ import division  # floating point division
    import csv
    import random
    import math"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler
from sklearn.metrics import r2_score
import seaborn as sns

if __name__ == '__main__':

    #import dataset
    dataset=pd.read_csv('..\datasets\Financial_Distress.csv',float_precision='high')
    #dataset.fillna(dataset.mean())
    #First row is the custom index
    X=dataset.iloc[:,1:-1].values
    #Use bracket to transform Y as 2d array
    Y=dataset.iloc[:,[-1]].values
    
    Y=np.asarray([1 if x[0]<-0.5 else 0 for x in Y])
    
    
    """
    X.max(axis=0)
    X.min(axis=0)
    X.argmax(axis=0)
    """
    
    
    """#Taking care of missing value
    
    """
    
    """ Another possible way to fill missing value:
        dataset.fillna(dataset.mean())
    """
    if not dataset.isnull().values.any():
        from sklearn.impute import SimpleImputer
        imputer=SimpleImputer(missing_values=np.nan, strategy='mean' )
        imputer=imputer.fit(X[:,:])
        X[:,:]=imputer.transform(X[:,:])
        
    
    from sklearn.preprocessing import LabelEncoder,OneHotEncoder
    #Encoding categorical data
    """
    #Transform label to different number
    labelEncoder_X=LabelEncoder()
    X[:,79]=labelEncoder_X.fit_transform(X[:,79])
    """
    
    #Dummy variable 虚拟编码
    from sklearn.compose import ColumnTransformer 
    ct = ColumnTransformer([("Unknown_category", OneHotEncoder(),[79])],remainder="passthrough")
    # The last arg ([0]) is the list of columns you want to transform in this step
    X=ct.fit_transform(X)  
    
    """
    #Required if Y is categorical data
    labelEncoder_y=LabelEncoder()
    Y=labelEncoder_y.fit_transform(Y)
    """

    #split the dataset into training set and test set
    X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
    
 
    
    #feature scaling with Standardization
    x_sc=StandardScaler()
    X_train=x_sc.fit_transform(X_train)
    X_test=x_sc.transform(X_test)
    """
    #Rescaling on Y is not necessary for classification problem
    y_sc=StandardScaler()
    y_train=y_sc.fit_transform(y_train)
    #y_test=y_sc.fit(y_test)
    """
    

    """
    x_nc=MinMaxScaler()
    X_train=x_nc.fit_transform(X_train)
    X_test=x_nc.transform(X_test)
   
    #Rescaling on Y is not necessary for classification problem
    y_nc=MinMaxScaler()
    y_train=y_nc.fit_transform(y_train)
    #y_test=y_nc.transform(y_test)
  
    """
    
    
   
    """    
    #feature scaling with Robust scaler
    x_rb=RobustScaler(quantile_range = (0.20,0.80))
    X_train=x_rb.fit_transform(X_train)
    X_test=x_rb.transform(X_test)
    
    #Rescaling on Y is not necessary for classification problem
    y_rb=RobustScaler(quantile_range = (0.20,0.80))
    y_train=y_rb.fit_transform(y_train)
    y_test=y_rb.transform(y_test)
    """
    
    state = np.random.RandomState(42)
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    models={"logistic":LogisticRegression(random_state=0),
            "naive_bayes":GaussianNB(),
            "SVM":SVC(kernel='linear')
            }
    """
    "SVM":OneClassSVM(kernel='rbf', degree=3, gamma=0.1,nu=0.05, 
                                         max_iter=-1, random_state=state)
    """
    
    from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
    for name,model in models.items():
        
        classifier = model
        classifier.fit(X_train, y_train)
        
        y_pred = classifier.predict(X_test)
        
        
        cm=confusion_matrix(y_test,y_pred)
        n_errors = (y_pred != y_test).sum()
        # Run Classification Metrics
        print("Model {}: {}".format(name,n_errors))
        print("Accuracy Score :")
        print(accuracy_score(y_test,y_pred))
        print("Classification Report :")
        print(classification_report(y_test,y_pred))

    
    """
    regressor.score(X_train, y_train)
    regressor.score(X_test, y_test)
    print(r2_score(y_test,y_pred))
    
    import statsmodels.formula.api as sm
    from  statsmodels.regression.linear_model import OLS
    X_train_2 = np.append(arr = np.ones((X_train.shape[0], 1)).astype(float), values = X_train, axis = 1)
    X_opt = X_train_2[:, :]
    regressor_OLS = OLS(endog = y_train, exog = X_opt).fit()
    regressor_OLS.summary()
    regressor_OLS.pvalues
    while(regressor_OLS.rsquared_adj)
    """
    
    """
    #Using Pearson Correlation
    plt.figure(figsize=(12,10))
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    plt.show()
    cor = dataset.corr()
    #Correlation with output variable
    cor_target = abs(cor["Financial Distress"])
    #Selecting highly correlated features
    relevant_features = cor_target[cor_target>0.5]
    """
    

    

