import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

from project1_d import  ols_gradient , ridge_gradient , ADAgrad , RMSProp , ADAM , momentum_gd




def Stochastic_gradient(x , y  , degree , theta , lambd , method , optimizer , nr_batches , size_batch , epochs):
    X = PolynomialFeatures(degree).fit_transform(x.reshape(-1 , 1))
    
    datapoint = len(x)
    
    for epoch in range(epochs):
        k = np.random.randint(datapoint)

        if method == "OLS":
            grad = ols_gradient(X , y , theta)
        elif method == "Ridge":
            grad = ridge_gradient(X , y , lambd)

    '''
    Here we will be splitting into batches of our dataset.
    '''


    