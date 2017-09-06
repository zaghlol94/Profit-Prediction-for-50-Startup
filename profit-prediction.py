# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 19:23:38 2017

@author: zaghlollight
"""
#importing lib and dataset

import numpy as np
import pandas as pd
dataset = pd.read_csv('50_Startups.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

#encoding cat data

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
lex=LabelEncoder()
x[:,3]=lex.fit_transform(x[:,3])
ohe=OneHotEncoder(categorical_features=[3])
x=ohe.fit_transform(x).toarray()

#avoid dummy var trap and multicollinearity

x=x[:,1:]

'''
we don't need feat. scaling here because since y is a linear combination of the
independent variables,the coefficients can adapt their scale to put everything
on the same scale.
For example if you have two independent variables x1 and x2 and if y takes 
values between 0 and 1, x1 takes values between 1 and 10 and x2 takes values 
between 10 and 100, then b1 can be multiplied by 0.1 and b2 can be multiplied
by 0.01 so that y, b1*x1 and b2*x2 are all on the same scale.
'''

#spliting data

from sklearn.cross_validation import train_test_split
Xtrain,Xtest,Ytrain,Ytest =train_test_split(x,y,test_size=0.2)


#train model

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(Xtrain,Ytrain)

#predicting and test

Ypred=reg.predict(Xtest)

#feat. selection using backward technique
import statsmodels.formula.api as sm
x=np.append(arr=np.ones((50,1)).astype(int),values=x,axis=1)
Xopt=x[:,[0,1,2,3,4,5]]
regOls=sm.OLS(endog=y,exog=Xopt).fit()
regOls.summary()


#eliminate 2 which has the heightest P-value
Xopt=x[:,[0,1,3,4,5]]
regOls=sm.OLS(endog=y,exog=Xopt).fit()
regOls.summary()

#eliminate 1 which has the heightest P-value
Xopt=x[:,[0,3,4,5]]
regOls=sm.OLS(endog=y,exog=Xopt).fit()
regOls.summary()

#eliminate 3 which has the heighst p-value
Xopt=x[:,[0,4,5]]
regOls=sm.OLS(endog=y,exog=Xopt).fit()
regOls.summary()


