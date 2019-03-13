import numpy as np
import pandas as pd
from scipy.special import expit

## Read csv and Feature Extraction

# feature to be dropped: final weight
Cols = pd.read_csv('./X_train', nrows=1)
'''candidate toDrop = ['fnlwgt', ' Divorced',' Married-AF-spouse',' Married-civ-spouse',
              ' Married-spouse-absent',' Never-married',' Separated',' Widowed']'''

toDrop = ['fnlwgt']
raw_X = pd.read_csv('./X_train', usecols=[x for x in Cols if x not in toDrop] )

## add years of education and age squared
edu_yr = pd.read_csv('./train.csv', usecols=['education_num']).to_numpy(dtype=np.double)
edu_yr = edu_yr.reshape( (edu_yr.shape[0],1) )
age_sqr = ( raw_X['age'].to_numpy(dtype=np.double) ) ** 2
age_sqr = age_sqr.reshape( (age_sqr.shape[0],1) )

y = pd.read_csv('./Y_train').to_numpy(dtype=np.double)
y = y.reshape( (len(y),) )

print( raw_X.head() )

X = raw_X.to_numpy(dtype=np.double)
print(X.shape, age_sqr.shape, edu_yr.shape)
X = np.concatenate( (X, age_sqr, edu_yr), axis=1 )
print(X[:30])
print(y[:30])


## Feature scaling & append bias term
featMins = np.min(X, axis=0)
featMaxs = np.max(X, axis=0)
X = (X - featMins) / (featMaxs - featMins + 1e-32)
bias = np.ones( (len(X), 1) )

X = np.append(X, bias, axis=1)

## Training: Logistic Regression & Gradient Descent w/ Adagrad
w = np.zeros( len(X[0]) )
Adagrad = np.zeros( len(X[0]) )
Xt = X.transpose()

epochs = 16000
eta = 15.0
N = len(X)

for i in range(epochs):
    Xw = np.dot(X, w)
    Predict = expit(Xw)

    gradient = np.dot(Xt, Predict - y) / N
    cost = -1.0 * ( np.sum(y * np.log(Predict.clip(min=1e-32)) ) + np.sum( (1.0-y) * np.log( (1.0-Predict).clip(min=1e-32) ) ) ) / N

    Adagrad += gradient ** 2
    w -= eta/np.sqrt(Adagrad) * gradient

    if not i % 100:
        print('Round %d: error = %.6f' % (i, cost) )

np.savez('model_GD_oneHot', w=w, mins=featMins, maxs=featMaxs)

