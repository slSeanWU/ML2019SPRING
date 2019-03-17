import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

## Read csv and Feature Extraction
def read_data(inputFile, catFile, y_File): # second file for education_yrs
    
    # decide features to be dropped
    Cols = pd.read_csv(inputFile, nrows=1)
    toDrop = ['fnlwgt']
    '''candidates of toDrop = ['fnlwgt', ' Divorced',' Married-AF-spouse',' Married-civ-spouse',
              ' Married-spouse-absent',' Never-married',' Separated',' Widowed']
    '''

    raw_X = pd.read_csv(inputFile, usecols=[x for x in Cols if x not in toDrop] )
   
    # add years of education and age squared
    edu_yr = pd.read_csv(catFile, usecols=['education_num']).to_numpy(dtype=np.double)
    edu_yr = edu_yr.reshape( (edu_yr.shape[0],1) )
    age_sqr = ( raw_X['age'].to_numpy(dtype=np.double) ) ** 2
    age_sqr = age_sqr.reshape( (age_sqr.shape[0],1) )

    # produce input matrix
    print( raw_X.head() )
    X = raw_X.to_numpy(dtype=np.double)
    print(X.shape, age_sqr.shape, edu_yr.shape)
    X = np.concatenate( (X, age_sqr, edu_yr), axis=1 )
    print(X[:30])
    
    if not y_File:
        return X

    if y_File: # training data labels
        y = pd.read_csv(y_File).to_numpy(dtype=np.double)
        y = y.reshape( (len(y),) )
        print(y[:30])

        return X, y


X, y = read_data('./X_train', './train.csv', './Y_train')

## Feature scaling & append bias term
featMins = np.min(X, axis=0)
featMaxs = np.max(X, axis=0)
X = (X - featMins) / (featMaxs - featMins + 1e-32)

scaler = StandardScaler()
scaler.fit_transform(X)

## Training: Sklearn LinearSVM w/ C=2.0

''' for validation: X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) '''

X_train, y_train = X, y
''' for parameter tuning:
Cs = [0.5, 1, 3, 10, 30, 100, 300, 1500, 4000]

    y_pred = LSVC.predict(X_test)
    print( 'c=%.2f, accuracy = %.6f (+/-%.6f)' % (c, 100.0*scores.mean(), 200*scores.std() ) )
'''
LSVC = LinearSVC(dual=False, fit_intercept=False, C=3.0)
LSVC.fit(X_train, y_train)


## Testing and Output to csv

X_test = read_data('./X_test', './test.csv', '')

X_test = (X_test - featMins) / (featMaxs - featMins + 1e-32)
scaler.transform(X_test)

Ans = LSVC.predict(X_test).astype(int)
ans_Frame = pd.DataFrame(data=Ans, index=range(1, len(Ans)+1), columns=['label'])
ans_Frame.to_csv('./submission.csv', index_label='id')

