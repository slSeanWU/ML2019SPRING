import numpy as np
import pandas as pd
from scipy.special import expit

## Read csv and Feature Extraction

# features to be dropped: final weight, marital status
Cols = pd.read_csv('./X_train', nrows=1)
toDrop = ['fnlwgt', ' Divorced',' Married-AF-spouse',' Married-civ-spouse',
              ' Married-spouse-absent',' Never-married',' Separated',' Widowed']
raw_X = pd.read_csv('./X_train', usecols=[x for x in Cols if x not in toDrop] )

y = pd.read_csv('./Y_train').to_numpy(dtype=np.double)
y = y.reshape( (len(y),) )

print( raw_X.head() )

X = raw_X.to_numpy(dtype=np.double)
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

'''
## Read test data and produce output

test_data = pd.read_csv('./test.csv', usecols=cols[:-1])
testX = []

for idx, person in test_data.iterrows():
    inputVec = [ float(person['age']), float(person['age'] ** 2) ]

    if person['education'] in eduDict:
        inputVec.append( eduDict[ person['education'] ] )
    else:
        inputVec.append( 0.0 )

    inputVec.extend( [ float(person['education_num']), float(person['education_num'] ** 2) ] )
    inputVec.append( float( person['occupation'] in occuT ) )
    inputVec.append( float( person['race'] in raceT ) )
    inputVec.append( float( person['sex'] == ' Male' ) )
    inputVec.append( float(person['hours_per_week']) )

    testX.append( inputVec )

testX = (testX - featMins) / (featMaxs - featMins + 1e-18)
bias = np.ones( (len(testX), 1) )
testX = np.append(testX, bias, axis=1)

Soft = expit( np.dot(testX, w) )
plt.hist( Soft, [0.05*x for x in range(21)] )
plt.show()'''
