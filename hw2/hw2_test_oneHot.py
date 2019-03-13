import pandas as pd
import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt

## Read test data and produce output

model = np.load('./model_GD_oneHot.npz')
w, featMins, featMaxs = model['w'], model['mins'], model['maxs']

# features to be dropped
Cols = pd.read_csv('./X_test', nrows=1)
'''toDrop = ['fnlwgt', ' Divorced',' Married-AF-spouse',' Married-civ-spouse',
              ' Married-spouse-absent',' Never-married',' Separated',' Widowed']'''
toDrop = ['fnlwgt']
test_Data = pd.read_csv('./X_test', usecols=[x for x in Cols if x not in toDrop] )

test_edu = pd.read_csv('./test.csv', usecols=['education_num']).to_numpy(dtype=np.double)
test_edu.reshape( (test_edu.shape[0],1) )
test_age_sqr = test_Data['age'].to_numpy(dtype=np.double)
test_age_sqr = (test_age_sqr ** 2).reshape( (test_age_sqr.shape[0],1) )

testX = test_Data.to_numpy(dtype=np.double)
testX = np.concatenate( (testX, test_age_sqr, test_edu), axis=1 )
testX = (testX - featMins) / (featMaxs - featMins + 1e-32)
bias = np.ones( (len(testX), 1) )
testX = np.append(testX, bias, axis=1)

Soft = expit( np.dot(testX, w) )

## Show and plot the distribution of Soft results
record = np.percentile(Soft, [5*x for x in range(21)])
for i in range(21):
    print('%d-th = %.6f' % (i*5, record[i]))
plt.hist( Soft, [0.05*x for x in range(21)] )
plt.show()

## Define threshold and write to csv
Ans = np.where(Soft > 0.5, int(1), int(0))
ans_Frame = pd.DataFrame(data=Ans, index=range(1, len(Ans)+1), columns=['label'])
ans_Frame.to_csv('./submission.csv', index_label='id')

