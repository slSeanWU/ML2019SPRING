import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

## Read csv and Feature Extraction

# feature to be dropped: final weight
Cols = pd.read_csv('./X_train', nrows=1)
'''candidate toDrop = ['fnlwgt', ' Divorced',' Married-AF-spouse',' Married-civ-spouse',
              ' Married-spouse-absent',' Never-married',' Separated',' Widowed']'''

toDrop = ['fnlwgt']
raw_X = pd.read_csv('./X_train', usecols=[x for x in Cols if x not in toDrop] )

y = pd.read_csv('./Y_train').to_numpy(dtype=np.double)
y = y.reshape( (len(y),) )

print( raw_X.head() )

X = raw_X.to_numpy(dtype=np.double)
print(X.shape)
print(X[:30])
print(y[:30])


## Feature scaling
featMins = np.min(X, axis=0)
featMaxs = np.max(X, axis=0)
X = (X - featMins) / (featMaxs - featMins + 1e-32)


## Training: Sklearn Gradient Boosting Classifier w/ parameter tuning
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train, y_train = X, y
Parameters = { 'min_samples_split': [400], 
               'min_samples_leaf': [5],
               'max_depth': [6],
               'max_features': [0.5],
               'n_estimators': [250],
               'learning_rate': [0.1],
               'subsample': [1]}

''' for tuning
Parameters = {'min_samples_split': [50, 200, 400],
              'min_samples_leaf': [1, 5, 10, 30],
              'max_depth': [4, 6, 8, 12, 16],
              'max_features': [0.2, 0.33, 0.5],
              'n_estimators: [50, 100, 250, 400],
              'subsample: [0.8, 1]}               
'''
Model = GridSearchCV(GradientBoostingClassifier(), Parameters, cv=5, scoring='accuracy', n_jobs=-1)
Model.fit(X_train, y_train)

print('best set:', Model.best_params_)
print('All results:')

means = Model.cv_results_['mean_test_score']
stds = Model.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, Model.cv_results_['params']):
    print('%.6f (+/-%.6f) for %r' % (mean, std*2, params))

''' for validation
Pred = Model.predict(X_test)
print( 'final test: accuracy = %.6f' % (100*accuracy_score(y_test, Pred)) )
'''
## Testing and Output to csv

test_Data = pd.read_csv('./X_test', usecols=[x for x in Cols if x not in toDrop] )

X_test = test_Data.to_numpy(dtype=np.double)
X_test = (X_test - featMins) / (featMaxs - featMins + 1e-32)

Ans = Model.predict(X_test).astype(int)
ans_Frame = pd.DataFrame(data=Ans, index=range(1, len(Ans)+1), columns=['label'])
ans_Frame.to_csv('./submission.csv', index_label='id')

