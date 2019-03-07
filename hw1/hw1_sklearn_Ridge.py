import csv
import numpy
import pickle
from sklearn.linear_model import Ridge

## Stiching the days together, store in "raw_table"

num_indicators = 18
raw_table = [ [] for x in range(18) ]

with open('./train.csv', 'r', encoding='big5') as train_csv:

    train_reader = csv.reader( train_csv )
    row_cnt = 0
    next( train_reader, None )

    for row in train_reader:
        indic_now = row_cnt % num_indicators 
        for col in range(3, 27):
            if row[col] == 'NR':
                raw_table[ indic_now ].append(0.0)
            elif float(row[col]) < 0.0:
                raw_table[ indic_now ].append( raw_table[ indic_now ][-1] )
            else:
                raw_table[ indic_now ].append( float(row[col]) )

        row_cnt += 1


## Produce input matrix X and output vector Y
## Apply feature engineering

entries_per_month = 480
sets_per_month = 471

X = []
y = []

# Linear: NOx, O3, PM10, PM2.5, Rainfall, SO2, Wind-directions, Wind-Speeds
linear_ele = [6, 7, 8, 9, 10, 12, 14, 15, 16, 17]
# Quadratic: All linear ones except Wind-directions
quadratic_ele = [6, 7, 8, 9, 10, 12, 16, 17]
# Cubic: PM10, PM2.5
cubic_ele = [8, 9]

for M in range(12):
    month_offset = M * entries_per_month

    for startHR in range(sets_per_month):
        inputVec = []
        for dim in range(18):
            if dim in linear_ele:
                for hr in range(9):
                    inputVec.append( raw_table[dim][ month_offset + startHR + hr ] )
            if dim in quadratic_ele:
                for hr in range(9):
                    inputVec.append( raw_table[dim][ month_offset + startHR + hr ] ** 2 )
            if dim in cubic_ele:
                for hr in range(9):
                    inputVec.append( raw_table[dim][ month_offset + startHR + hr ] ** 3 )

        X.append( inputVec )
        y.append( raw_table[9][ month_offset + startHR + 9 ] )  #pm2.5 after 10 hrs

#print (len(X[0]))
#print (len(X))

## Training: Scikit-learn Ridge Regression

X = numpy.array(X)
featMins = numpy.min(X, axis=0)
featMaxs = numpy.max(X, axis=0)

X = ( X - featMins ) / ( featMaxs - featMins + 1e-16 )
y = numpy.array(y)

#print(X[0][-30:])
LM = Ridge( alpha=3e-2 )
LM.fit(X, y)


## Save model and scaling information
pickle.dump(LM, open('./Sklearn_Ridge_model.sav', 'wb'))
numpy.savez('./Scaling', mins=featMins, maxs=featMaxs)

