import csv
import numpy

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

entries_per_month = 480
sets_per_month = 471

X = []
y = []

for M in range(12):
    month_offset = M * entries_per_month

    for startHR in range(sets_per_month):
        inputVec = []
        for dim in range(18):
            for hr in range(9):
                inputVec.append( raw_table[dim][ month_offset + startHR + hr ] )

        X.append( inputVec )
        y.append( raw_table[9][ month_offset + startHR + 9 ] )  #pm2.5 after 10 hrs


## Training: AdaGrad

X = numpy.array(X)
featMins = numpy.min(X, axis=0)
featMaxs = numpy.max(X, axis=0)

# minMax normalization & adding bias term
X = (X - featMins) / ( featMaxs - featMins + 1e-16 )
bias = numpy.ones( (len(X), 1) )
X = numpy.append(X, bias, axis=1)

y = numpy.array(y)
Xt = X.transpose()

w = numpy.zeros( len(X[0]) )
AdaGrad = numpy.zeros( len(X[0]) )
epochs = 5000
eta = 150.0

for i in range(epochs):
    Xw = numpy.dot(X, w)
    gradient = 2.0 * numpy.dot( Xt, (Xw-y) )
    
    AdaGrad += numpy.inner(gradient, gradient)
    w -= ( eta/numpy.sqrt(AdaGrad) * gradient )
    
    if i >= epochs-10:
        print('round %d: error = %.6f' % (i, numpy.sqrt( numpy.inner(Xw-y, Xw-y)/len(X) ) ) )
        print('Last 18 weights:')
        print( w[-18:] )


## Save to model

numpy.savez('Adagrad_model', w=w, featMins=featMins, featMaxs=featMaxs)

'''## Read test input

testX = []
with open('./test.csv', 'r', encoding='big5') as test_csv:

    test_reader = csv.reader( test_csv )
    row_cnt = 0
    inputVec = []
    
    for row in test_reader:
        if (row_cnt%18) == 0:
        	inputVec = []
        	
        for col in range(2, 11):
            if row[col] != 'NR':
                inputVec.append(float(row[col]))
            else:
                inputVec.append(0.0)
        
        if (row_cnt%18) == 17:
            testX.append( inputVec )

        row_cnt += 1

#print( len(testX) )
#print( testX[1][-20:] )

## Produce prediction and write to CSV

testX = numpy.array(testX)
testX = (testX - featMins) / (featMaxs - featMins + 1e-16)

bias = numpy.ones( (len(testX), 1) )
testX = numpy.append( testX, bias, axis=1 )
testY = numpy.dot(testX, w)

#print( len(testY) )
print( testY[:30] )

with open('./submission.csv', 'w') as ans_csv:
    ans_writer = csv.writer(ans_csv, delimiter=',')
    ans_writer.writerow(['id', 'value'])

    for i in range( len(testY) ):
        if testY[i] > 0.0:
            ans_writer.writerow( [ 'id_'+str(i), testY[i] ] )
        else:
            ans_writer.writerow( [ 'id_'+str(i), 0.0 ] )

'''

