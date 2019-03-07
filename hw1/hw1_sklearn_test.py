import csv
import sys
import numpy
import pickle
from sklearn.linear_model import Ridge

testFile, outputFile = sys.argv[1], sys.argv[2]
## Read test input

testX = []

linear_ele = [6, 7, 8, 9, 10, 12, 14, 15, 16, 17]
quadratic_ele = [6, 7, 8, 9, 10, 12, 16, 17]
cubic_ele = [8, 9]

with open(testFile, 'r', encoding='big5') as test_csv:

    test_reader = csv.reader( test_csv )
    row_cnt = 0
    inputVec = []
    
    for row in test_reader:
        field = row_cnt%18

        if field not in linear_ele:
            row_cnt += 1
            continue

        if field == min(linear_ele):
            inputVec = []
        	
        if field in linear_ele:
            for col in range(2, 11):
                if row[col] != 'NR':
                    inputVec.append( float(row[col]) )
                else:
                    inputVec.append(0.0)
        if field in quadratic_ele:
            for col in range(2, 11):
                if row[col] != 'NR':
                    inputVec.append( float(row[col]) ** 2 )
                else:
                    inputVec.append(0.0)
        if field in cubic_ele:
            for col in range(2, 11):
                inputVec.append( float(row[col]) ** 3 )
                 
        if field == max(linear_ele):
            testX.append( inputVec )

        row_cnt += 1

#print( len(testX) )
#print( testX[0] )

## Produce prediction and write to CSV
scale = numpy.load('./Scaling.npz')
LM = pickle.load( open('./Sklearn_Ridge_model.sav', 'rb') )

featMins, featMaxs = scale['mins'], scale['maxs']
testX = numpy.array(testX)
testX = (testX - featMins) / (featMaxs - featMins + 1e-16)
testY = LM.predict(testX)

#print( len(testY) )
print( 'First 30 predictions:' )
print( testY[:30] )

with open(outputFile, 'w') as ans_csv:
    ans_writer = csv.writer(ans_csv, delimiter=',')
    ans_writer.writerow(['id', 'value'])

    for i in range( len(testY) ):
        ans_writer.writerow( [ 'id_'+str(i), max(0.0, testY[i]) ] )


