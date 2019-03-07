import sys
import csv
import numpy

## Read test input
print(sys.argv)
testFile, outputFile = sys.argv[1], sys.argv[2]
testX = []

with open(testFile, 'r', encoding='big5') as test_csv:

    test_reader = csv.reader( test_csv )
    row_cnt = 0
    inputVec = []
    
    for row in test_reader:
        if (row_cnt%18) == 0:
        	inputVec = []
        	
        for col in range(2, 11):
            if row[col] != 'NR':
                inputVec.append( float(row[col]) )
            else:
                inputVec.append(0.0)
        
        if (row_cnt%18) == 17:
            testX.append( inputVec )

        row_cnt += 1

#print( len(testX) )
#print( testX[1][-20:] )

## Produce prediction and write to CSV
model = numpy.load('./Adagrad_model.npz')
w, featMins, featMaxs = model['w'], model['featMins'], model['featMaxs']

testX = numpy.array(testX)
testX = (testX - featMins) / (featMaxs - featMins + 1e-16)

bias = numpy.ones( (len(testX), 1) )
testX = numpy.append( testX, bias, axis=1 )
testY = numpy.dot(testX, w)

#print( len(testY) )
print( 'First 30 predictions:' )
print( testY[:30] )

with open(outputFile, 'w') as ans_csv:
    ans_writer = csv.writer(ans_csv, delimiter=',')
    ans_writer.writerow(['id', 'value'])

    for i in range( len(testY) ):
        if testY[i] > 0.0:
            ans_writer.writerow( [ 'id_'+str(i), testY[i] ] )
        else:
            ans_writer.writerow( [ 'id_'+str(i), 0.0 ] )



