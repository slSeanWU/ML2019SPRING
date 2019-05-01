import sys
import numpy as np
import pandas as pd

from keras.models import load_model
from keras.utils import plot_model

name1, name2, name3 = sys.argv[1], sys.argv[2], sys.argv[3]
raw_Data = pd.read_csv('./test.csv')
raw_X = raw_Data['feature'].tolist()

X = []
for row in raw_X:
    X.append( [int(pix) for pix in row.split(' ')] )

X = np.array(X, dtype=np.float32)
X = X / 255.0
X = X.reshape( (X.shape[0], 48, 48, 1) )

print (X.shape)

Model1, Model2, Model3 = load_model(name1), load_model(name2), load_model(name3)
#plot_model(Model, to_file='cnn_structure.png')

prob1, prob2, prob3 = Model1.predict_proba(X), Model2.predict_proba(X), Model3.predict_proba(X)
prob = prob1+prob2+prob3
print ('Shape of Prob:', prob.shape)
print (prob[:10])

ans = np.argmax(prob, axis=1)
print ('Shape of answer:', ans.shape)
print (ans[:10])

ans_Frame = pd.DataFrame(data=ans, index=range(len(ans)), columns=['label'])
ans_Frame.to_csv('./submission.csv', index_label='id')

