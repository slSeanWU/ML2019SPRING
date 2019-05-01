import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.models import load_model

Data = np.load('./extracted_data.npz')
X, y = Data['X'], Data['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

Model = load_model( sys.argv[1] )
y_pred = Model.predict_classes( X_test )
y_test = y_test.reshape( (y_test.shape[0],) )

labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
cm = confusion_matrix(y_test, y_pred, labels=[0,1,2,3,4,5,6])
sums = np.sum(cm, axis=1, keepdims=True)
percentages = cm / sums.astype(float) * 100

annotations = np.empty_like(percentages).astype(str)
for row in range(7):
    cls_sum = sums[row]
    for col in range(7):
        perc = percentages[row, col]
        cnt = cm[row, col]
        annotations[row, col] = '%.2f%%\n%d/%d' % (perc, cnt, cls_sum)

cm = pd.DataFrame(percentages, index=labels, columns=labels)
cm.index.name = 'True'
cm.columns.name = 'Predicted'
fig, axarr = plt.subplots( figsize=(16,16) )
axarr.set(aspect='equal')
sns.heatmap(cm, annot=annotations, fmt='', cmap='YlGnBu')
plt.savefig('./Confusion_Matrix.png')
plt.show()


