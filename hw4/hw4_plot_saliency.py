import random
import sys
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.utils import to_categorical
from keras import activations
from vis.visualization import visualize_saliency
from vis.utils import utils

emotion = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad', 5:'Surprise', 6:'Neutral'}
to_use = [19, 0, 18, 15, 172, 23, 29]

Data = np.load('./extracted_data.npz')
X, y = Data['X'][:4000], Data['y'][:4000]
y = to_categorical(y, 7)

name = sys.argv[1]
Model = load_model(name)

y_test = Model.predict_classes(X)
y_test = to_categorical(y_test, 7)

Model.layers[-1].activation = activations.linear
Model = utils.apply_modifications(Model)

for cls in range(7):
    class_indices = np.where( (y[:, cls] == 1) & (y_test[:, cls] == 1) )[0]
    print (class_indices.shape)

    fig, axarr = plt.subplots(1, 2)
    fig.suptitle( emotion[cls], fontsize=16 )
    
    axarr[0].imshow( X[ class_indices[ to_use[cls] ] ].reshape( (48,48) ), cmap='gray')
        
    grads = visualize_saliency(Model, -1, filter_indices=cls, seed_input=X[ class_indices[ to_use[cls] ] ], backprop_modifier='guided')
    smap = axarr[1].imshow( grads, cmap='jet' )
    
    plt.colorbar(smap, ax=axarr[1], fraction=0.046)
    plt.savefig('./%s.png' % emotion[cls])
    plt.show()
