import random
import sys
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.utils import to_categorical

from skimage.segmentation import slic
from skimage.color import gray2rgb
from lime import lime_image

emotion = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad', 5:'Surprise', 6:'Neutral'}
to_use = [19, 0, 18, 15, 172, 23, 29]

Data = np.load('./extracted_data.npz')
X, y = Data['X'][:4000], Data['y'][:4000]
y = to_categorical(y, 7)

name = sys.argv[1]
Model = load_model(name)

y_test = Model.predict_classes(X)
y_test = to_categorical(y_test, 7)

Explainer = lime_image.LimeImageExplainer()

def predict(img):
    return Model.predict_proba( img[..., 0:1] )

def segmentation(img):
    return slic(img, n_segments=64)

for cls in range(7):
    class_indices = np.where( (y[:, cls] == 1) & (y_test[:, cls] == 1) )[0]
    image = X[ class_indices[ to_use[cls] ] ].reshape( (48, 48) )
    
    explanation = Explainer.explain_instance(
                    image=image,
                    classifier_fn=predict,
                    segmentation_fn=segmentation
                  )
    result, mask = explanation.get_image_and_mask(
                    label=cls,
                    positive_only=False,
                    hide_rest=False,
                    num_features=5
                   )

    fig, axarr = plt.subplots(1, 2)
    fig.suptitle( emotion[cls], fontsize=16 )
    
    axarr[0].imshow( image, cmap='gray')
    axarr[0].set_title( 'Original' )
    axarr[1].imshow( result )
    axarr[1].set_title( 'Explanation' )
    
    plt.savefig('./%s.png' % emotion[cls])
    plt.show()
