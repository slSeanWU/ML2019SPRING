import sys
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import load_model

Model = load_model( sys.argv[1] )
layer_name = 'conv2d_4'

def deprocess(img):
    img -= np.mean(img)
    img /= ( np.std(img) + 1e-16 )
    img *= 0.1

    img += 0.5
    img = np.clip(img, 0., 1.)

    img *= 255
    img = np.clip(img, 0, 255).astype('uint8')

    return img


def generateFilter(layer_name, filter_idx):
    layer_output = Model.get_layer(layer_name).output
    #print( 'Shape of layer_output:', layer_output.get_shape() )
    
    loss = K.mean(layer_output[:, :, :, filter_idx])
    grads = K.gradients(loss, Model.input)[0]
    #print( 'Shape of grads:', grads.get_shape() )

    grads /= ( K.sqrt( K.mean(K.square(grads)) ) + 1e-16 )
    fctn = K.function( [Model.input], [loss, grads] )
    img_data = np.random.random( (1, 48, 48, 1) )*32 + 127

    lr = 1.0
    for i in range(64):
        cur_loss, cur_grads = fctn([img_data])
        img_data += cur_grads*lr

    img = img_data[0]
    img = deprocess(img)

    return img.reshape( (48, 48) )

fig, axarr = plt.subplots(8, 8, figsize=(32,32) )
fig.suptitle( 'Conv2d_4 Filters' )

for i in range(64):
    axarr[i//8, i%8].imshow( generateFilter(layer_name, i), cmap='gray' )
    axarr[i//8, i%8].axis('off')

plt.subplots_adjust(left=0.05, right=0.95, wspace=0.08, hspace=0.08)
plt.show()
