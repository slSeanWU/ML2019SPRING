import numpy as np
import pandas as pd
import os, signal, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from sklearn.model_selection import train_test_split, StratifiedKFold
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dropout
from keras.layers import Flatten, Dense
from keras.layers import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.callbacks import History, EarlyStopping, ModelCheckpoint, CSVLogger
from keras.utils import plot_model, to_categorical
import matplotlib.pyplot as plt


## Read csv and transform to numpy array
''' for reading training data
raw_Data = pd.read_csv('./train.csv')
raw_X = raw_Data['feature'].tolist()

X = []
for row in raw_X:
    X.append( [int(pix) for pix in row.split(' ')] )

X = np.array(X, dtype=np.float32)
X = X / 255.0
X = X.reshape( (X.shape[0], 48, 48, 1) )
y = raw_Data['label'].to_numpy( dtype=np.int32 )
y = y.reshape( (y.shape[0], 1) )
print ( X.shape, y.shape )

np.savez('extracted_data', X=X, y=y)
'''


Data = np.load('./extracted_data.npz')
X, y = Data['X'], Data['y']

Strat_Kfold = StratifiedKFold(n_splits=6)
Strat_Kfold.get_n_splits(X, y)

Augmentation = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                zoom_range=[0.8, 1.2],
                shear_range=0.1,
                horizontal_flip=True )

def buildModel(Type):
    CNN = Sequential()
    
    if Type == 0:
        CNN.add( Conv2D(64, (5, 5), input_shape=(48, 48, 1), padding='same', kernel_initializer='glorot_normal') )
    else:
        CNN.add( Conv2D(128, (3, 3), input_shape=(48, 48, 1), padding='same', kernel_initializer='glorot_normal') )

    CNN.add( LeakyReLU(alpha=0.04) )
    CNN.add( BatchNormalization() )

    if Type == 0:
        CNN.add( MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same') )
    else:
        CNN.add( MaxPooling2D(pool_size=(2, 2), padding='same') )
    CNN.add( Dropout(0.25) )

    CNN.add( Conv2D(128, (3, 3), padding='same', kernel_initializer='glorot_normal') )
    CNN.add( LeakyReLU(alpha=0.04) )
    CNN.add( BatchNormalization() )
    CNN.add( MaxPooling2D(pool_size=(2, 2), padding='same') )
    CNN.add( Dropout(0.3) )

    CNN.add( Conv2D(256, (3, 3), padding='same', kernel_initializer='glorot_normal') )
    CNN.add( LeakyReLU(alpha=0.04) )
    CNN.add( BatchNormalization() )
    CNN.add( MaxPooling2D(pool_size=(2, 2), padding='same') )
    CNN.add( Dropout(0.3) )

    CNN.add( Conv2D(512, (3, 3), padding='same', kernel_initializer='glorot_normal') )
    CNN.add( LeakyReLU(alpha=0.04) )
    CNN.add( BatchNormalization() )
    CNN.add( MaxPooling2D(pool_size=(2, 2), padding='same') )
    CNN.add( Dropout(0.35) )


    CNN.add( Flatten() )
    
    if Type == 0:
        CNN.add( Dense(1024, activation='relu', kernel_initializer='glorot_normal') )
        CNN.add( BatchNormalization() )
        CNN.add( Dropout(0.5) )

    else:
        CNN.add( Dense(512, activation='relu', kernel_initializer='glorot_normal') )
        CNN.add( BatchNormalization() )
        CNN.add( Dropout(0.5) )

        CNN.add( Dense(512, activation='relu', kernel_initializer='glorot_normal') )
        CNN.add( BatchNormalization() )
        CNN.add( Dropout(0.5) )

    CNN.add( Dense(7, activation='softmax', kernel_initializer='glorot_normal') )

    CNN.compile( loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'] )
    print('Model %d' % (Type))
    print( CNN.summary() )

    return CNN

cnt = 0
for tr_idx, te_idx in Strat_Kfold.split(X, y):
    print( 'building model #%d...' % (cnt) )
    
    if not cnt%2:
        Model = buildModel(1)
    else:
        Model = buildModel(0)

    X_train, X_test, y_train, y_test = X[tr_idx], X[te_idx], y[tr_idx], y[te_idx]
    
    y_train, y_test = to_categorical(y_train, 7), to_categorical(y_test, 7)

    Hist = History()
    Early_stop = EarlyStopping(monitor='val_acc', patience=30, verbose=1)
    Checkpoint = ModelCheckpoint('./CNN%d-{epoch:05d}-{val_acc:.5f}.h5' % (cnt), monitor='val_acc', save_best_only=True)
    Logger = CSVLogger('./progress.csv', append=True)

    n_batch = 128
    Progress = Model.fit_generator(
        Augmentation.flow(X_train, y_train, batch_size=n_batch),
        epochs = 400,
        steps_per_epoch = 5*(X_train.shape[0]) // n_batch,
        validation_data = (X_test, y_test),
        callbacks = [Hist, Checkpoint, Logger, Early_stop],
        workers = 10
    )


    plt.plot(Progress.history['acc'])
    plt.plot(Progress.history['val_acc'])
    plt.title('Accuracy History')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('./accuracy_chart_#%d.png' % (cnt))

    Model.save('./CNN%d.h5' % (cnt) )
    cnt += 1

