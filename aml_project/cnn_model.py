import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import random


'split data to test and train sets'
def split_data(data):
    split_size = 0.8
    split_index = int(len(data) * split_size)
    train_ds = data[:split_index]
    test_ds = data[split_index+1:]
    return train_ds, test_ds

'predict results from image'
def predict(model, images):
    x = random.randint(1,3500)
    return model.predict(np.expand_dims(images[x], axis=0)), images[x]

'model architecture and training'
def multi_task_model(images_train, images_test, face_train, face_test, smile_train, smile_test, scaled_size, n_epochs):
    'input layer'
    input_layer = keras.layers.Input(shape=(scaled_size, scaled_size, 3))
    
    'base branch'
    base_model = keras.layers.experimental.preprocessing.Rescaling(1./255, name='bm1')(input_layer)
    base_model = keras.layers.Conv2D(16, 3, padding='same', activation='relu', name='bm2')(base_model)
    base_model = keras.layers.MaxPooling2D(name = 'bm3')(base_model)
    base_model = keras.layers.Conv2D(32, 3, padding='same', activation='relu', name='bm4')(base_model)
    base_model = keras.layers.MaxPooling2D(name = 'bm5')(base_model)
    base_model = keras.layers.Conv2D(64, 3, padding='same', activation='relu', name='bm6')(base_model)
    base_model = keras.layers.MaxPooling2D(name = 'bm7')(base_model)
    base_model = keras.layers.Flatten(name='bm8')(base_model)

    'smile branch'
    smile_model = keras.layers.Dense(128, activation='relu', name='sm1')(base_model)
    smile_model = keras.layers.Dense(2, name='sm_head')(smile_model)

    'face branch'
    face_model = keras.layers.Dense(128, activation = 'relu', name = 'fm1')(base_model)
    face_model = keras.layers.Dense(64, activation = 'relu', name = 'fm2')(face_model)
    face_model = keras.layers.Dense(32, activation = 'relu', name = 'fm3')(face_model)
    face_model = keras.layers.Dense(4, activation='sigmoid', name = 'fm_head')(face_model)

    'model structure'
    model = keras.Model(input_layer, outputs=[smile_model, face_model])

    'losses functions for both branches'
    losses = {"sm_head": keras.losses.SparseCategoricalCrossentropy(from_logits=True), "fm_head": keras.losses.MSE}

    'compile model'
    model.compile(loss = losses, optimizer = 'Adam', metrics=['accuracy'])

    'separate validation targets for both branches'
    trainTargets = {
        "sm_head": smile_train,
        "fm_head": face_train
    }
    testTargets = {
        "sm_head": smile_test,
        "fm_head": face_test
    }

    'fit model'
    model.fit(images_train, trainTargets, validation_data = (images_test, testTargets), epochs = n_epochs, shuffle = True, verbose = 1)
    
    'save model'
    model.save("cnn_model")

