import tensorflow as tf
from tensorflow import keras
import numpy as np

def split_data(data):
    train_ds, test_ds = tf.keras.utils.split_dataset(
            data, left_size=0.8, seed=1
        )
    return train_ds, test_ds

def predict(model, test_ds, test_smile_ds):
    # TODO fix predicting
    prediction_face = model.predict(test_ds)
    prediction_smile = model.predict(test_smile_ds)

    score_face = tf.nn.softmax(prediction_face[0])
    score_smile = tf.nn.softmax(prediction_smile[0])

    print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score_face)], 100 * np.max(score_face))
    )

    print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score_smile)], 100 * np.max(score_smile))
    )

def smile_model(train_smile_ds, test_smile_ds, n_epochs):
    num_classes = 2
    model = keras.Sequential([
        keras.layers.Conv2D(32, 3, activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(32, 3, activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(32, 3, activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(num_classes)
    ])
    #compile model
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    #train model
    model.fit(train_smile_ds, 
                    validation_data=(test_smile_ds), epochs = n_epochs)
    
    model.save("smile_model")

def multi_task_model(face_train, face_test, smile_train, smile_test, sz, n):

    input_shape = (sz,sz,3)
    input_layer = keras.layers.Input(input_shape)
    
    base_model = keras.layers.Conv2D(16, 3, padding='same', activation='relu', name='bm1')(input_layer)
    base_model = keras.layers.MaxPooling2D(name = 'bm2')(base_model)
    base_model = keras.layers.Conv2D(32, 3, padding='same', activation='relu', name='bm3')(base_model)
    base_model = keras.layers.MaxPooling2D(name = 'bm4')(base_model)
    base_model = keras.layers.Conv2D(64, 3, padding='same', activation='relu', name='bm5')(base_model)
    base_model = keras.layers.MaxPooling2D(name = 'bm6')(base_model)
    base_model = keras.layers.Flatten(name='bm7')(base_model)

    #smile branch
    smile_model = keras.layers.Dense(128, activation='relu', name='sm1')(base_model)
    smile_model = keras.layers.Dense(2, name='sm_head')(smile_model)

    #face branch
    face_model = keras.layers.Dense(128, activation = 'relu', name = 'fm1')(base_model)
    face_model = keras.layers.Dense(64, activation = 'relu', name = 'fm2')(face_model)
    face_model = keras.layers.Dense(32, activation = 'relu', name = 'fm3')(face_model)
    face_model = keras.layers.Dense(3, activation = 'sigmoid', name = 'fm_head')

    model = keras.Model(input_layer, outputs=[smile_model, face_model])


    losses = {"cl_head":tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), "bb_head":tf.keras.losses.MSE}

    model.compile(loss = losses, optimizer = 'Adam', metrics=['accuracy'])

    train_sets = {
        'sm_head': smile_train,
        'fm_head': face_train
    }
    test_sets = {
        'sm_head': smile_test,
        'fm_head': face_test
    }

    model.fit(train_sets, validation_data = test_sets, epochs = n)
