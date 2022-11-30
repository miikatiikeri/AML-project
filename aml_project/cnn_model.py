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

def multi_task_model(face_train, face_test, smile_train, smile_test):
    return 1