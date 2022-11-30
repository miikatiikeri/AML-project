import tensorflow as tf
from tensorflow import keras
from read_data import read_data, read_labels
from plotting import ev_model
from keras import datasets, layers, models

def split_data(data):
    train_ds, test_ds = tf.keras.utils.split_dataset(
            data, left_size=0.8, seed=1
        )
    return train_ds, test_ds



def model(train_ds, test_ds, train_smile_ds, test_smile_ds):
    
    num_classes = 2
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes)
    ])

   

    #compile model
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    #train model
    model.fit(train_smile_ds, 
                    validation_data=(test_smile_ds), epochs = 3)
    
    model.save("cnn_model")
