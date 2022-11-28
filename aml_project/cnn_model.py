#NETWORK
#layers = [

#imageInputLayer([64, 64, 3]);

#convolution2dLayer([5, 5], 32, 'Padding','same');
#batchNormalizationLayer;
#reluLayer;
#maxPooling2dLayer(2, 'Stride', 2);

#convolution2dLayer([5, 5], 32, 'Padding','same');
#batchNormalizationLayer;
#reluLayer;
#maxPooling2dLayer(2, 'Stride', 2);

#convolution2dLayer([5, 5], 32, 'Padding','same');
#batchNormalizationLayer;
#reluLayer;
#maxPooling2dLayer(2, 'Stride', 2);

#fullyConnectedLayer(128);
#reluLayer;

#fullyConnectedLayer(2);
#softmaxLayer;
#classificationLayer;
#];

# TRAINING OPTINS
#options = trainingOptions('sgdm', ...
#    'MaxEpochs',40, ...
#    'ValidationData',imdsValidation, ...
#    'ValidationFrequency',30, ...
#    'Verbose',false, ...
#    'Plots','training-progress');


# TTRAIN YOUR NETWORK
#net = trainNetwork(imdsTrain,layers,options);

import tensorflow as tf
from tensorflow import keras
from read_data import read_data, read_labels
from plotting import ev_model

def split_data(data):
    train_ds, test_ds = tf.keras.utils.split_dataset(
            data, left_size=0.8, seed=1
        )
    return train_ds, test_ds

def model(train_ds, test_ds, train_smile_ds, test_smile_ds):
    # read labels for face and smile
    labels_face = read_labels("dataset/GENKI-R2009a/Subsets/GENKI-4K/GENKI-4K_Labels.txt", "f")
    labels_smile = read_labels("dataset/GENKI-R2009a/Subsets/GENKI-SZSL/GENKI-SZSL_labels.txt", "s")

    # split face labels
    train_labels, test_labels = tf.keras.utils.split_dataset(
        labels_face, left_size=0.8, seed=1
    )   

    # split smile labels
    train_smile_labels, test_smile_labels = tf.keras.utils.split_dataset(
        labels_smile, left_size=0.8, seed=1
    )

    # train model and save it
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    # model summary
    model.summary()

    # compile model
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # train model with face data
    face = model.fit(train_ds, train_labels, epochs=10,
                 validation_data=(test_ds, test_labels))

    """
    smile = model.fit(train_smile_ds, train_smile_labels, epochs=10,
                    validation_data=(test_smile_ds, test_smile_labels))
    """

    # evaluate the model
    ev_model(face)
    test_loss, test_acc = model.evaluate(test_ds, test_labels, verbose=2)
