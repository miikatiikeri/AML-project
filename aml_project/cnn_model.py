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
from keras import datasets, layers, models

def split_data(data):
    train_ds, test_ds = tf.keras.utils.split_dataset(
            data, left_size=0.8, seed=1
        )
    return train_ds, test_ds

# def extract(dataset):
#     images = list()
#     labels = list()
#     for i, l in dataset:
#         images.append(i)
#         labels.append(l)
#     return images, labels


def model(train_ds, test_ds, train_smile_ds, test_smile_ds):
    # model structure
    # model = models.Sequential()
    # model.add(layers.Conv2D(32, 3, activation='relu'))
    # model.add(layers.MaxPooling2D())
    # model.add(layers.Conv2D(32, 3, activation='relu'))
    # model.add(layers.MaxPooling2D())
    # model.add(layers.Conv2D(32, 3, activation='relu'))
    # model.add(layers.MaxPooling2D())
    # model.add(layers.Flatten())
    # model.add(layers.Dense(128, activation='relu'))
    # model.add(layers.Dense(2))
    
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

    # model summary
    #model.summary()

    #extract images and labels from dataset
    #train_images, train_labels = extract(train_ds)
    #test_images, test_labels = extract(test_ds)

    #train model
    model.fit(train_smile_ds, 
                    validation_data=(test_smile_ds), epochs = 3)
    
    model.save("cnn_model")
