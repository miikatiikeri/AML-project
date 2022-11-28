# general imports
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tensorflow as tf

# project imports
from read_data import read_data, read_labels
import plotting;
from plotting import ev_model
# global variables
# determines the size that images get scaled in read_data
# labels also get scaled accordingly
scaled_size = 180
# Normalizing images takes some time, set this to true when training model, otherwise keep false
normalize = False


def main():  # pragma: no cover

    # read data and labels s for smile, f for face
    # data_smile contains images and labels, labels are in array where N:th label corresponds to N:th image
    # 1 = smiling, 0 = not smiling
    data_smile = read_data("dataset/GENKI-R2009a/Subsets/GENKI-4K",
                           "dataset/GENKI-R2009a/Subsets/GENKI-4K/GENKI-4K_Labels.txt", 's', scaled_size, normalize)
    # data_face contains images and labels, labels are in array where N:th label corresponds to N:th image
    # each indice in labels contains subarray where l[i][0] = x cordinate of center of face, l[i][1] = y cordinate of center of face, l[i][2] = box size
    data_face = read_data("dataset/GENKI-R2009a/Subsets/GENKI-SZSL",
                          "dataset/GENKI-R2009a/Subsets/GENKI-SZSL/GENKI-SZSL_labels.txt", 'f', scaled_size, normalize)
    # normalize images

    # visualize data
    # plotting.plot_pixels(data_smile)
    # plotting.plot_smile(data_smile)
    # plotting.plot_face(data_face)

    # split dataset here or in training? always use set seeds

    train_ds, test_ds = tf.keras.utils.split_dataset(
        data_face, left_size=0.8, seed=1
    )

    train_smile_ds, test_smile_ds = tf.keras.utils.split_dataset(
        data_smile, left_size=0.8, seed=1
    )

    print(train_ds)
    print(test_ds)

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

    # load model

    # predict test data

    # visualize results


    # grad-cam or lime?
