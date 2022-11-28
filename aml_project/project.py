#general imports
import tensorflow as tf
from tensorflow import keras
import tensorflow as tf


#project imports
from read_data import read_data, read_labels
import plotting;

#global variables
#determines the size that images get scaled in read_data
#labels also get scaled accordingly
scaled_size = 180

def main():  # pragma: no cover  
    
    #read data and labels s for smile, f for face
    #data_smile contains images and labels, labels are in array where N:th label corresponds to N:th image
    #1 = smiling, 0 = not smiling
    data_smile = read_data("dataset/GENKI-R2009a/Subsets/GENKI-4K", "dataset/GENKI-R2009a/Subsets/GENKI-4K/GENKI-4K_Labels.txt", 's', scaled_size)
    #data_face contains images and labels, labels are in array where N:th label corresponds to N:th image
    #each indice in labels contains subarray where l[i][0] = x cordinate of center of face, l[i][1] = y cordinate of center of face, l[i][2] = box size
    data_face = read_data("dataset/GENKI-R2009a/Subsets/GENKI-SZSL", "dataset/GENKI-R2009a/Subsets/GENKI-SZSL/GENKI-SZSL_labels.txt", 'f', scaled_size)
    
    #visualize data
    plotting.plot_smile(data_smile)
    plotting.plot_face(data_face)

    
    #split dataset here or in training? always use set seeds 
    train_ds, test_ds = tf.keras.utils.split_dataset(
        data_face, left_size=0.8, seed=1
    )
    
    print(train_ds)
    print(test_ds)

    # TODO read labels and split them accordingly
    read_labels()
    train_labels, test_labels = tf.keras.utils.split_dataset(

    )

    #train model and save it
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

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    history = model.fit(train_ds, train_labels, epochs=10,
                        validation_data=(test_ds, test_labels))

    model.summary()
    #load model


    #predict test data

    #visualize results

    #grad-cam or lime?
   
