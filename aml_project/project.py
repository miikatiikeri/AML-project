# general imports
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

import platform

# project imports
from read_data import read_data, read_labels
import plotting;
import labels
import cnn_model
# global variables
# determines the size that images get scaled in read_data
# labels also get scaled accordingly
scaled_size = 32
# Normalizing images takes some time, set this to true when training model, otherwise keep false
normalize = True
# number of epochs for training, set higher when actually training model
n_epochs = 3


def main():  # pragma: no cover

    # platform detection to fix read issues between os
    if platform.system() == "Linux" or "Darwin":
      print("using unix read")
      data_smile = read_data("../dataset/GENKI-R2009a/Subsets/GENKI-4K",
                            "../dataset/GENKI-R2009a/Subsets/GENKI-4K/GENKI-4K_Labels.txt", 's', scaled_size, normalize)
      data_face = read_data("../dataset/GENKI-R2009a/Subsets/GENKI-SZSL",
                            "../dataset/GENKI-R2009a/Subsets/GENKI-SZSL/GENKI-SZSL_labels.txt", 'f', scaled_size, normalize)
    else:
      print("using windows read")
      # read data and labels s for smile, f for face
      # data_smile contains images and labels, labels are in array where N:th label corresponds to N:th image
      # 1 = smiling, 0 = not smiling
      data_smile = read_data("dataset/GENKI-R2009a/Subsets/GENKI-4K",
                             "dataset/GENKI-R2009a/Subsets/GENKI-4K/GENKI-4K_Labels.txt", 's', scaled_size, normalize)
      # data_face contains images and labels, labels are in array where N:th label corresponds to N:th image
      # each indice in labels contains subarray where l[i][0] = x cordinate of center of face, l[i][1] = y cordinate of center of face, l[i][2] = box size
      data_face = read_data("dataset/GENKI-R2009a/Subsets/GENKI-SZSL",
                            "dataset/GENKI-R2009a/Subsets/GENKI-SZSL/GENKI-SZSL_labels.txt", 'f', scaled_size, normalize)

  
    
    # visualize data
    #plotting.plot_pixels(data_smile)
    # plotting.plot_smile(data_smile)
    # plotting.plot_face(data_face)

    # split dataset here or in training? always use set seeds
    # moved data splitting to different function in cnn_model
    #train_ds, test_ds = cnn_model.split_data(data_face)
    #train_smile_ds, test_smile_ds = cnn_model.split_data(data_smile)
   
    # smile detection model
    #cnn_model.smile_model(train_smile_ds, test_smile_ds, n_epochs)
    # load model
    #smile_model = keras.models.load_model("smile_model", compile=True)

    #multi task model
    #cnn_model.multi_task_model(train_ds, test_ds, train_smile_ds, test_smile_ds, scaled_size,n_epochs)
    # predict test data
    #print(test_ds)
    #print(test_smile_ds)

    #predict code here
    #cnn_model.predict(model, test_ds, test_smile_ds)
    
    # visualize results


    # grad-cam or lime?
