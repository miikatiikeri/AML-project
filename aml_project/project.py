# general imports
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

import platform

# project imports
from read_data import read_data
import plotting
import cnn_model

# global variables
# determines the size that images get scaled in read_data
# labels also get scaled accordingly
scaled_size = 180
# Normalizing images takes some time, set this to true when training model, otherwise keep false
normalize = True
# number of epochs for training, set higher when actually training model
n_epochs = 3
# quick fix for dataread
user = False


def main():  # pragma: no cover

    #platform detection to fix read issues between os
    if (platform.system() == "Linux" or "Darwin") and user != True:
      print("using unix read")
      images, face_labels, smile_labels = read_data(
                        "../dataset/GENKI-R2009a/Subsets/GENKI-SZSL/GENKI-SZSL_labels.txt",
                        "../dataset/GENKI-R2009a/Subsets/GENKI-SZSL/smile_labels.txt",
                        scaled_size, normalize, user)
    else:
      print("using windows read")
      # read data and labels s for smile, f for face
      # smile_labels contains images and labels, labels are in array where N:th label corresponds to N:th image
      # 1 = smiling, 0 = not smiling
      # face_labels contains images and labels, labels are in array where N:th label corresponds to N:th image
      # each indice in labels contains subarray where l[i][0] = x cordinate of center of face, l[i][1] = y cordinate of center of face, l[i][2] = box size
      images, face_labels, smile_labels = read_data(
                        "dataset/GENKI-R2009a/Subsets/GENKI-SZSL/GENKI-SZSL_labels.txt",
                        "dataset/GENKI-R2009a/Subsets/GENKI-SZSL/smile_labels.txt", 
                        scaled_size, normalize, user)
    
    # visualize data
    #plotting.plot_pixels(images)
    #plotting.plot_face(images, face_labels, smile_labels)

    #split data
    images_train, images_test = cnn_model.split_data(images)
    face_train, face_test = cnn_model.split_data(face_labels)
    smile_train, smile_test = cnn_model.split_data(smile_labels)
   
    # smile detection model
    #cnn_model.smile_model(train_smile_ds, test_smile_ds, n_epochs)
    # load model
    #smile_model = keras.models.load_model("smile_model", compile=True)

    #multi task model
    cnn_model.multi_task_model(images_train, images_test, face_train, face_test, smile_train, smile_test, scaled_size, n_epochs)
    # predict test data
    #print(test_ds)
    #print(test_smile_ds)

    #predict code here
    #cnn_model.predict(model, test_ds, test_smile_ds)
    
    # visualize results


    # grad-cam or lime?
