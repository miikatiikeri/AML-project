# general imports
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import platform

# project imports
from read_data import read_data
import plotting
import cnn_model

'global variables'
'determines size that images scale to'
scaled_size = 180
'number of epoch for training the model'
n_epochs = 100
'quick fix for dataread'
user = True


def main():  # pragma: no cover

    'platform detection to fix read issues between os'
    if (platform.system() == "Linux" or "Darwin") and user != True:
      print("using unix read")
      images,images_original, face_labels, smile_labels = read_data(
                        "../dataset/GENKI-R2009a/Subsets/GENKI-SZSL/GENKI-SZSL_labels.txt",
                        "../dataset/GENKI-R2009a/Subsets/GENKI-SZSL/smile_labels.txt",
                        scaled_size, user)
    else:
      print("using windows read")
      images, images_original, face_labels, smile_labels = read_data(
                        "dataset/GENKI-R2009a/Subsets/GENKI-SZSL/GENKI-SZSL_labels.txt",
                        "dataset/GENKI-R2009a/Subsets/GENKI-SZSL/smile_labels.txt", 
                        scaled_size, user)
    
    'visualize data, uncomment to plot'
    #plotting.plot_pixels(images_original)
    #plotting.plot_face(images, face_labels, smile_labels)
   
    'split data to train and test sets'
    images_train, images_test = cnn_model.split_data(images)
    face_train, face_test = cnn_model.split_data(face_labels)
    smile_train, smile_test = cnn_model.split_data(smile_labels)

    'uncomment to train multitask model'
    #cnn_model.multi_task_model(images_train, images_test, face_train, face_test, smile_train, smile_test, scaled_size, n_epochs)
    
    'load saved model'
    model = keras.models.load_model("cnn_model", compile=True)
  

    # predict test data
    #print(test_ds)
    #print(test_smile_ds)

    #predict code here
    #cnn_model.predict(model, test_ds, test_smile_ds)
    
    # visualize results


    # grad-cam or lime?
