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
    '''images contains new images with boxes drawn on them
    images original contains the original images
    face_labels contains (xmin, xmax, ymin, ymax) values of each box
    smile_labels contains 1 if person in image is smiling, 0 if not'''
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
    'uncomment to show pixel values before and after normalization'
    #plotting.plot_pixels(images_original)
    'uncomment to show example set of labels and bounding boxes'
    #plotting.plot_face(images, smile_labels)
   
    'split data to train and test sets'
    images_train, images_test = cnn_model.split_data(images)
    face_train, face_test = cnn_model.split_data(face_labels)
    smile_train, smile_test = cnn_model.split_data(smile_labels)

    'uncomment to train multitask model'
    #cnn_model.multi_task_model(images_train, images_test, face_train, face_test, smile_train, smile_test, scaled_size, n_epochs)
    
    'load saved model'
    model = keras.models.load_model("cnn_model", compile=True)

    'predicts random image from images_original'
    prediction, image = cnn_model.predict(model, images_original)
    print((prediction[1][0][0]))
    print((prediction[1][0][1]))
    print((prediction[1][0][2]))
    print((prediction[1][0][3]))
   
    'visualize prediction'
    plotting.plot_prediction(prediction,image, scaled_size)

    # grad-cam or lime?
