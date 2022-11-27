import tensorflow as tf
from tensorflow import keras
import os
import cv2 as cv
from pathlib import Path

new_size = 180

def read_labels(y_path, type):
    y = list()
    #read labels
    data = open(y_path, 'r').read().splitlines()
    if(type == 's'):
        for d in data:
            y.append(int(d.split(" ")[0]))
    elif(type == 'f'):
        for d in data:
            y.append([int(d.split(" ")[0]), int(d.split(" ")[1]), int(d.split(" ")[2])])
        #uncomment to scale labels, needs some bug fixing
        #fix_labels(y)
    return y

def fix_labels(labels):
    dir = "dataset/GENKI-R2009a/Subsets/GENKI-SZSL/files/"
    i = 0
    for image in os.listdir(dir):
        im = cv.imread(os.path.join(dir, image))
        dimensions = im.shape
        y_ratio = new_size / dimensions[0]
        x_ratio = new_size / dimensions[1]
        box_ratio = (x_ratio + y_ratio) / 2
        labels[i][0] = int(labels[i][0] * x_ratio)
        labels[i][1] = int(labels[i][1] * y_ratio)
        labels[i][2] = int(labels[i][2] * box_ratio)
        i = i + 1



def read_data(X_path, y_path, type):
    y = read_labels(y_path, type)
    data = keras.utils.image_dataset_from_directory(
        X_path,
        labels = y,
        label_mode = 'int',
        class_names = None,
        color_mode = 'rgb',
        batch_size = 32,
        image_size = [new_size, new_size],
        shuffle = True,
        seed = 1,
        validation_split = None,
        subset = None,
        interpolation = 'bilinear',
        follow_links = False,
        crop_to_aspect_ratio = False
    )  
    return data 