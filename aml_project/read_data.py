import tensorflow as tf
from tensorflow import keras
import os
import cv2 as cv


def read_labels(y_path, type, sz):
    y = list()
    #read labels
    data = open(y_path, 'r').read().splitlines()
    #read smile labels
    if(type == 's'):
        for d in data:
            y.append(int(d.split(" ")[0]))
    #read face labels
    elif(type == 'f'):
        for d in data:
            y.append([int(d.split(" ")[0]), int(d.split(" ")[1]), int(d.split(" ")[2])])
        fix_labels(y, sz)
    return y

#fixes the label scaling
def fix_labels(labels, new_size):
    dir = "dataset/GENKI-R2009a/Subsets/GENKI-SZSL/files/"
    i = 0
    for image in sorted(os.listdir(dir)):
        im = cv.imread(os.path.join(dir, image))
        dimensions = im.shape
        x_ratio = labels[i][0] / dimensions[0]
        y_ratio = labels[i][1] / dimensions[1]
        box_ratio = ((new_size / dimensions[0]) + (new_size / dimensions[1])) / 2
        labels[i][0] = int(new_size * x_ratio)
        labels[i][1] = int(new_size * y_ratio)
        labels[i][2] = int(labels[i][2] * box_ratio)
        i = i + 1



def read_data(X_path, y_path, type, new_size, normalize):
    y = read_labels(y_path, type, new_size)
    X = list()
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
        interpolation = 'bilinear',
        follow_links = False,
        crop_to_aspect_ratio = False
    )  
    #normalize images
    if(normalize):
        for images, labels in data:
            for im in images:
                im = im /255
    return data 