import tensorflow as tf
from tensorflow import keras
import os
import cv2 as cv
import platform

def read_labels(y_path, type, scaled_size, user):
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
        #fix_labels(y, scaled_size, user)
    return y

#fixes the label scaling
def fix_labels(labels, scaled_size, user):
    if (platform.system() == "Linux" or "Darwin") and user != True:
        dir = "../dataset/GENKI-R2009a/Subsets/GENKI-SZSL/files/"
    else:
        dir = "dataset/GENKI-R2009a/Subsets/GENKI-SZSL/files/"
    i = 0
    for image in sorted(os.listdir(dir)):
        im = cv.imread(os.path.join(dir, image))
        dimensions = im.shape
        x_ratio = labels[i][0] / dimensions[0]
        y_ratio = labels[i][1] / dimensions[1]
        box_ratio = ((scaled_size / dimensions[0]) + (scaled_size / dimensions[1])) / 2
        labels[i][0] = int(scaled_size * x_ratio)
        labels[i][1] = int(scaled_size * y_ratio)
        labels[i][2] = int(labels[i][2] * box_ratio)
        i = i + 1

def read_images(user):
    images = list()
    if (platform.system() == "Linux" or "Darwin") and user != True:
        dir = "../dataset/GENKI-R2009a/Subsets/GENKI-SZSL/files/"
    else:
        dir = "dataset/GENKI-R2009a/Subsets/GENKI-SZSL/files/"
    for i in sorted(os.listdir(dir)):
        img = cv.imread(os.path.join(dir, i))
        images.append(img)
    return images

def read_data(image_path, face_path, smile_path, scaled_size, normalize, user):
    face_labels = read_labels(face_path, "f", scaled_size, user)
    smile_labels = read_labels(smile_path, "s", scaled_size, user)
    images = read_images(user)
    #normalize images
    # if(normalize):
    #     for images, labels in data:
    #         for im in images:
    #             im = im /255
    return images, face_labels, smile_labels