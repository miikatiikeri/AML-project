import tensorflow as tf
from tensorflow import keras
import os
import cv2 as cv
import platform
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

'''reads labels
type = s reads smile labels
type = f reads face labels'''
def read_labels(y_path, type):
    data = open(y_path, 'r').read().splitlines()
    i = 0
    if(type == 's'):
        y = []
        for d in data:
            y.append(int(d.split(" ")[0]))
            i = i+1
    elif(type == 'f'):
        y = []
        for d in data:
            y.append([int(d.split(" ")[0]), int(d.split(" ")[1]), int(d.split(" ")[2])])
            i = i+1
    return np.array(y)

'''fixes face label locations to match new image size'''
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
    return labels

'''calculates xmin, xmax, ymin, ymax
from face labels'''
def transform_labels(labels):
    new_labels = []
    for i in labels:
        xmin = int(i[0] - i[2]/2)
        xmax = int(i[0] + i[2]/2)
        ymin = int(i[1] - i[2]/2)
        ymax = int(i[1] + i[2]/2)
        new_labels.append((xmin, xmax, ymin, ymax))
    return np.asarray(new_labels)

'''reads images to np array
type = b reads boxed images
type = o reads original images'''
def read_images(user, scaled_size, type):
    images = []
    if (platform.system() == "Linux" or "Darwin") and user != True:
        if(type == 'b'):
            dir = "../dataset/GENKI-R2009a/box_images/"
        if(type == 'o'):
            dir = "../dataset/GENKI-R2009a/Subsets/GENKI-SZSL/files"
    else:
        if(type == 'b'):
            dir = "dataset/GENKI-R2009a/box_images/"
        if(type == 'o'):
            dir = "dataset/GENKI-R2009a/Subsets/GENKI-SZSL/files"
    for i in sorted(os.listdir(dir)):
        img = cv.imread(os.path.join(dir, i))
        img = cv.resize(img, (scaled_size, scaled_size))
        x = np.asarray(img)
        images.append(x)
    return np.asarray(images)

'''makes images with bounding boxes
and stores them to box_images folder'''
def box_images(labels, images):
    j = 0
    for i in images:
        print(labels[0])
        i = cv.rectangle(i,(labels[j][0],labels[j][3]),(labels[j][1],labels[j][2]),(244,0,0),3)
        cv.imwrite("dataset/GENKI-R2009a/box_images/"+str(j)+".jpg", i)
        j = j + 1
        
'reads images and labels'
def read_data(face_path, smile_path, scaled_size, user):
    'read face labels'
    face_labels = read_labels(face_path, "f")
    
    'scale them to new image size'
    fix_labels(face_labels, scaled_size, user)
    
    'transform them for'
    face_labels = transform_labels(face_labels)
    
    'read non boxed images'
    images_original = read_images(user, scaled_size,"o")
    
    'uncomment to make new boxed images if needed'
    #box_images(face_labels, images_original)
    
    'read boxed images'
    images = read_images(user, scaled_size, "b")
    
    'read smile labels'
    smile_labels = read_labels(smile_path, "s")
    
    return images, images_original, face_labels, smile_labels