import tensorflow as tf
from tensorflow import keras

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
    return y

def read_data(X_path, y_path, type):
    y = read_labels(y_path, type)
    data = keras.utils.image_dataset_from_directory(
        X_path,
        labels = y,
        label_mode = 'int',
        class_names = None,
        color_mode = 'rgb',
        batch_size = 32,
        image_size = [180, 180],
        shuffle = True,
        seed = 1,
        validation_split = None,
        subset = None,
        interpolation = 'bilinear',
        follow_links = False,
        crop_to_aspect_ratio = False
    )  
    return data 