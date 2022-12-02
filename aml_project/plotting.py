import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import tensorflow as tf
import cv2 as cv
import random

#fixes cordinates for box drawning
#objects x, y and z are tensors not numbers, that's why use of tf methods
def fix_cord(x, y, sz):
    sz = tf.math.divide(sz, 2)
    sz = tf.cast(sz, dtype=tf.int32)
    x = tf.math.subtract(x, sz)
    y = tf.math.subtract(y, sz)
    return x,y 

#plots pixel values before and after standardization
def plot_pixels(images):
    fig, axs = plt.subplots(3, 2, figsize=(10, 10))
    for i in range(3):
        im = images[i]
        fig1 = axs[i,0].matshow(im, cmap = 'rainbow')
        axs[i,0].axis("off")
        fig.colorbar(fig1, ax = axs[i,0])
        new_im = images[i]/ 255.0
        fig2 = axs[i,1].matshow(new_im, cmap = 'rainbow')
        axs[i,1].axis("off")
        fig.colorbar(fig2, ax = axs[i,1])
    fig.suptitle("Greyscale pixel values before and after standardization")
    plt.show()
        

def plot_face(images, face_labels, smile_labels):
    plt.figure(figsize=(10, 10))
    plt.suptitle("Example of Genki-szsl dataset")
    random.seed(1)
    for i in range(9):
        r = random.randint(1,1000)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[r])
        size = face_labels[r][2]
        x, y = fix_cord(face_labels[r][0],face_labels[r][1], size)
        ax.add_patch(Rectangle((x, y), size, size, linewidth=1, edgecolor='r', facecolor='none'))
        if smile_labels[r] == 1:
            ax.set_title("smiling")
        else:
            ax.set_title("not smiling")
        plt.axis("off")
    plt.show()

def ev_model(face):
    plt.plot(face.face['accuracy'], label='accuracy')
    plt.plot(face.face['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()
