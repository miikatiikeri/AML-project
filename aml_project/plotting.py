import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import tensorflow as tf
import cv2 as cv
import random

'displays heatmap over images showing pixel values before and after normalisation'
def plot_pixels(images):
    fig, axs = plt.subplots(3, 2, figsize=(10, 10))
    for i in range(3):
        im = images[i]
        im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        fig1 = axs[i,0].matshow(im, cmap = 'rainbow')
        axs[i,0].axis("off")
        fig.colorbar(fig1, ax = axs[i,0])
        new_im = im/ 255.0
        fig2 = axs[i,1].matshow(new_im, cmap = 'rainbow')
        axs[i,1].axis("off")
        fig.colorbar(fig2, ax = axs[i,1])
    fig.suptitle("Greyscale pixel values before and after standardization")
    plt.show()
        
'plot faces with their corresponding labels'
def plot_face(images, smile_labels):
    plt.figure(figsize=(10, 10))
    plt.suptitle("Example of Genki-szsl dataset")
    arr = [3, 9, 17, 18, 19, 20, 21, 22, 23]
    j = 0
    for i in arr:
        ax = plt.subplot(3, 3, j+ 1)
        im = cv.cvtColor(images[i], cv.COLOR_BGR2RGB)
        plt.imshow(im)
        if smile_labels[i] == 1:
            ax.set_title("smiling")
        else:
            ax.set_title("not smiling")
        plt.axis("off")
        j = j + 1
    plt.show()

def ev_model(face):
    plt.plot(face.face['accuracy'], label='accuracy')
    plt.plot(face.face['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()
