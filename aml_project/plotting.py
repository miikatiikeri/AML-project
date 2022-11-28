import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import tensorflow as tf
import cv2 as cv

def plot_smile(data_smile):
    plt.figure(figsize=(10, 10))
    plt.suptitle("Example of Genki-4k dataset")
    for images, labels in data_smile.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            if(labels[i] == 1):
                plt.title("smiling")
            else:
                plt.title("not smiling")
            plt.axis("off")
    plt.show()

#fixes cordinates for box drawning
#objects x, y and z are tensors not numbers, that's why use of tf methods
def fix_cord(x, y, sz):
    sz = tf.math.divide(sz, 2)
    sz = tf.cast(sz, dtype=tf.int32)
    x = tf.math.subtract(x, sz)
    y = tf.math.subtract(y, sz)
    return x,y 

#plots pixel values before and after standardization
def plot_pixels(data):
    fig, axs = plt.subplots(3, 2, figsize=(10, 10))
    for images, labels in data.take(1):
        for i in range(3):
            im = images[i].numpy().astype("uint8")
            im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
            fig1 = axs[i,0].matshow(im_gray, cmap = 'rainbow')
            axs[i,0].axis("off")
            fig.colorbar(fig1, ax = axs[i,0])
            new_im = tf.image.per_image_standardization(images[i]).numpy().astype("float32")
            new_im_gray = cv.cvtColor(new_im, cv.COLOR_BGR2GRAY)
            fig2 = axs[i,1].matshow(new_im_gray, cmap = 'rainbow')
            axs[i,1].axis("off")
            fig.colorbar(fig2, ax = axs[i,1])
    fig.suptitle("Greyscale pixel values before and after standardization")
    plt.show()
        

def plot_face(data_face):
    plt.figure(figsize=(10, 10))
    plt.suptitle("Example of Genki-szsl dataset")
    for images, labels in data_face.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            size = labels[i][2]
            x, y = fix_cord(labels[i][0],labels[i][1], size)
            ax.add_patch(Rectangle((x, y), size, size, linewidth=1, edgecolor='r', facecolor='none'))
            plt.axis("off")

    plt.show()

