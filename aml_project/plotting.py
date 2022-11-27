import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import tensorflow as tf

def plot_smile(data_smile):
    plt.figure(figsize=(10, 10))
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
def fix_cord(x, y, sz):
    sz = tf.math.divide(sz, 2)
    sz = tf.cast(sz, dtype=tf.int32)
    x = tf.math.subtract(x, sz)
    y = tf.math.subtract(y, sz)
    return x,y 
    

def plot_face(data_face):
    plt.figure(figsize=(10, 10))
    for images, labels in data_face.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            size = labels[i][2]
            x, y = fix_cord(labels[i][0],labels[i][1], size)
            ax.add_patch(Rectangle((x, y), size, size, linewidth=1, edgecolor='r', facecolor='none'))
            plt.axis("off")
    plt.show()

