import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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

def plot_face(data_face):
    plt.figure(figsize=(10, 10))
    for images, labels in data_face.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            ax.add_patch(Rectangle((labels[i][0],labels[i][1]), labels[i][2], labels[i][2], linewidth=1, edgecolor='r', facecolor='none'))
            plt.axis("off")
    plt.show()