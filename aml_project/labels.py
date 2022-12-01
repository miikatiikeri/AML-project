import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy
import matplotlib.image as mpimg

def labels():
    dir = "dataset/GENKI-R2009a/Subsets/GENKI-SZSL/files/"
    i = 0
    f = open("smile_labels.txt2", 'w')
    for image in sorted(os.listdir(dir)):
        i = i +1
        if i >= 320:
            im = mpimg.imread(os.path.join(dir, image))
            imgplot = plt.imshow(im)
            plt.show()
            input1 = input()
            if input1 == "2":
                f.write("0")
                f.write("\n")
            elif input1 ==  "1":
                f.write(input1)
                f.write("\n")
            else:
                f.write("0")
                f.write("\n")
    