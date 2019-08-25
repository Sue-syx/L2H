import os
import numpy as np
import heapq
import matplotlib.pyplot as plt
from PIL import Image
import IterativeQuantizer as hashGet


def main():
    # Feature & ImgRecord
    if os.path.exists("feature.npy") and os.path.exists("imgrecord.txt"):
        feature = np.load("feature.npy")
        imgrecord = []
        with open('imgrecord.txt', 'r') as f:
            for line in f:
                imgrecord.append(list(line.strip('\n').split(',')))
    else:
        path = "./flower_photos"
        feature, imgrecord = hashGet.featureGet(path)

    # HashBits
    if  os.path.exists("bits.npy"):
        bits = np.load("bits.npy")
    else:
        bits = hashGet.Hash2(feature)
        np.save("bits.npy", bits)

    testindex = np.random.randint(0, high=len(bits))
    bitest = bits[testindex]
    minHamming = hashGet.hammingDis(bits, bitest, 10)

    plt.figure()
    imgtest = Image.open(imgrecord[testindex][0])
    imgtest = imgtest.resize((224, 224))
    plt.subplot(3, 5, 3)
    plt.imshow(imgtest)
    count = 5
    for ind in minHamming:
        count += 1
        img = Image.open(imgrecord[ind][0])
        img = img.resize((224, 224))
        plt.subplot(3, 5, count)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
    plt.show()


if __name__ == '__main__':
    main()