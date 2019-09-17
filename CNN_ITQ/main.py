import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import IterativeQuantizer as hashGet


def main_flower(bit):
    # Feature & ImgRecord
    if os.path.exists("./result/flower/feature.npy") and os.path.exists("./result/flower/imgrecord.txt"):
        feature = np.load("./result/flower/feature.npy")
        imgrecord = []
        with open('./result/flower/imgrecord.txt', 'r') as f:
            for line in f:
                imgrecord.append(list(line.strip('\n').split(',')))
    else:
        path = "./flower_photos"
        feature, imgrecord = hashGet.featureGet_flower(path)

    # HashBits
    if os.path.exists("./result/flower/bits.npy"):
        bits = np.load("./result/flower/bits.npy")
    else:
        bits = hashGet.Hash(feature, bit)
        np.save("./result/flower/bits.npy", bits)

    testindex = np.random.randint(0, high=len(bits))
    bitest = bits[testindex]
    minHamming = hashGet.hammingDis(bits, bitest, 10)

    plt.figure()
    imgtest = Image.open(imgrecord[testindex][0])
    imgtest = imgtest.resize((224, 224))
    plt.subplot(3, 5, 3)
    plt.imshow(imgtest)
    plt.xticks([])
    plt.yticks([])
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


def main_cifar10(bit):
    # Feature & ImgRecord
    train_fea_path = "./result/cifar10/train_feature.npy"
    test_fea_path = "./result/cifar10/test_feature.npy"
    if os.path.exists(train_fea_path) and os.path.exists(test_fea_path):
        train_feature = np.load(train_fea_path)
        test_feature = np.load(test_fea_path)
    else:
        train_feature, test_feature = hashGet.featureGet_cifar10()

    # HashBits
    train_bits_path = "./result/cifar10/train_bits" + str(bit) + ".npy"
    test_bits_path = "./result/cifar10/test_bits" + str(bit) + ".npy"
    if os.path.exists(train_bits_path) and os.path.exists(test_bits_path):
        train_bits = np.load(train_bits_path)
        test_bits = np.load(test_bits_path)
    else:
        train_bits = hashGet.Hash(train_feature, bit)
        test_bits = hashGet.Hash(test_feature, bit)
        np.save(train_bits_path, train_bits)
        np.save(test_bits_path, test_bits)

    # compute mAP
    mAP = hashGet.compute_mAP(train_bits, test_bits)
    print(mAP)
    print("mAP : %f" % (mAP))


if __name__ == '__main__':
    bit_num = 12
    main_cifar10(48)
    # main_cifar10(24)
