from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model

from keras.datasets import cifar10
from keras.models import load_model

import os
import numpy as np
from sklearn.decomposition import PCA


def featureGet_flower(path):
    feature = []
    imgrecord = []
    base_model = VGG16(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
    for root, dirs, files in os.walk(path):
        for dir_ in dirs:
            dirpath = os.path.join(root, dir_)
            for root_, dirs_, files_ in os.walk(dirpath):
                for img in files_:
                    img_path = os.path.join(root_, img)
                    img = image.load_img(img_path, target_size=(224, 224))
                    x = image.img_to_array(img)
                    x = np.expand_dims(x, axis=0)
                    x = preprocess_input(x)
                    featureX = model.predict(x)
                    if feature == []:
                        feature = featureX
                    else:
                        feature = np.concatenate((feature, featureX), axis=0)
                    imgrecord.append(img_path)
                    print(img_path)
    np.save('./result/flower/feature.npy', feature)
    with open('./result/flower/imgrecord.txt', 'w') as f:
        f.write('\n'.join(imgrecord))
    return feature, imgrecord


def featureGet_cifar10():
    base_model = load_model('./model/vgg19.h5')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # data preprocessing
    x_train[:, :, :, 0] = (x_train[:, :, :, 0] - 123.680)
    x_train[:, :, :, 1] = (x_train[:, :, :, 1] - 116.779)
    x_train[:, :, :, 2] = (x_train[:, :, :, 2] - 103.939)
    x_test[:, :, :, 0] = (x_test[:, :, :, 0] - 123.680)
    x_test[:, :, :, 1] = (x_test[:, :, :, 1] - 116.779)
    x_test[:, :, :, 2] = (x_test[:, :, :, 2] - 103.939)

    train_fea = model.predict(x_train)
    test_fea = model.predict(x_test)
    np.save('./result/cifar10/train_feature.npy', train_fea)
    np.save('./result/cifar10/test_feature.npy', test_fea)

    return train_fea, test_fea


# IterativeQuantizer
def trainITQ(feature, bits):
    r = np.random.random(size=(bits, bits))
    U, _, _ = np.linalg.svd(r)
    r = U
    for i in range(50):
        z = np.dot(feature, r)
        ux = -1 * np.ones([z.shape[0], z.shape[1]])
        ux[z > 0] = 1
        c = np.dot(ux.T, feature)
        ub, _, ua = np.linalg.svd(c)
        r = np.dot(ua.T, ub.T)
    return r


# hashGet
def Hash(feature, bits):
    # PCA
    pca = PCA(n_components=bits)
    pca_feature = pca.fit_transform(feature)
    # HashEncode
    (_, nbits) = pca_feature.shape
    R = trainITQ(pca_feature, nbits)
    B = np.dot(pca_feature, R)
    B[B > 0] = 1
    B[B <= 0] = 0
    return B


def hammingDis(B, b, num):
    # hammingDists
    hamming = []
    for bitemp in B:
        smstr = np.nonzero(b - bitemp)
        sm = np.shape(smstr[0])[0]
        hamming.append(sm)
    # minNum
    minNum = []
    for i in range(num):
        minind = hamming.index(min(hamming))
        minNum.append(minind)
        hamming[minind] = max(hamming)
    return minNum


def compute_mAP(trn_binary, tst_binary):
    (_, trn_label), (_, tst_label) = cifar10.load_data()
    AP = np.zeros(tst_binary.shape[0])
    Ns = np.arange(1, trn_binary.shape[0] + 1)
    for i in range(tst_binary.shape[0]):
        print('Query ', i + 1)
        query_label = tst_label[i]
        query_binary = tst_binary[i, :]
        query_result = np.count_nonzero(query_binary != trn_binary, axis=1)
        sort_indices = np.argsort(query_result)
        buffer_yes = np.equal(query_label, trn_label[sort_indices]).astype(int)
        P = np.cumsum(buffer_yes) / Ns
        AP[i] = np.sum(P * buffer_yes.T) / sum(buffer_yes)
    mAP = np.mean(AP)
    return mAP