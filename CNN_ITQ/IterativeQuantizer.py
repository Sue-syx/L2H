import os
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import numpy as np
from sklearn.decomposition import PCA


def featureGet(path):
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
    np.save("feature.npy", feature)
    with open('imgrecord.txt', 'w') as f:
        f.write('\n'.join(imgrecord))

    return feature, imgrecord


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
def Hash2(feature):
    # PCA
    pca = PCA(n_components=64)
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
    minNum=[]
    for i in range(num):
        minind = hamming.index(min(hamming))
        minNum.append(minind)
        hamming[minind] = max(hamming)
    return minNum