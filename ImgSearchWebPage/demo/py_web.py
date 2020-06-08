# coding=gbk

import flask
import os
import io
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import models.alexnet as alexnet


#######################################################

app = flask.Flask(__name__)
app.jinja_env.auto_reload = True
app.config['TEMPLATES_AUTO_RELOAD'] = True

########################################################


def load_model(code_length, device):
    path = './checkpoints/adsh_nuswide_checkpoints/ADSH_NUSWIDE_48bits.pt'
    M = torch.load(path, map_location=lambda storage, loc: storage)
    B = M['rB']
    model = alexnet.load_model(code_length).to(device)
    model.load_state_dict(M['model'])
    model.eval()
    # load imgrecord
    root = '../datasets/NUS-WIDE'
    img_txt = 'database_img.txt'
    img_txt_path = os.path.join(root, img_txt)
    with open(img_txt_path, 'r') as f:
        imgrecord = np.array([i.strip() for i in f])
    return B, imgrecord, model


def binary_output(query, model, device, bit_num=48):
    query_transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input = query_transform(query).to(device)
    with torch.no_grad():
        input = Variable(input)
        hash_code = model(torch.unsqueeze(input, 0)).sign().cpu()
    model.train()
    return hash_code


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


def ImageSearch(image, device, bit_num=48, query_num=10):
    [B, imgrecord, model] = load_model(bit_num, device)
    rB = B.detach().numpy()
    query_code = binary_output(image, model, device, bit_num).numpy()
    minHamming = hammingDis(rB, query_code, query_num)
    return [minHamming, imgrecord]

###########################  Web-page #######################################

# 访问首页时的调用函数
@app.route('/')
def index_page():  # flask库要求'web_page.html'必须在templates文件夹下
    return flask.render_template('web_page.html')


# 获取用户输入
@app.route('/predict', methods=['POST'])
def upload_file():
    img_bytes = flask.request.files['input_image'].read()
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    [minHamming, imgrecord] = ImageSearch(img, device=torch.device("cpu"), bit_num=48, query_num=10)
    res = []
    for ind in minHamming:
        img_path = imgrecord[ind]
        res.append(img_path)
    return flask.jsonify(predict_images=res)


if __name__ == '__main__':
    app.run("127.0.0.1", port=5000)

################################ 可视化验证 ########################################

# import random
#
# device = torch.device("cpu")
# root = '../datasets/NUS-WIDE'
# img_txt = 'database_img.txt'
# img_txt_path = os.path.join(root, img_txt)
# with open(img_txt_path, 'r') as f:
#     imgrecord = np.array([i.strip() for i in f])
# index = random.randint(1, 190000)
# img = Image.open(os.path.join(root, imgrecord[index])).convert('RGB')
#
# [minHamming, imgrecord] = ImageSearch(img, device, bit_num=48, query_num=5)
#
# plt.figure()
# plt.subplot(3, 5, 3)
# plt.imshow(img)
# plt.xticks([])
# plt.yticks([])
# count = 5
# for ind in minHamming:
#     count += 1
#     img = Image.open(os.path.join(root, imgrecord[ind])).convert('RGB')
#     plt.subplot(3, 5, count)
#     plt.imshow(img)
#     plt.xticks([])
#     plt.yticks([])
# plt.show()