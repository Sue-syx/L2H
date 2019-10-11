import numpy as np
import cv2
import glob
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
import datetime
import os


def calc_hist(flow):
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=1)
    q1 = ((0 < ang) & (ang <= 45)).sum()
    q2 = ((45 < ang) & (ang <= 90)).sum()
    q3 = ((90 < ang) & (ang <= 135)).sum()
    q4 = ((135 < ang) & (ang <= 180)).sum()
    q5 = ((180 < ang) & (ang <= 225)).sum()
    q6 = ((225 <= ang) & (ang <= 270)).sum()
    q7 = ((270 < ang) & (ang <= 315)).sum()
    q8 = ((315 < ang) & (ang <= 360)).sum()
    hist = [q1, q2, q3, q4, q5, q6, q7, q8]
    return hist


def dataSet():
    bins_n = 10
    for root, dirs, files in os.walk("UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train"):
        video_hist = []
        for dir_ in dirs:
            for subroot, subdir, imgs in os.walk(os.path.join(root, dir_)):
                # imgs 代表一个视频
                for i, img in enumerate(imgs):
                    if i == len(imgs)-1:
                        break
                    prev_frm = cv2.imread(os.path.join(subroot, img), cv2.IMREAD_UNCHANGED)
                    prev_frm = cv2.resize(prev_frm, (240, 160))
                    tmp_frm = cv2.imread(os.path.join(subroot, imgs[i+1]), cv2.IMREAD_UNCHANGED)
                    tmp_frm = cv2.resize(tmp_frm, (240, 160))
                    flow = cv2.calcOpticalFlowFarneback(prev_frm, tmp_frm, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    bins = np.hsplit(flow, bins_n)
                    out_bins = []
                    for b in bins:
                        out_bins.append(np.vsplit(b, bins_n))
                    frame_hist = []
                    for col in out_bins:
                        for block in col:
                            frame_hist.append(calc_hist(block))
                    video_hist.append(np.array(frame_hist))
                # average per frame
                sum_desc = video_hist[0]
                for i in range(1, len(video_hist)):
                    sum_desc = sum_desc + video_hist[i]
                ave = np.asarray(sum_desc / len(video_hist))

                # max per bin
                maxx = np.array(np.amax(video_hist, 0))

                a_desc = []
                a_desc.append(np.asarray(ave_desc, dtype=np.uint8).ravel())
                max_desc = np.asarray(maxx)
                m_desc = np.asarray(max_desc, dtype=np.uint8).ravel()

                # a_desc, m_desc 特征


dataSet()
