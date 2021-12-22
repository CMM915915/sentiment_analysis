#!/usr/bin/python
# -*- coding: utf-8 -*-
# author:mengmeng
# 日期:2020/3/27
import cv2
import config as cfg
import initialization as ini
from RECG import demo as Recg
from DETECT import demo as Detect
from DRAW import painting as pt
from multiprocessing import Process
import multiprocessing
from network.network import East
from RECG.network import crnn
import torch
import config
import numpy as np
from RECG.utils import alphabets
nclass = len(alphabets.alphabet) + 1


def cv_imread(filePath):
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)
    return cv_img


# 传.
img_npy = cv2.imread("D:\\PythonProjects\\TextDetection_V1.1\\image_02.png")

list_npys=[]
# 初始化网络
def  detection(i,q):
    # 初始化检测网络
    east = East()
    detect_module = east.east_network()
    detect_module.load_weights(config.detect_module_path)
    detect = Detect.Detection(img_npy, detect_module, cfg.slice_size, cfg.step_size, cfg.max_predict_size,
                              cfg.pixel_threshold, cfg.nms_rate)
    # global list_npys
    quad_im_array, list_npy, out_txt = detect.detect()  # return：检测框打标图、文字框numpy集、检测框坐标
    # print("list_npy待传参:",list_npy)
    # q.put(list_npy)
    print('检测结束')
    # return list_npy
def  recg(list_npy):
    # 文字识别模型初始化
    recg_module = crnn.CRNN(32, 1, nclass, 256)
    if torch.cuda.is_available():
        recg_module = recg_module.cuda()
    recg_module.load_state_dict(torch.load(config.recg_module_path))
    recognize = Recg.Recognize(list_npy, recg_module, cfg.text_library, cfg.is_pick)
    result = recognize.recognize()  # return：检测点位置与识别结果
    # print(result)
    print('识别结束')
    result_npy = pt.change_cv2_draw(img_npy, result, 20, (255, 0, 0))  # 识别文字复现
    # print(type(result_npy))
    # cv2.imshow('image', result_npy)
    cv2.imwrite('image_02_crnn.png',result_npy)
    # cv2.waitKey(1000)
if __name__=='__main__':
    # 在windows必须在这句话下面开启多进程
    # p = Process(target=detection)
    i=1
    q = multiprocessing.Queue()
    p = Process(target=detection, args=(str(i), q))
    p.start()
    rec = q.get()
    p.join() # 进程结束后，GPU显存会自动释放

    print("q.get():",q.get())

    p = Process(target=recg,args=(rec)) # 重新识别
    p.start()
    p.join()
