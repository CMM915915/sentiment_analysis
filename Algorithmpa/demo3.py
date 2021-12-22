#!/usr/bin/python
# -*- coding: utf-8 -*-
# author:mengmeng
# 日期:2020/3/27
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
path_img="D:\\PythonProjects\\TextDetection_V1.1\\image_03.png"
img_npy = cv2.imread("D:\\PythonProjects\\TextDetection_V1.1\\image_03.png")

list_npys=[]
# 初始化网络
def  detection(q):
    # 初始化检测网络
    east = East()
    detect_module = east.east_network()
    detect_module.load_weights(config.detect_module_path)
    detect = Detect.Detection(img_npy, detect_module, cfg.slice_size, cfg.step_size, cfg.max_predict_size,
                              cfg.pixel_threshold, cfg.nms_rate)

    quad_im_array, list_npy, out_txt = detect.detect(path_img)  # return：检测框打标图、文字框numpy集、检测框坐标
    cv2.imwrite('image_03_pre.png', quad_im_array)
    print("检测到的框数:", len(out_txt))
    q.put(list_npy)
    print('检测结束')
    # return list_npy
# list_npys=detection()
def  recg(list_npy):
    # 文字识别模型初始化
    recg_module = crnn.CRNN(32, 1, nclass, 256)
    if torch.cuda.is_available():
        recg_module = recg_module.cuda()
    recg_module.load_state_dict(torch.load(config.recg_module_path))
    recognize = Recg.Recognize(list_npy, recg_module, cfg.text_library, cfg.is_pick)
    result = recognize.recognize()  # return：检测点位置与识别结果
    print('识别结束')
    result_npy = pt.change_cv2_draw(img_npy, result, 20, (255, 0, 0))  # 识别文字复现
    cv2.imwrite('image_03_crnn.png',result_npy)

if __name__=='__main__':
    q = multiprocessing.Queue()
    p = Process(target=detection,args=(q,))
    p.start()
    rec = q.get()
    p.join() # 进程结束后，GPU显存会自动释放

    p = Process(target=recg,args=(rec,)) # 重新识别
    p.start()
    p.join()

