import cv2
import config as cfg
import initialization as ini
from RECG import demo as Recg
from DETECT import demo as Detect
from DRAW import painting as pt
from numba import cuda

# 传参
img_npy = cv2.imread("D:\\PythonProjects\\TextDetection_V1.1\\image_10112.png")
# 初始化网络
def detect():
    detect_module = ini.ini_dete()

    detect = Detect.Detection(img_npy, detect_module, cfg.slice_size, cfg.step_size, cfg.max_predict_size,
                              cfg.pixel_threshold, cfg.nms_rate)
    quad_im_array, list_npy, out_txt = detect.detect()  # return：检测框打标图、文字框numpy集、检测框坐标
    print('检测结束')
    return quad_im_array, list_npy, out_txt

cuda.select_device(0)
cuda.close()
def recg(list_npy):
    recg_module = ini.ini_recg()
    recognize = Recg.Recognize(list_npy, recg_module, cfg.text_library, cfg.is_pick)
    result = recognize.recognize()  # return：检测点位置与识别结果
    print('识别结束')
    return result


quad_im_array, list_npy, out_txt = detect()
result = recg(list_npy)

result_npy = pt.change_cv2_draw(img_npy, result, 60, (255, 0, 0))  # 识别文字复现
cv2.imwrite('quad_im_array.png', quad_im_array)
cv2.imwrite('result_npy.png', result_npy)