import time
import numpy as np
from PIL import Image, ImageDraw
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from DETECT.pre_processing.label import point_inside_of_quad
from DETECT.pre_processing.preprocess import resize_image
import math
from DETECT.utils.nms import nms
# 倾斜inms，用于二次过滤
from DETECT.utils.inms import nms_rotate
import tensorflow as tf
import config

def sigmoid(x):
    """`y = 1 / (1 + exp(-x))`"""
    return 1 / (1 + np.exp(-x))
class Detection():
    def __init__(self, img,detect_module, slice_size=(736,736), step_size=(368,368), max_predict_size=736, pixel_threshold=0.9,
                 nms_rate=0.1):
        self.img = img
        self.module = detect_module
        self.slice_size = slice_size
        self.step_size = step_size
        self.max_predict_size = max_predict_size
        self.pixel_threshold = pixel_threshold
        self.nms_rate = nms_rate

    @staticmethod
    def sigmoid(x):
        """`y = 1 / (1 + exp(-x))`"""
        return 1 / (1 + np.exp(-x))

    def sliding_window(self):
        image = self.img
        for y in range(0, image.shape[0], self.step_size[1]):
            for x in range(0, image.shape[1], self.step_size[0]):
                # yield the current window
                # 如果大于原始图片的宽度，则返回当前行的最后一个滑块
                if_x_over = x + self.slice_size[0]
                if_y_over = y + self.slice_size[1]
                # print(self.slice_size[1])
                if if_x_over > image.shape[1]:
                    # print('宽大于图片的宽,左上点的x坐标为:', image.shape[1] - self.slice_size[1] - 1)
                    yield (image.shape[1] - self.slice_size[1] - 1, y - 1,
                           image[y - 1:y + self.slice_size[1] - 1,
                           image.shape[1] - self.slice_size[1] - 1: image.shape[1] - 1])
                elif if_y_over > image.shape[0]:
                    # print('高大于图片的高,左上点的y坐标为:', image.shape[0] - self.slice_size[0]-1)
                    yield (x - 1, image.shape[0] - self.slice_size[0] - 1,
                           image[image.shape[0] - self.slice_size[0] - 1: image.shape[0] - 1,
                           x - 1:x + self.slice_size[0] - 1])
                elif if_y_over > image.shape[0] and if_x_over > image.shape[1]:
                    # print('都大于')
                    yield (image.shape[1] - self.slice_size[1] - 1, image.shape[0] - self.slice_size[0] - 1,
                           image[image.shape[0] - self.slice_size[0] - 1: image.shape[0] - 1,
                           image.shape[1] - self.slice_size[1] - 1: image.shape[1] - 1])
                else:
                    yield (x, y, image[y:y + self.slice_size[1], x:x + self.slice_size[0]])

    def get_slice(self):
        slice_sets = []
        position_sets = []
        for (x, y, window) in self.sliding_window():
            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != self.slice_size[1] or window.shape[1] != self.slice_size[0]:
                continue
            slice = window
            position = (x, y)
            slice_sets.append(slice)
            position_sets.append(position)
        return slice_sets, position_sets

    def cut_text_line(self, geo):
        p_min = np.amin(geo, axis=0)
        p_max = np.amax(geo, axis=0)
        min_xy = p_min.astype(int)
        max_xy = p_max.astype(int) + 2
        sub_im_arr = self.img[min_xy[1]:max_xy[1], min_xy[0]:max_xy[0], :].copy()
        for m in range(min_xy[1], max_xy[1]):
            for n in range(min_xy[0], max_xy[0]):
                if not point_inside_of_quad(n, m, geo, p_min, p_max):
                    sub_im_arr[m - min_xy[1], n - min_xy[0], :] = 255
        return sub_im_arr

    def angle(self, v1, v2):
        """
        calculation theta
        :param v1: 水平线
        :param v2: 矩形框的一条边
        :return: 角度
        """
        dx1 = v1[2] - v1[0]
        dy1 = v1[3] - v1[1]
        dx2 = v2[2] - v2[0]
        dy2 = v2[3] - v2[1]
        angle1 = math.atan2(dy1, dx1)
        angle1 = int(angle1 * 180 / math.pi)
        # print(angle1)
        angle2 = math.atan2(dy2, dx2)
        angle2 = int(angle2 * 180 / math.pi)
        # print(angle2)
        if angle1 * angle2 >= 0:
            included_angle = abs(angle1 - angle2)
        else:
            included_angle = abs(angle1) + abs(angle2)
            if included_angle > 180:
                included_angle = 360 - included_angle
        return included_angle

    def predict(self, position_sets, slice_sets):
        txt_items = []
        for count, img in enumerate(slice_sets):
            im = Image.fromarray(img)  # 转换为array数组
            d_wight, d_height = resize_image(im, self.max_predict_size)
            img = im.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
            img = image.img_to_array(img)
            img = preprocess_input(img, mode='tf')  # 归一化函数
            x = np.expand_dims(img, axis=0)  # 改变数组维度 表示在0位置添加数据
            y = self.module.predict(x)
            y = np.squeeze(y, axis=0)
            y[:, :, :3] = self.sigmoid(y[:, :, :3])
            cond = np.greater_equal(y[:, :, 0], self.pixel_threshold)
            activation_pixels = np.where(cond)
            quad_scores, quad_after_nms = nms(y, activation_pixels)
            scale_ratio_w = d_wight / im.width
            scale_ratio_h = d_height / im.height
            # 可以跳出此函数进行坐标输出和打标了
            for score, geo in zip(quad_scores, quad_after_nms):
                if np.amin(score) > 0:
                    rescaled_geo = geo / [scale_ratio_w, scale_ratio_h]
                    rescaled_geo_list = np.reshape(rescaled_geo, (8,)).tolist()
                    # 转换坐标
                    for i, rescaled_geo_one in enumerate(rescaled_geo_list):
                        # 如果索引为单数，则为y坐标
                        if i % 2 != 0:
                            # 加上滑窗区域的左上角y值
                            rescaled_geo_list[i] = rescaled_geo_one + float(position_sets[count][1])
                        else:
                            # 加上滑窗区域的左上角x值
                            rescaled_geo_list[i] = rescaled_geo_one + float(position_sets[count][0])
                    rescaled_geo_list.append(np.amin(score))  # 坐标后加上score
                    # txt_item = ','.join(map(str, rescaled_geo_list))
                    # txt_items.append(txt_item + '\n')
                    txt_items.append(rescaled_geo_list)
        return txt_items

    def rotate_format(self, txt_items):
        """
        倾斜nms预处理：format[x_c, y_c, w, h, theta]
        [x中心点，y中心点，长，宽，斜率]
        :param txt_items:
        :return:
        """
        rotate_list = []
        # 遍历出所有点的坐标
        # boxes: format[x_c, y_c, w, h, theta]
        x_line = [0, 0, 1000, 0]
        for txt in txt_items:
            x1 = txt[0]
            y1 = txt[1]
            x2 = txt[2]
            y2 = txt[3]
            x3 = txt[4]
            y3 = txt[5]
            x4 = txt[6]
            y4 = txt[7]
            # 中心点
            x_c = (x3 - x1 + 1) / 2 + x1
            y_c = (y3 - y1 + 1) / 2 + y1
            p1 = x2 - x1
            p2 = y2 - y1
            p3 = x4 - x1
            p4 = y4 - y1
            # width距离
            pw = math.hypot(p1, p2)
            # height距离
            ph = math.hypot(p3, p4)
            # theta角度
            p_angle = self.angle(x_line, [x2, y2, x1, y1])
            rotate = [x_c, y_c, pw, ph, p_angle]
            rotate_list.append(rotate)
        return rotate_list

    def detect(self,path_img):

        """
        Detection of contract diagram
        :param im:合同图原图
        :return:quad_im_array检测结果图;
                list_npy:文字框numpy集合;
                out_txt:检测框点值列表;
        """
        image0 = self.img
        # print("image_10112:",image0)
        w=image0.shape[1]
        h=image0.shape[0]
        if w>config.max_predict_size and h>config.max_predict_size:
            start = time.time()
            # 返回列表
            slice_sets, position_sets = self.get_slice()
            # 返回检测到的所有点
            txt_items = self.predict(position_sets, slice_sets)
            boxes_list = np.array(txt_items)  # 创建数组
            # [:, 8]只取出第八列
            scores = boxes_list[:, 8]

            rotate_list = self.rotate_format(txt_items)
            # 值越小，滤除得越多
            boxes = nms_rotate(tf.convert_to_tensor(rotate_list, dtype=tf.float32),
                               tf.convert_to_tensor(scores, dtype=tf.float32),
                               self.nms_rate, 500)
            new_box = boxes_list[boxes]
            print('过滤的标签数:', (len(txt_items) - len(new_box)))
            quad_im = Image.fromarray(self.img)

            quad_draw = ImageDraw.Draw(quad_im)
            list_npy = []
            for box in new_box:

                list1 = [box[0], box[1]]
                list2 = [box[2], box[3]]
                list3 = [box[4], box[5]]
                list4 = [box[6], box[7]]
                quad_draw.line([tuple(list1),
                                tuple(list2),
                                tuple(list3),
                                tuple(list4),
                                tuple(list1)], width=5, fill='red')
                ggeo = [list1, list2, list3, list4]

                ggeo = np.array(ggeo)
                # try:
                numpy_img = self.cut_text_line(ggeo)
                dict_position = []
                dict_position.extend(x for x in box)
                # 弹出最后一个score元素
                dict_position.pop()
                tuple_for_save = (numpy_img, dict_position)
                list_npy.append(tuple_for_save)
                # print("list_npy类型 :",type(list_npy))
                # except Exception as e:
                #     print(e.args)
            out_txt = []
            txt_all = new_box.tolist()
            for txt in txt_all:
                txt_item = ','.join(map(str, txt))
                out_txt.append(txt_item + '\n')
            cost_time = (time.time() - start)
            print("cost time: {:.2f}s".format(cost_time))
            quad_im_array = image.img_to_array(quad_im)
            return (quad_im_array, list_npy, out_txt)
        else:
            list_npy = []
            img = image.load_img(path_img)
            d_wight, d_height = resize_image(img, config.max_predict_size)
            img = img.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
            img = image.img_to_array(img)
            img = preprocess_input(img, mode='tf')
            x = np.expand_dims(img, axis=0)
            y = self.module.predict(x)
            y = np.squeeze(y, axis=0)
            y[:, :, :3] = sigmoid(y[:, :, :3])
            cond = np.greater_equal(y[:, :, 0], config.pixel_threshold)
            activation_pixels = np.where(cond)
            # 原有的nms处理得到的score+八个值
            quad_scores, quad_after_nms = nms(y, activation_pixels)

            with Image.open(path_img) as im:
                im_array = image.img_to_array(im.convert('RGB'))
                # 图片归一化
                d_wight, d_height = resize_image(im, config.max_predict_size)
                scale_ratio_w = d_wight / im.width
                scale_ratio_h = d_height / im.height
                im = im.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
                quad_im = im.copy()
                quad_draw = ImageDraw.Draw(quad_im)
                out_txt = []
                for score, geo, s in zip(quad_scores, quad_after_nms,
                                         range(len(quad_scores))):
                    if np.amin(score) > 0:
                        quad_draw.line([tuple(geo[0]),
                                        tuple(geo[1]),
                                        tuple(geo[2]),
                                        tuple(geo[3]),
                                        tuple(geo[0])], width=2, fill='green')
                        ggeo = [geo[0], geo[1], geo[2], geo[3]]
                        ggeo = np.array(ggeo)
                        numpy_img = self.cut_text_line(ggeo)
                        dict_position = []
                        rescaled_geo = geo / [scale_ratio_w, scale_ratio_h]
                        rescaled_geo_list = np.reshape(rescaled_geo, (8,)).tolist()

                        dict_position.extend(x for x in rescaled_geo_list)
                        tuple_for_save = (numpy_img, dict_position)
                        list_npy.append(tuple_for_save)
                        txt_item = ','.join(map(str, rescaled_geo_list))
                        out_txt.append(txt_item + '\n')

                quad_im_array = image.img_to_array(quad_im)
                with open('非滑窗.txt', 'a+') as f_txt:
                    f_txt.writelines(path_img + '\n')
                return (quad_im_array, list_npy, out_txt)
