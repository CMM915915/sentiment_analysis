# 接口设计概述
+ 输入
    + 图片（img）
+ 输出
    + 返回检测图片
    + 返回检测点坐标与识别结果

### 文字检测接口
+ 模型:model
+ 滑窗大小:slice_winW, slice_winH，默认(736,736)
+ 步长大小:stepSize(元组)，默认(736,736)
+ 最大检测尺寸:max_predict_size，默认736
+ 像素阈值:pixel_threshold，默认0.9
+ 过滤权重:nms_rate，默认0.1


### 文字识别接口
+ 模型:model
+ 文字库选择:text_library(默认为True。True为仅包含部分文字库/False包含所有的文字库)
+ 是否关键字筛选与分类: is_pick
    + xml
    + json
    + pickle(☑)