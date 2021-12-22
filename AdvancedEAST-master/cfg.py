import os

train_task_id = '3T736'
initial_epoch = 0
epoch_num = 24

lr = 1e-3

decay = 5e-4
# clipvalue = 0.5  # default 0.5, 0 means no clip
patience = 5
# 是否加载原来模型
# load_weights = False
load_weights = True

lambda_inside_score_loss = 4.0
lambda_side_vertex_code_loss = 1.0
lambda_side_vertex_coord_loss = 1.0

total_img = 5568
# 验证集比率
validation_split_ratio = 0.1
max_train_img_size = int(train_task_id[-3:])
# max_predict_img_size = int(train_task_id[-3:])  # 2400
max_predict_img_size = 3500
assert max_train_img_size in [256, 384, 512, 640, 736], \
    'max_train_img_size must in [256, 384, 512, 640, 736]'
if max_train_img_size == 256:
    batch_size = 8
elif max_train_img_size == 384:
    batch_size = 4
elif max_train_img_size == 512:
    batch_size = 2
else:
    batch_size = 1
steps_per_epoch = total_img * (1 - validation_split_ratio) // batch_size
validation_steps = total_img * validation_split_ratio // batch_size

# data_dir = 'Test_train/'
data_dir = 'Mydata/'
origin_image_dir_name = 'img_5568/'
origin_txt_dir_name = 'txt_5568/'
train_image_dir_name = 'images_%s/' % train_task_id
train_label_dir_name = 'labels_%s/' % train_task_id
show_gt_image_dir_name = 'show_gt_images_%s/' % train_task_id

show_act_image_dir_name = 'show_act_images_%s/' % train_task_id

gen_origin_img = True

draw_gt_quad = True

draw_act_quad = True

val_fname = 'val_%s.txt' % train_task_id
train_fname = 'train_%s.txt' % train_task_id
# in paper it's 0.3, maybe to large to this problem
shrink_ratio = 0.2
# pixels between 0.2 and 0.6 are side pixels:边的缩放率
shrink_side_ratio = 0.6
# 学习率
epsilon = 1e-6
# 通道数
num_channels = 3
# 特征层范围
# 5到1:5,4,3,2,1
feature_layers_range = range(5, 1, -1)
# feature_layers_range = range(3, 0, -1)
feature_layers_num = len(feature_layers_range)
# pixel_size = 4
pixel_size = 2 ** feature_layers_range[-1]
# 锁定
locked_layers = False

if not os.path.exists('model'):
    os.mkdir('model')
if not os.path.exists('saved_model'):
    os.mkdir('saved_model')

model_weights_path = 'model/weights_%s.{epoch:03d}-{val_loss:.3f}.h5' \
                     % train_task_id
saved_model_file_path = 'saved_model/east_model_%s.h5' % train_task_id
# saved_model_weights_file_path = 'Model_no_migrating/east_model_weights_%s.h5'\
#                                 % train_task_id
saved_model_weights_file_path = 'saved_model/east_model_weights_%s.h5'\
                                % train_task_id
# 目标点像素阈值
pixel_threshold = 0.9
# 边像素阈值
side_vertex_pixel_threshold = 0.9

# 头尾像素阈值
trunc_threshold = 0.1

predict_cut_text_line = False
# 生成预测坐标的txt
predict_write2txt = True
