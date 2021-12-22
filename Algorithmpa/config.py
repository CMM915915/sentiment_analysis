img_path = '安徽区域二部-亳州春暖花开-洋房-7#-终稿-0.png'

detect_module_path = 'DETECT/saved_model/east_model_weights_3T736.h5'
recg_module_path = 'RECG/model/crnn_Rec_done_162_0.91875.pth'

slice_size=(368,368)
step_size=(184,184)
max_predict_size=736
pixel_threshold=0.7
nms_rate=0.1

text_library = True
is_pick = 'RECG/type_dict/testv1.0.pkl'
max_predict_img_size=736