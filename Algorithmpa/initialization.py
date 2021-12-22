import config
import torch
import RECG.network.crnn as crnn
from DETECT.network.network import East
from RECG.utils import alphabets

nclass = len(alphabets.alphabet) + 1


# 初始化模型
def ini_dete():
    # 文字检测模型初始化
    east = East()
    detect_module = east.east_network()
    detect_module.load_weights(config.detect_module_path)
    return detect_module


def ini_recg():
    # 文字识别模型初始化
    recg_module = crnn.CRNN(32, 1, nclass, 256)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        recg_module = recg_module.cuda()
    recg_module.load_state_dict(torch.load(config.recg_module_path))

    return recg_module