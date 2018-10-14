import numpy as np

res50_path = 'ckpts/resnet_v1_50.ckpt'
res152_path = 'ckpts/resnet_v1_152.ckpt'
mobilenet_path = 'ckpts/mobilenet_v2_1.0_96.ckpt'

def _get_resnet_input():
    return np.random.randn(1, 224, 224, 3)

def _get_mobilenet_input():
    return np.random.randn(1, 96, 96, 3)
