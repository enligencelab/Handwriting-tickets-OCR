import cv2
import numpy as np
from openvino.inference_engine import IECore
from ctccodec import CtcCodec

# %%
model_xml = 'intel/handwritten-simplified-chinese-recognition-0001/FP16-INT8/handwritten-simplified-chinese' \
            '-recognition-0001.xml'
model_bin = 'intel/handwritten-simplified-chinese-recognition-0001/FP16-INT8/handwritten-simplified-chinese' \
            '-recognition-0001.bin'
# Prepare the language specific information, characters list and codec method
chars_list_file = 'scut_ept.txt'
with open(chars_list_file, 'r') as f:
    model_characters = f.read()
codec = CtcCodec(model_characters)
ie = IECore()  # Plugin initialization for specified device and load extensions library if specified
net = ie.read_network(model=model_xml, weights=model_bin)  # Read OpenVino IR model
net_exec = ie.load_network(network=net, device_name='CPU')
_, _, net_h, net_w = net.input_info['actual_input'].input_data.shape


# %%
def preprocess_input(img, net_h_, net_w_):
    # this function is applicable when text alignment is horizontal
    cell_h, cell_w = img.shape
    # when slimmer than the requirement of model, keep the aspect ratio and pad white
    adjusted_cell_w = int(cell_w / cell_h * net_h_)
    if adjusted_cell_w <= net_w_:
        processed_img = np.full((net_h_, net_w_), 255)
        processed_img[:, :adjusted_cell_w] = cv2.resize(img, (adjusted_cell_w, net_h_), interpolation=cv2.INTER_AREA)
    # when wider than the requirement of model,
    else:
        processed_img = cv2.resize(img, (net_w_, net_h_), interpolation=cv2.INTER_AREA)
    return processed_img[np.newaxis, np.newaxis, :, :]


def table_to_text(img, anchors_):
    tb_h, tb_w = img.shape
    x = np.round(anchors_['avg_relative_x'].values * tb_w).astype(int)
    y = np.round(anchors_['avg_relative_y'].values * tb_h).astype(int)
    w = np.round(anchors_['avg_relative_w'].values * tb_w).astype(int)
    h = np.round(anchors_['avg_relative_h'].values * tb_h).astype(int)
    img = cv2.bitwise_not(img)  # convert to white background
    res_decoded = []
    for i in range(anchors_.shape[0]):
        cell = img[y[i]:y[i] + h[i], x[i]:x[i] + w[i]]
        processed_cell = preprocess_input(cell, net_h, net_w)
        res = net_exec.infer(inputs={'actual_input': processed_cell})
        res_decoded += codec.decode(res['output'])
    return res_decoded
