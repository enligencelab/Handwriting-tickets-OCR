# Handwriting Tickets OCR

 Recognizing tables with Chinese characters and numbers in handwritting tickets

![](https://img.shields.io/badge/OS-Ubuntu-lightgray)
![](https://img.shields.io/badge/dependencies-Anaconda%203-brightgreen)

## Installation

Run the following commands to install OpenVino.

```shell
cd $project_root
pip install openvino-dev
omz_downloader --name handwritten-simplified-chinese-recognition-0001
```

Copy Python library into OpenVino. The location of Python library is different 
depending on operating systems and whether you use Anaconda. Generally, if you 
use Anaconda in Linux system and your environment is named `ocr`, Python library 
is at `~/.conda/envs/ocr/lib/libpython3.9.so.1.0` and OpenVino is at 
`~/.conda/envs/ocr/lib/python3.9/site-packages/openvino/libs`.

Follow the guidance of https://pytorch.org/get-started/locally/ to install 
PyTorch. If you don't have an Intel GPU, please install CPU version.

Run the following commands to install other packages. Remove PyTorch in this file 
if you have installed it and configured carefully.

```
pip install -r requirements.txt
```

### If you have an Intel GPU

If you have a Intel GPU, please choose the calculation component that is 
compatible to both OpenVino and PyTorch. Also, please replace the following 
line in `table_ocr.py`
```
net_exec = ie.load_network(network=net, device_name='CPU')
```
with
```
net_exec = ie.load_network(network=net, device_name='GPU')
```

This config hasn't been tested because the author does not have a Intel GPU.

## Acknowledgement

This project use the compiled binary form of LSTM-RNN-CTC pretrained model by 
Intel when testing, but does not have right to take actions including but not 
limiting to redistribute, copy, modify the upstreaming resources. This model 
is located at [https://github.com/intel/handwritten-chinese-ocr-samples](https://github.com/intel/handwritten-chinese-ocr-samples)
