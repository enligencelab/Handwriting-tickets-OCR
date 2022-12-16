# Handwriting Tickets OCR

 Recognizing tables and Chinese, Latin characters and numbers contained in handwriting tickets

![](https://img.shields.io/badge/OS-Ubuntu-lightgray)
![](https://img.shields.io/badge/dependencies-Anaconda%203-brightgreen)

## Installation

Run the following commands.

```shell
cd $project_root
pip install openvino-dev
omz_downloader --name handwritten-simplified-chinese-recognition-0001
```

Copy Python library into OpenVino:

It may be different depending on operating systems and machines. Generally, if you use Anaconda in Linux system and your environment is named `ocr`, Python library is at `~/.conda/envs/ocr/lib/libpython3.9.so.1.0` and OpenVino collects all libraries at `~/.conda/envs/ocr/lib/python3.9/site-packages/openvino/libs`. Please run the following command.

```shell
cp ~/.conda/envs/ocr/lib/libpython3.9.so.1.0 ~/.conda/envs/ocr/lib/python3.9/site-packages/openvino/libs
```

Follow the guidance of https://pytorch.org/get-started/locally/ to install PyTorch. If you don't have a Intel GPU, please install CPU version.

If you have a Intel GPU, please choose the calculation component that is compatible to both OpenVino and PyTorch. Also, please replace the following line in `table_ocr.py`

```
net_exec = ie.load_network(network=net, device_name='CPU')
```

with

```
net_exec = ie.load_network(network=net, device_name='GPU')
```

This config hasn't been tested because the author doesn't have a Intel GPU.

Run the following commands.

```
pip install -r requirements.txt
```

