import time
import tkinter

# import cv2
from nvjpeg_decoder import decode
from os import listdir, getcwd
from os.path import join
from numpy import array, fromfile, uint8
# from matplotlib import pyplot, figure
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# import matplotlib_fontja
from onnxruntime import InferenceSession

onnx_path = r"C:\Users\tomokazu\RustroverProjects\ameba_blog_downloader\src\retinaface\resnet_retinaface.onnx"
datadir = r"D:\helloproject-ai-data\blog_images"

session = InferenceSession(
    path_or_bytes=onnx_path,
    # providers=[
    #     # 'TensorrtExecutionProvider',
    #     # 'CUDAExecutionProvider',
    #     'CPUExecutionProvider'
    # ]
)
for member in listdir(datadir):
    for file in listdir(join(datadir, member)):
        # with open(join(datadir, member, file), mode="rb") as f:
        (data, (scale, (width, height))) = decode(fromfile(join(datadir, member, file), dtype=uint8), "imagenet",
                                                  (1080, 1080))
        print(width, height)
        image_arr = array(data).reshape((1, 3, height, width))  # .transpose([1, 2, 0])
        session.run(input_feed={'input': image_arr}, output_names=['bbox', 'confidence', 'landmark'])
