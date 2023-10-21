from os import getcwd
from os.path import join
from onnxruntime import InferenceSession, SessionOptions, __version__
from PIL import Image
import numpy

onnx_session = InferenceSession(
    path_or_bytes="test_script/retinaface.onnx",
    providers=[
        'CUDAExecutionProvider',
        ('TensorrtExecutionProvider', {
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': join(getcwd(), 'onnx_cache'),
            'trt_fp16_enable': True,
        }),
        'CPUExecutionProvider'
    ]
)
print(__version__)
image_arr = numpy.expand_dims(numpy.array(
    Image.open(r'C:\Users\tomokazu\CLionProjects\ameba_blog_downloader\manaka_test.jpg').convert('RGB')), 0).transpose(
    0, 3, 1, 2).astype(numpy.float32)
print(image_arr)
print(image_arr.shape)
res = onnx_session.run(input_feed={'input': image_arr}, output_names=["bbox", "confidence", "landmark"])
for val in res:
    print(val)
    print(val.shape)
