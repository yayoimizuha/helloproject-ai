from os import getcwd
from os.path import join
from onnxruntime import InferenceSession, SessionOptions

onnx_session = InferenceSession(
    path_or_bytes="/home/tomokazu/.insightface/models/buffalo_l/w600k_r50.onnx",
    providers=[
        ('TensorrtExecutionProvider', {
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': join(getcwd(), 'onnx_cache'),
            'trt_fp16_enable': True,
        })
    ]
)
