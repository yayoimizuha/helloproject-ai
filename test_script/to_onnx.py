from torch import load, randn, float, half, jit, ones, no_grad
import torch_tensorrt
from torch.nn import Module
from torch.onnx import export

model: Module = load(
    f='/home/tomokazu/PycharmProjects/helloproject-ai/data/artifact/facenet-tl_2023-10-15 14:46:51.187699/checkpoints/80.pth')
model.cuda()
model.eval()
model = model.half()
with no_grad():
    example_input = randn(1, 3, 224, 224).cuda().half()

    export(
        model=model,
        args=example_input,
        f="onnx_test.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {
                0: "batch_size",
                2: "height",
                3: "width"
            }
        },
        verbose=False
    )
