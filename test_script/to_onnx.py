from torch import load, randn, float, half, jit, ones, no_grad
# import torch_tensorrt
from torchinfo import summary
from torch.nn import Module
from torch.onnx import export

model: Module = load(
    f=r"\\tomokazu-ubuntu-server\share\helloproject-ai-data\artifact\facenet-tl_2023-10-22 213825.539264\model.pth")
# model.cuda()
model.eval()
summary(
    model=model,
    input_size=[1, 3, 224, 224],
    device='cpu',
    col_names=["input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"]
)
with no_grad():
    example_input = randn(1, 3, 224, 224)

    export(
        model=model,
        args=example_input,
        f="face_recognition.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {
                0: "batch_size",
                # 2: "height",
                # 3: "width"
            }
        },
        verbose=False
    )
