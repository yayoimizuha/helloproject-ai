from torch import load, randn, float, half, jit, ones, no_grad
import torch_tensorrt
from torch.nn import Module
from torch_tensorrt import Input

model: Module = load(
    f='/home/tomokazu/PycharmProjects/helloproject-ai/data/artifact/facenet-tl_2023-05-28 23:05:09.874085/model.pth')
model.cuda()
model.eval()
with no_grad():
    example_input = ones(1, 3, 224, 224).cuda()

    traced_script_module = jit.trace(model, example_inputs=[example_input])
    tensorrt_module = torch_tensorrt.compile(module=traced_script_module, inputs=[
        Input(
            min_shape=[1, 3, 224, 224],
            opt_shape=[32, 3, 224, 224],
            max_shape=[32, 3, 224, 224]
        )
    ],
                                             enabled_precisions={float},
                                             truncate_long_and_double=True,
                                             allow_shape_tensors=True)

    jit.save(tensorrt_module, "trt_test.ts")
