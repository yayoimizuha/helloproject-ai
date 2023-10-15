from torch import load, randn, float, half, jit
import torch_tensorrt
from torch.nn import Module

model: Module = load(
    f='/home/tomokazu/PycharmProjects/helloproject-ai/data/artifact/facenet-tl_2023-05-28 23:05:09.874085/model.pth')
model.cuda()
model.eval()

example_input = randn(size=[1, 3, 224, 224]).float().cuda()

traced_script_module = jit.trace(model, example_inputs=[example_input])
tensorrt_module = torch_tensorrt.compile(module=traced_script_module, inputs=[example_input],
                                         enabled_precisions={float, half},
                                         truncate_long_and_double=True)

jit.save(tensorrt_module, "trt_test.ts")
