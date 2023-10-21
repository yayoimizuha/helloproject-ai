import random
from itertools import product
from math import ceil

import torch
from PIL import Image
from numpy import array, transpose
from retinaface.pre_trained_models import get_model
from retinaface.predict_single import Model
from retinaface.network import RetinaFace
from torch import jit, randn, no_grad, Tensor, int64, tensor, onnx, from_numpy
import albumentations as A
from torchinfo import summary
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import cv2
from torch.nn import functional as F
from torchvision.extension import _assert_has_ops
from torchvision.utils import _log_api_usage_once

# model: Model = get_model(model_name='resnet50_2020-07-20', max_size=512, device='cuda')
# model.eval()
# Python random
seed = 0
random.seed(seed)
# Numpy
np.random.seed(seed)
# Pytorch
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms = True

image = Image.open(r"C:\Users\tomokazu\CLionProjects\ameba_blog_downloader\manaka_test.jpg").convert(mode="RGB")
image_arr = from_numpy(np.array(object=image, dtype=np.float32)).unsqueeze(0).permute(0, 3, 1, 2)

max_size = 512

example_input = randn(size=[10, 3, 256, 256]).float()

retina_model = RetinaFace(
    name="Resnet50",
    pretrained=False,
    return_layers={"layer2": 1, "layer3": 2, "layer4": 3},
    in_channels=256,
    out_channels=256,
).eval()

print(image_arr.size())
print(image_arr)

torch.onnx.export(
    model=retina_model,
    args=example_input,
    f="retinaface.onnx",
    input_names=["input"],
    output_names=["bbox", "confidence", "landmark"],
    dynamic_axes={"input": {
        0: "batch_size",
        2: "height",
        3: "width"
    },
        "bbox": {0: "batch_size", 1: "length"},
        "confidence": {0: "batch_size", 1: "length"},
        "landmark": {0: "batch_size", 1: "length"},
    },

)
with no_grad():
    bbox_regressions, classifications, ldm_regressions = retina_model(image_arr)
    print(bbox_regressions)
    print(classifications)
    print(ldm_regressions)
    # print(bbox_regressions.data[0][:, :2])
    print(bbox_regressions.size())
    print(classifications.size())
    print(ldm_regressions.size())
