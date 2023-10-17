from itertools import product
from math import ceil

import torch
from PIL import Image
from numpy import array
from retinaface.pre_trained_models import get_model
from retinaface.predict_single import Model
from retinaface.network import RetinaFace
from torch import jit, randn, no_grad, Tensor, int64, tensor, onnx
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


image = Image.open(
    fp="/home/tomokazu/PycharmProjects/helloproject-ai/data/blog_images"
       "/稲場愛香/稲場愛香=juicejuice-official=12737097989-2.jpg").convert(mode="RGB")
image_arr = array(image)
max_size = 512

example_input = randn(size=[1, 3, 256, 256]).float().cuda()

retina_model = RetinaFace(
    name="Resnet50",
    pretrained=False,
    return_layers={"layer2": 1, "layer3": 2, "layer4": 3},
    in_channels=256,
    out_channels=256,
).cuda()


# onnx.export(
#     model=retina_model, args=example_input, export_params=True, verbose=False, input_names=["input"],
#     output_names=["bbox", "confidence", "landmark"],
#     dynamic_axes={"input": {
#         0: "batch_size",
#         2: "height",
#         3: "width"
#     }, "bbox": {1: "bbox"}, "confidence": {1: "confidence"}, "landmark": {1: "landmark"}}, opset_version=16,
#     f="retinaface.onnx"
# )

def pad_to_size(
        target_size: Tuple[int, int],
        image: np.array,
        bboxes: Optional[np.ndarray] = None,
        keypoints: Optional[np.ndarray] = None,
) -> Dict[str, Union[np.ndarray, Tuple[int, int, int, int]]]:
    target_height, target_width = target_size

    image_height, image_width = image.shape[:2]

    if target_width < image_width:
        raise ValueError(f"Target width should bigger than image_width" f"We got {target_width} {image_width}")

    if target_height < image_height:
        raise ValueError(f"Target height should bigger than image_height" f"We got {target_height} {image_height}")

    if image_height == target_height:
        y_min_pad = 0
        y_max_pad = 0
    else:
        y_pad = target_height - image_height
        y_min_pad = y_pad // 2
        y_max_pad = y_pad - y_min_pad

    if image_width == target_width:
        x_min_pad = 0
        x_max_pad = 0
    else:
        x_pad = target_width - image_width
        x_min_pad = x_pad // 2
        x_max_pad = x_pad - x_min_pad

    result = {
        "pads": (x_min_pad, y_min_pad, x_max_pad, y_max_pad),
        "image": cv2.copyMakeBorder(image, y_min_pad, y_max_pad, x_min_pad, x_max_pad, cv2.BORDER_CONSTANT),
    }

    if bboxes is not None:
        bboxes[:, 0] += x_min_pad
        bboxes[:, 1] += y_min_pad
        bboxes[:, 2] += x_min_pad
        bboxes[:, 3] += y_min_pad

        result["bboxes"] = bboxes

    if keypoints is not None:
        keypoints[:, 0] += x_min_pad
        keypoints[:, 1] += y_min_pad

        result["keypoints"] = keypoints

    return result


def tensor_from_rgb_image(image: np.ndarray) -> torch.Tensor:
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(image)


def priorbox(min_sizes, steps, clip, image_size):
    feature_maps = [[ceil(image_size[0] / step), ceil(image_size[1] / step)] for step in steps]

    anchors = []
    for k, f in enumerate(feature_maps):
        t_min_sizes = min_sizes[k]
        for i, j in product(range(f[0]), range(f[1])):
            for min_size in t_min_sizes:
                s_kx = min_size / image_size[1]
                s_ky = min_size / image_size[0]
                dense_cx = [x * steps[k] / image_size[1] for x in [j + 0.5]]
                dense_cy = [y * steps[k] / image_size[0] for y in [i + 0.5]]
                for cy, cx in product(dense_cy, dense_cx):
                    anchors += [cx, cy, s_kx, s_ky]

    # back to torch land
    output = torch.Tensor(anchors).view(-1, 4)
    if clip:
        output.clamp_(max=1, min=0)
    return output


def decode(
        loc: torch.Tensor, priors: torch.Tensor, variances: Union[List[float], Tuple[float, float]]
) -> torch.Tensor:
    boxes = torch.cat(
        (
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1]),
        ),
        1,
    )
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def decode_landm(
        pre: torch.Tensor, priors: torch.Tensor, variances: Union[List[float], Tuple[float, float]]
) -> torch.Tensor:
    return torch.cat(
        (
            priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
        ),
        dim=1,
    )


def nms(boxes: Tensor, scores: Tensor, iou_threshold: float) -> Tensor:
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(nms)
    _assert_has_ops()
    return torch.ops.torchvision.nms(boxes, scores, iou_threshold)


def unpad_from_size(
        pads: Tuple[int, int, int, int],
        image: Optional[np.array] = None,
        bboxes: Optional[np.ndarray] = None,
        keypoints: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    x_min_pad, y_min_pad, x_max_pad, y_max_pad = pads

    result = {}

    if image is not None:
        height, width = image.shape[:2]
        result["image"] = image[y_min_pad: height - y_max_pad, x_min_pad: width - x_max_pad]

    if bboxes is not None:
        bboxes[:, 0] -= x_min_pad
        bboxes[:, 1] -= y_min_pad
        bboxes[:, 2] -= x_min_pad
        bboxes[:, 3] -= y_min_pad

        result["bboxes"] = bboxes

    if keypoints is not None:
        keypoints[:, 0] -= x_min_pad
        keypoints[:, 1] -= y_min_pad

        result["keypoints"] = keypoints

    return result


device = "cuda"
transform = A.Compose([A.LongestMaxSize(max_size=max_size, p=1), A.Normalize(p=1)])
variance = [0.1, 0.2]
nms_threshold = .4
confidence_threshold = .7
_priorbox = priorbox(
    min_sizes=[[16, 32], [64, 128], [256, 512]],
    steps=[8, 16, 32],
    clip=False,
    image_size=(max_size, max_size),
).to(device)
original_height, original_width = image_arr.shape[:2]

scale_landmarks = torch.from_numpy(np.tile([max_size, max_size], 5)).to(device)
scale_bboxes = torch.from_numpy(np.tile([max_size, max_size], 2)).to(device)

transformed_image = transform(image=image_arr)["image"]

paded = pad_to_size(target_size=(max_size, max_size), image=transformed_image)

pads = paded["pads"]

torched_image = tensor_from_rgb_image(paded["image"]).to(device)


# loc, conf, land = retina_model(torched_image.unsqueeze(0))


def infer(loc, conf, land):
    conf = F.softmax(conf, dim=-1)

    annotations = []

    boxes = decode(loc.data[0], _priorbox, variance)

    boxes *= scale_bboxes
    scores = conf[0][:, 1]

    landmarks = decode_landm(land.data[0], _priorbox, variance)
    landmarks *= scale_landmarks

    # ignore low scores
    valid_index = torch.where(scores > confidence_threshold)[0]
    boxes = boxes[valid_index]
    landmarks = landmarks[valid_index]
    scores = scores[valid_index]

    # Sort from high to low
    order = scores.argsort(descending=True)
    boxes = boxes[order]
    landmarks = landmarks[order]
    scores = scores[order]

    # do NMS
    keep = nms(boxes, scores, nms_threshold)
    boxes = boxes[keep, :].int()

    if boxes.shape[0] == 0:
        return [{"bbox": [], "score": -1, "landmarks": []}]

    landmarks = landmarks[keep]

    scores = scores[keep].cpu().detach().numpy().astype(np.float64)
    boxes = boxes.cpu().numpy()
    landmarks = landmarks.cpu().numpy()
    landmarks = landmarks.reshape([-1, 2])

    unpadded = unpad_from_size(pads, bboxes=boxes, keypoints=landmarks)

    resize_coeff = max(original_height, original_width) / max_size

    boxes = (unpadded["bboxes"] * resize_coeff).astype(int)
    landmarks = (unpadded["keypoints"].reshape(-1, 10) * resize_coeff).astype(int)

    for box_id, bbox in enumerate(boxes):
        x_min, y_min, x_max, y_max = bbox

        x_min = np.clip(x_min, 0, original_width - 1)
        x_max = np.clip(x_max, x_min + 1, original_width - 1)

        if x_min >= x_max:
            continue

        y_min = np.clip(y_min, 0, original_height - 1)
        y_max = np.clip(y_max, y_min + 1, original_height - 1)

        if y_min >= y_max:
            continue

        annotations += [
            {
                "bbox": bbox.tolist(),
                "score": scores[box_id],
                "landmarks": landmarks[box_id].reshape(-1, 2).tolist(),
            }
        ]
    return annotations


ans = infer(*retina_model(torched_image.unsqueeze(0)))
print(ans)
