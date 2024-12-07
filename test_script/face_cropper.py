# import cv2
import os
# print(os.environ)
for p in os.environ['Path'].split(os.pathsep):
    if os.path.isdir(p) and p != ".":
        print(p)
        os.add_dll_directory(p)

import msgspec
from torch import tensor
import torch
from torchvision.transforms import functional, InterpolationMode
from torchvision.io import decode_jpeg
# import shutil
import numpy
from PIL import Image
from io import BytesIO
from more_itertools import chunked
from tqdm import tqdm
import math

ROOT_DIR = r"E:\helloproject-ai-data\blog_images"
CROPPED_DIR = r"E:\helloproject-ai-data\face_cropped"
CROP_THRESHOLD = 0.8

inference_size = 640
device = torch.device("cuda")
device = torch.device("xpu") if torch.xpu.is_available() else exit(-1)


def calc_rotate(landmark: list[list[float]]) -> tuple[tuple[int, int], float]:
    left_eye, right_eye, nose, left_mouth, right_mouth = landmark
    center_x = sum((left_eye[0], right_eye[0], left_mouth[0], right_mouth[0])) / 4
    center_y = sum((left_eye[1], right_eye[1], left_mouth[1], right_mouth[1])) / 4
    eye_center = (right_eye[0] + left_eye[0]) / 2, (right_eye[1] + left_eye[1]) / 2
    mouth_center = (right_mouth[0] + left_mouth[0]) / 2, (right_mouth[1] + left_mouth[1]) / 2
    return (int(center_x), int(center_y)), numpy.arctan2(eye_center[0] - mouth_center[0],
                                                         mouth_center[1] - eye_center[1])


def cropper(pos_list: list[tuple[str, list[list] | None]], tqdm_pbar: tqdm, exist_set: set[str]):
    for name, pos_s in pos_list:
        file_name = name
        sub_dir_name = file_name.split("=", maxsplit=1)[0]
        if pos_s:
            decoded_image = None
            # host_image = decoded_image.cpu().numpy().transpose([1, 2, 0])
            for order, pos in enumerate(pos_s):
                dest_name = file_name.split(".")[0] + f"-{order}.jpg"
                if dest_name in exist_set:
                    continue
                # print(dest_name)
                if decoded_image is None:
                    dat = numpy.fromfile(os.path.join(ROOT_DIR, sub_dir_name, file_name),
                                         dtype=numpy.uint8)
                    try:
                        decoded_image = decode_jpeg(tensor(dat), device=device)
                    except:
                        decoded_image = tensor(
                            numpy.array(Image.open(BytesIO(dat.tobytes()))).transpose([2, 0, 1])).to(device)
                bbox, acc, landmark = pos
                if acc > CROP_THRESHOLD:
                    # scale = 1.0
                    # if max(decoded_image.shape[1:]) > inference_size:
                    scale = max(decoded_image.shape[1:]) / inference_size

                    bbox = list(map(lambda x: x * scale, bbox))
                    landmark = list(map(lambda x: x * scale, landmark))

                    # print(file_name, decoded_image.shape[1:], scale, acc)
                    center, rotate_angle = calc_rotate(list(chunked(landmark, n=2)))
                    # print(bbox, landmark, center)
                    rotated = functional.rotate(decoded_image, angle=(360 / (2 * math.pi)) * rotate_angle,
                                                center=list(center), interpolation=InterpolationMode.BILINEAR)
                    crop_size = max([int(bbox[3] - bbox[1]), int(bbox[2] - bbox[0])])
                    if crop_size < 100:
                        continue
                    cropped = functional.crop(rotated,
                                              top=int(center[1] - crop_size / 2),
                                              left=int(center[0] - crop_size / 2),
                                              height=crop_size, width=crop_size)
                    with open(os.path.join(CROPPED_DIR, sub_dir_name, dest_name), mode="wb") as fp:
                        functional.to_pil_image(cropped, mode="RGB").save(fp, format="jpeg", quality=85)
        if tqdm_pbar.desc != sub_dir_name:
            tqdm_pbar.set_description(sub_dir_name)
        tqdm_pbar.update(n=1)


#         cv2.rectangle(host_image, (int(bbox[0]), int(bbox[1])),
#                       (int(bbox[2]), int(bbox[3])),
#                       (255, 0, 0), 2, cv2.LINE_AA)
# with open(
#         os.path.join(CROPPED_DIR, sub_dir_name, file_name),
#         mode="wb") as fp:
#     cv2.imencode(".jpg", cv2.cvtColor(host_image, cv2.COLOR_BGR2RGB))[1].tofile(fp)


if __name__ == '__main__':
    with open(file="faces.jsonl", mode="r", encoding="utf-8") as fp:
        face_pos_list: list[dict[str, list | None]] = [msgspec.json.decode(line) for line in
                                                       fp.read().removesuffix("\n").split("\n")]

    face_pos_dict = {}
    for face_pos in face_pos_list:
        if next(iter(face_pos.values())) is None:
            face_pos_dict[next(iter(face_pos.keys()))] = None
        else:
            if not next(iter(face_pos.keys())) in face_pos_dict.keys():
                face_pos_dict[next(iter(face_pos.keys()))] = []
            face_pos_dict[next(iter(face_pos.keys()))].append(next(iter(face_pos.values())))

    # shutil.rmtree(CROPPED_DIR)
    os.makedirs(CROPPED_DIR, exist_ok=True)
    # print(face_pos)
    names_set = {next(iter(_dict.keys())).split("=", maxsplit=1)[0] for _dict in face_pos_list}
    print(names_set)
    [os.makedirs(os.path.join(CROPPED_DIR, name), exist_ok=True) for name in names_set]
    pbar = tqdm(total=face_pos_dict.items().__len__())

    exist_set = set().union(
        *[set(os.listdir(os.path.join(CROPPED_DIR, sub_dir))) for sub_dir in os.listdir(CROPPED_DIR)])
    cropper(list(face_pos_dict.items()), pbar, exist_set)
