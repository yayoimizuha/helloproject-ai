import torch.cuda
from numpy import ndarray
import numpy as np
from DBFace_without_OpenCV import DBFace
import settings
from os import makedirs, listdir, stat, utime
from os.path import join, exists
from tqdm import tqdm
from PIL import Image, ImageDraw
from numpy import array, arctan2, pi, zeros, uint8, float32
from aiofiles import open as a_open
from asyncio import gather, run
from multiprocessing import Queue, Process, get_start_method, set_start_method
from time import time, sleep
from io import BytesIO
from math import ceil, sqrt
from torch import from_numpy, cuda, Tensor, inference_mode, nn
import atexit

face_dir = join(settings.datadir(), 'face_cropped')
blog_images = join(settings.datadir(), 'blog_images')

if not exists(face_dir):
    makedirs(face_dir)


def truncate(landmark: list[tuple[float]]) -> tuple[tuple[int, int], float]:
    left_eye, right_eye, nose, left_mouth, right_mouth = landmark
    center_x = sum((left_eye[0], right_eye[0], left_mouth[0], right_mouth[0])) / 4
    center_y = sum((left_eye[1], right_eye[1], left_mouth[1], right_mouth[1])) / 4
    eye_center = (right_eye[0] + left_eye[0]) / 2, (right_eye[1] + left_eye[1]) / 2
    mouth_center = (right_mouth[0] + left_mouth[0]) / 2, (right_mouth[1] + left_mouth[1]) / 2
    return (int(center_x), int(center_y)), arctan2(eye_center[0] - mouth_center[0], mouth_center[1] - eye_center[1])


def load_image(basedir: str, queue: Queue) -> None:
    def list_up():
        for name in listdir(basedir):
            for image_file in listdir(join(basedir, name)):
                yield name, image_file

    async def single_read(path: tuple[str, str]):
        async with a_open(join(basedir, *path), mode='rb') as f:
            return await f.read(), path

    async def parallel_read(paths: list[tuple[str, str]]):
        return await gather(*[single_read(path) for path in paths])

    file_list = [i for i in list_up()]
    bar = tqdm(total=file_list.__len__())
    for i in range(0, file_list.__len__(), 20):

        while queue.qsize() > 300:
            sleep(1e-3)

        chunk = file_list[i:i + 20]

        img_bins = run(parallel_read(chunk))
        for img_bin, p in img_bins:
            queue.put((Image.open(BytesIO(img_bin)), p))
            bar.update(1)
    return


def pre_process(q1: Queue, q2: Queue):
    mean = [0.408, 0.447, 0.47]
    std = [0.289, 0.274, 0.278]
    while True:
        while q2.qsize() > 4:
            sleep(1e-4)
        image, path = q1.get()
        width, height = image.size
        if width * height > 400_0000:
            image = image.resize(size=(width // 2, height // 2))
        # print(path, image.size)
        image = image.crop((0, 0, ceil(width / 32) * 32, ceil(height / 32) * 32))  # padding
        img_arr = array(image)
        img_arr = ((img_arr / 255.0 - mean) / std).astype(float32).transpose(2, 0, 1)
        torch_image = from_numpy(img_arr)[None]
        q2.put((torch_image.cuda(), path))
        pass


def predict(q1: Queue, q2: Queue):
    model_path = '/home/tomokazu/PycharmProjects/helloproject-ai/DBFace_without_OpenCV/model/dbface.pth'
    db_face = DBFace()
    db_face.eval()
    db_face.cuda()
    db_face.load(model_path)
    i = 0
    start = time()
    while True:
        # if i % 5000 == 0:
        #     cuda.empty_cache()
        # torch_image:Tensor=torch_image
        # print(i, path, torch_image.size())
        i += 1
        try:
            torch_image, path = q1.get()
            with inference_mode():
                q2.put((db_face(torch_image), path))
        except Exception as e:
            print(e)
            del db_face
            cuda.empty_cache()
            db_face = DBFace()
            db_face.eval()
            db_face.cuda()
            db_face.load(model_path)


def exp(v):
    if isinstance(v, tuple) or isinstance(v, list):
        return [exp(item) for item in v]
    elif isinstance(v, ndarray):
        return np.array([exp(item) for item in v], v.dtype)

    gate = 1
    base = np.exp(1)
    if abs(v) < gate:
        return v * base

    if v > 0:
        return np.exp(v)
    else:
        return -np.exp(-v)


def nms(objs, iou=0.5):
    if objs is None or len(objs) <= 1:
        return objs

    objs = sorted(objs, key=lambda obj: obj.score, reverse=True)
    keep = []
    flags = [0] * len(objs)
    for index, obj in enumerate(objs):

        if flags[index] != 0:
            continue

        keep.append(obj)
        for j in range(index + 1, len(objs)):
            if flags[j] == 0 and obj.iou(objs[j]) > iou:
                flags[j] = 1
    return keep


class BBox:

    def __init__(self, label, xyrb, score=0, landmark=None, rotate=False):
        self.label = label
        self.score = score
        self.landmark = landmark
        self.x, self.y, self.r, self.b = xyrb
        self.rotate = rotate

        minx = min(self.x, self.r)
        maxx = max(self.x, self.r)
        miny = min(self.y, self.b)
        maxy = max(self.y, self.b)
        self.x, self.y, self.r, self.b = minx, miny, maxx, maxy

    def __repr__(self):
        landmark_formated = ",".join(
            [str(item[:2]) for item in self.landmark]) if self.landmark is not None else "empty"
        return f"(BBox[{self.label}]: x={self.x:.2f}, y={self.y:.2f}, r={self.r:.2f}, " + \
            f"b={self.b:.2f}, width={self.width:.2f}, height={self.height:.2f}, landmark={landmark_formated})"

    @property
    def width(self):
        return self.r - self.x + 1

    @property
    def height(self):
        return self.b - self.y + 1

    @property
    def area(self):
        return self.width * self.height

    @property
    def haslandmark(self):
        return self.landmark is not None

    @property
    def xxxxxyyyyy_cat_landmark(self):
        x, y = zip(*self.landmark)
        return x + y

    @property
    def box(self):
        return [self.x, self.y, self.r, self.b]

    @box.setter
    def box(self, newvalue):
        self.x, self.y, self.r, self.b = newvalue

    @property
    def xywh(self):
        return [self.x, self.y, self.width, self.height]

    @property
    def center(self):
        return [(self.x + self.r) * 0.5, (self.y + self.b) * 0.5]

    # return cx, cy, cx.diff, cy.diff
    def safe_scale_center_and_diff(self, scale, limit_x, limit_y):
        cx = clip_value((self.x + self.r) * 0.5 * scale, limit_x - 1)
        cy = clip_value((self.y + self.b) * 0.5 * scale, limit_y - 1)
        return [int(cx), int(cy), cx - int(cx), cy - int(cy)]

    def safe_scale_center(self, scale, limit_x, limit_y):
        cx = int(clip_value((self.x + self.r) * 0.5 * scale, limit_x - 1))
        cy = int(clip_value((self.y + self.b) * 0.5 * scale, limit_y - 1))
        return [cx, cy]

    def clip(self, width, height):
        self.x = clip_value(self.x, width - 1)
        self.y = clip_value(self.y, height - 1)
        self.r = clip_value(self.r, width - 1)
        self.b = clip_value(self.b, height - 1)
        return self

    def iou(self, other):
        return computeIOU(self.box, other.box)


def computeIOU(rec1, rec2):
    cx1, cy1, cx2, cy2 = rec1
    gx1, gy1, gx2, gy2 = rec2
    S_rec1 = (cx2 - cx1 + 1) * (cy2 - cy1 + 1)
    S_rec2 = (gx2 - gx1 + 1) * (gy2 - gy1 + 1)
    x1 = max(cx1, gx1)
    y1 = max(cy1, gy1)
    x2 = min(cx2, gx2)
    y2 = min(cy2, gy2)

    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)
    area = w * h
    iou = area / (S_rec1 + S_rec2 - area)
    return iou


def clip_value(value, high, low=0):
    return max(min(value, high), low)


def post_process(queue: Queue, threshold: float = 0.4, nms_iou: float = 0.5):
    while True:
        tensor, path = queue.get()
        hm, box, landmark = tensor
        del tensor
        name, file = path
        hm_pool = nn.functional.max_pool2d(hm, 3, 1, 1)
        t = ((hm == hm_pool).float() * hm).view(1, -1).cpu()
        if t.size()[1] < 1000:
            continue
        scores, indices = t.topk(1000)
        hm_height, hm_width = hm.shape[2:]
        del hm
        scores = scores.squeeze()
        indices = indices.squeeze()
        ys = list((indices / hm_width).int().data.numpy())
        xs = list((indices % hm_width).int().data.numpy())
        scores = list(scores.data.numpy())
        box = box.cpu().squeeze().data.numpy()
        landmark = landmark.cpu().squeeze().data.numpy()

        stride = 4
        objs = []
        for cx, cy, score in zip(xs, ys, scores):
            if score < threshold:
                break

            x, y, r, b = box[:, cy, cx]
            xyrb = (array([cx, cy, cx, cy]) + [-x, -y, r, b]) * stride
            x5y5 = landmark[:, cy, cx]
            x5y5 = (exp(x5y5 * 4) + ([cx] * 5 + [cy] * 5)) * stride
            box_landmark = list(zip(x5y5[:5], x5y5[5:]))
            objs.append(BBox(0, xyrb=xyrb, score=score, landmark=box_landmark))
        predicted = nms(objs, iou=nms_iou)
        image = Image.open(join(blog_images, *path))

        width, height = image.size
        if width * height > 400_0000:
            image = image.resize(size=(width // 2, height // 2))

        for order, face in enumerate(predicted):
            trans = truncate(face.landmark)
            rotated = image.rotate(angle=trans[1] * 360 / (2 * pi), center=trans[0])
            image_size = max(face.width, face.height) * sqrt(2) // 2
            if image_size < 100:
                continue
            cropped = rotated.crop((trans[0][0] - image_size, trans[0][1] - image_size, trans[0][0] + image_size,
                                    trans[0][1] + image_size))
            if not exists(join(face_dir, name)):
                makedirs(join(face_dir, name), exist_ok=True)
            saved_path = join(face_dir, name, file.replace('.jpg', '-' + str(order + 1) + '.jpg'))
            cropped.save(saved_path)
            utime(path=saved_path, times=(stat(join(blog_images, *path)).st_atime,
                                          stat(join(blog_images, *path)).st_mtime))


if __name__ == '__main__':
    if get_start_method() == 'fork':
        set_start_method('spawn', force=True)

    try:
        Load_Q, PreProcess_Q, Predict_Q, PostProcess_Q = (Queue() for i in range(4))
        Load_Processes = [Process(target=load_image, args=(blog_images, Load_Q))
                          for _ in range(settings.FaceCropProcesses.load)]
        PreProcesses = [Process(target=pre_process, args=(Load_Q, PreProcess_Q))
                        for _ in range(settings.FaceCropProcesses.pre_process)]
        Predict_Process = [Process(target=predict, args=(PreProcess_Q, Predict_Q))
                           for _ in range(settings.FaceCropProcesses.predict)]
        PostProcesses = [Process(target=post_process, args=(Predict_Q,))
                         for _ in range(settings.FaceCropProcesses.post_process)]
        [p.start() for p in Load_Processes]
        [p.start() for p in PreProcesses]
        [p.start() for p in Predict_Process]
        [p.start() for p in PostProcesses]
        while True:
            sleep(5)
            # print(Load_Q.qsize(), PreProcess_Q.qsize(), Predict_Q.qsize(), PostProcess_Q.qsize())
            if sum((Load_Q.qsize(), PreProcess_Q.qsize(), Predict_Q.qsize(), PostProcess_Q.qsize())) == 0:
                raise KeyboardInterrupt

    except KeyboardInterrupt as e:
        print(e)
        [p.terminate() for p in Load_Processes]
        [p.terminate() for p in PreProcesses]
        [p.terminate() for p in Predict_Process]
        [p.terminate() for p in PostProcesses]
