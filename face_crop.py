import torch.backends.cudnn
from torch import no_grad, backends
import settings
from os import makedirs, listdir, stat, utime, devnull
from os.path import join, exists
from tqdm import tqdm
from PIL import Image
from numpy import array, arctan2, pi
from aiofiles import open as a_open
from asyncio import gather, run
from multiprocessing import Queue, Process, get_start_method, set_start_method
from time import sleep
from io import BytesIO
from math import sqrt
# from insightface.app import FaceAnalysis
# pip install retinaface_pytorch
from retinaface.pre_trained_models import get_model
from retinaface.predict_single import Model
from contextlib import redirect_stdout, redirect_stderr
from collections import OrderedDict

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


def load_image(basedir: str, queue: Queue, progress: tuple[Queue]) -> None:
    def list_up():
        for name in listdir(basedir):  # ["江端妃咲"]:
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

        while queue.qsize() > 50:
            sleep(1e-3)

        chunk = file_list[i:i + 20]

        img_bins = run(parallel_read(chunk))
        for img_bin, p in img_bins:
            queue.put((Image.open(BytesIO(img_bin)), p))
            bar.update(1)
            bar.set_postfix(OrderedDict(name=p[0], qsize=[q.qsize() for q in progress]))
    return


def pre_process(q1: Queue, q2: Queue):
    while True:
        while q2.qsize() > 50:
            # print("occur wait")
            pass
            # sleep(1e-4)
        image, path = q1.get()
        # print(type(image))
        img_arr = array(image)  # [:, :, ::-1]
        q2.put((img_arr, path))


def predict(q1: Queue, q2: Queue, gpu: int):
    sleep(gpu * 4)
    torch.backends.cudnn.benchmark = True
    with redirect_stderr(open(devnull, mode='w')):
        # face_analysis = FaceAnalysis(providers=['CUDAExecutionProvider'], allowed_modules=['detection'])
        # face_analysis.prepare(ctx_id=gpu)
        model: Model = get_model(model_name='resnet50_2020-07-20', max_size=512, device='cuda')
    model.eval()
    with no_grad():
        while True:
            image, path = q1.get()

            res = model.predict_jsons(image=image, confidence_threshold=0.4, nms_threshold=0.4)

            if res.__len__() == 1:
                if res[0]['score'] == -1:
                    continue

            faces = []
            for face in res:
                faces.append((face['landmarks'], face['score'], face['bbox']))
            q2.put((faces, path))


def post_process(queue: Queue):
    while True:

        res, path = queue.get()
        image = Image.open(join(blog_images, *path))
        name, file = path

        # width, height = image.size
        # if width * height > 400_0000:
        #     image = image.resize(size=(width // 2, height // 2))

        for order, face in enumerate(res):
            landmarks, score, bbox = face
            face_width = bbox[2] - bbox[0]
            face_height = bbox[3] - bbox[1]
            trans = truncate(landmarks)
            rotated = image.rotate(angle=trans[1] * 360 / (2 * pi), center=trans[0])
            image_size = max(face_width, face_height) * sqrt(2) // 2
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
        Load_Processes = [
            Process(target=load_image, args=(blog_images, Load_Q, (Load_Q, PreProcess_Q, Predict_Q, PostProcess_Q)))
            for _ in range(settings.FaceCropProcesses.load)]
        PreProcesses = [Process(target=pre_process, args=(Load_Q, PreProcess_Q))
                        for _ in range(settings.FaceCropProcesses.pre_process)]
        Predict_Process = [Process(target=predict, args=(PreProcess_Q, Predict_Q, gpu_id))
                           for gpu_id in range(settings.FaceCropProcesses.predict)]
        PostProcesses = [Process(target=post_process, args=(Predict_Q,))
                         for _ in range(settings.FaceCropProcesses.post_process)]
        [p.start() for p in Load_Processes]
        [p.start() for p in PreProcesses]
        [p.start() for p in Predict_Process]
        [p.start() for p in PostProcesses]
        sleep(20)
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
