import concurrent.futures
import os.path
from concurrent.futures.process import ProcessPoolExecutor
from itertools import chain
from PIL import Image
from more_itertools import chunked
from os import listdir
from torchvision import transforms
from torchvision.io import decode_jpeg
from tqdm import tqdm
from edgeface.backbones import get_model
import torch
import numpy
from edgeface.face_alignment import align

CROPPED_DIR = r"D:\helloproject-ai-data\face_cropped"
MODEL_NAME = "edgeface_s_gamma_05"
CHUNK_SIZE = 64
DEVICE = torch.device("cuda")
INPUT_SIZE = 112
TYPE = "edgeface"

transform = transforms.Compose([
    transforms.ToTensor(),
    # lambda x: x.to(torch.float32) / 255.,
    transforms.Resize(size=INPUT_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
    # transforms.Resize(size=int(INPUT_SIZE * 1.2), interpolation=transforms.InterpolationMode.BILINEAR),
    # transforms.CenterCrop(size=INPUT_SIZE)
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def align_edgeface(p: str):
    sub_dir_name = p.split("=")[0]
    aligned = align.get_aligned_face(os.path.join(CROPPED_DIR, sub_dir_name, p))
    if aligned is None:
        return None
    return transform(aligned).to(DEVICE)


if __name__ == '__main__':
    model: torch.nn.Module = get_model(name=MODEL_NAME)
    model.load_state_dict(
        torch.load(os.path.join(os.path.dirname(__file__), "edgeface", "checkpoints", f"{MODEL_NAME}.pt"),
                   weights_only=False))
    model = model.eval().cuda(device=DEVICE)
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
        # print(model.eval())
        # summary(model, input_size=[CHUNK_SIZE, 3, INPUT_SIZE, INPUT_SIZE])
        # trt_model = torch.compile(model)

        embeddings: numpy.ndarray | None = None
        labels = []
        if os.path.exists(f"embeddings_{TYPE}_label.npy"):
            embeddings: numpy.ndarray = numpy.load(f"embeddings_{TYPE}.npy")
            labels: list[str] = numpy.load(f"embeddings_{TYPE}_label.npy").tolist()

        all_cropped_list = list(
            chain.from_iterable([listdir(os.path.join(CROPPED_DIR, name)) for name in listdir(CROPPED_DIR)]))
        # all_cropped_list = all_cropped_list[:1000]

        labels_set = set(labels)
        pbar = tqdm(total=all_cropped_list.__len__())

        for chk in chunked(all_cropped_list, n=CHUNK_SIZE):
            decoded_images = []
            pool_res_list = []
            for file_name in chk:
                pbar.update(1)
                if pbar.desc != file_name.split("=")[0]:
                    pbar.set_description(file_name.split("=")[0])
                if file_name in labels_set:
                    continue
                pool_res = align_edgeface(file_name)
                pool_res_list.append(pool_res)
            for result, name in zip(pool_res_list, chk):
                # result = result.result()
                if result is not None:
                    decoded_images.append(result)
                    labels.append(name)

            if not decoded_images:
                continue
            stacked = torch.stack(decoded_images)
            # print(stacked.shape)
            res = model(stacked)
            # print(res.shape)
            if embeddings is not None:
                embeddings = numpy.concatenate([embeddings, res.cpu().numpy()], axis=0)
                # print(embeddings.shape)
            else:
                embeddings = res.cpu().numpy()

        # print(embeddings.shape, labels.__len__())
    numpy.save(f"embeddings_{TYPE}.npy", embeddings)
    numpy.save(f"embeddings_{TYPE}_label.npy", numpy.array(labels))
