import os.path
from io import BytesIO
from itertools import chain
from PIL import Image
from more_itertools import chunked
from os import listdir
from torchinfo import summary
from torchvision import transforms
from torchvision.io import decode_jpeg
from tqdm import tqdm
from facenet_pytorch import InceptionResnetV1
import torch
import numpy
from insightface.app import FaceAnalysis

CROPPED_DIR = r"D:\helloproject-ai-data\face_cropped"
CHUNK_SIZE = 64
DEVICE = torch.device("cuda")
INPUT_SIZE = 256
TYPE = "facenet"
face_analysis = FaceAnalysis(providers=[
    ('TensorrtExecutionProvider', {
        'trt_engine_cache_enable': True,
        'trt_engine_cache_path': 'trt_cache',
        'trt_fp16_enable': True,
    }),
    'CUDAExecutionProvider',
    'CPUExecutionProvider',
])
face_analysis.prepare(ctx_id=0, det_size=(INPUT_SIZE, INPUT_SIZE))

transform = transforms.Compose([
    # transforms.ToTensor(),
    transforms.Resize(size=int(INPUT_SIZE * 1.2), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(size=INPUT_SIZE)
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# model: torch.nn.Module = get_model(name=MODEL_NAME)
# model.load_state_dict(
#     torch.load(os.path.join(os.path.dirname(__file__), "edgeface", f"{MODEL_NAME}.pt"), weights_only=False))
# model = torch.load(r"\\192.168.250.1\share\helloproject-ai-data\artifact\vggface2_facenet.pth", weights_only=False)
# model = model.eval().cuda(device=DEVICE)
with torch.no_grad():
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

    for file_name in all_cropped_list:
        sub_dir_name = file_name.split("=")[0]
        pbar.update(1)
        if pbar.desc != sub_dir_name:
            pbar.set_description(sub_dir_name)
        if file_name in labels_set:
            continue

        image = numpy.array(Image.open(os.path.join(CROPPED_DIR, sub_dir_name, file_name)))[:, :, [2, 1, 0]]
        emb = face_analysis.get(image)
        if not emb:
            continue
        if embeddings is not None:
            embeddings = numpy.concatenate([embeddings, numpy.expand_dims(emb[0].embedding, 0)], axis=0)
            # print(embeddings.shape)
        else:
            embeddings = numpy.expand_dims(emb[0].embedding, 0)
        labels.append(file_name)

        # print(embeddings.shape, labels.__len__())
    numpy.save(f"embeddings_{TYPE}.npy", embeddings)
    numpy.save(f"embeddings_{TYPE}_label.npy", numpy.array(labels))
