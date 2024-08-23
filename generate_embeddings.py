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
from edgeface import get_model
import torch
import numpy
from insightface.app import FaceAnalysis

CROPPED_DIR = r"D:\helloproject-ai-data\face_cropped"
MODEL_NAME = "edgeface_s_gamma_05"
CHUNK_SIZE = 64
DEVICE = torch.device("cuda")
INPUT_SIZE = 256
face_analysis = FaceAnalysis()
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
model = torch.load(r"\\192.168.250.1\share\helloproject-ai-data\artifact\vggface2_facenet.pth", weights_only=False)
model = model.eval().cuda(device=DEVICE)
with torch.no_grad():
    # print(model.eval())
    summary(model, input_size=[CHUNK_SIZE, 3, INPUT_SIZE, INPUT_SIZE])
    # trt_model = torch.compile(model)

    embeddings: numpy.ndarray | None = None
    labels = []

    all_cropped_list = list(
        chain.from_iterable([listdir(os.path.join(CROPPED_DIR, name)) for name in listdir(CROPPED_DIR)]))
    # all_cropped_list = all_cropped_list[:10000]

    pbar = tqdm(total=all_cropped_list.__len__())

    for name_chunk in chunked(all_cropped_list, n=CHUNK_SIZE):
        decoded_images = []
        for file_name in name_chunk:
            sub_dir_name = file_name.split("=")[0]
            dat = numpy.fromfile(os.path.join(CROPPED_DIR, sub_dir_name, file_name), dtype=numpy.uint8)
            try:
                decoded_image = decode_jpeg(torch.tensor(dat), device=DEVICE)
            except BaseException as e:
                decoded_image = (torch.tensor(numpy.array(Image.open(BytesIO(dat.tobytes())))
                                              .transpose([2, 0, 1])).to(DEVICE))
            decoded_images.append(decoded_image)  # transform(decoded_image.to(torch.float32) / 255.))
            if pbar.desc != sub_dir_name:
                pbar.set_description(sub_dir_name)
            pbar.update(1)
        # input_tensor = torch.stack(decoded_images)
        # res: torch.Tensor = model(input_tensor)
        # print(res.shape)
        # print(face_analysis.get(decoded_images[0].cpu().numpy().transpose([1, 2, 0])[:, :, ::-1]))
        _label = []
        res = []
        for decoded_image, name in zip(decoded_images, name_chunk):
            if a := face_analysis.get(decoded_image.cpu().numpy().transpose([1, 2, 0])[:, :, ::-1]):
                _label.append(name)
                res.append(a[0].embedding)
                # print(a[0].embedding.shape)
        res = numpy.stack(res)
        if embeddings is not None:
            embeddings = numpy.concatenate([embeddings, res], axis=0)
        else:
            embeddings = res
        labels.extend(_label)
        # print(embeddings.shape, labels.__len__())
    numpy.save("embeddings.npy", embeddings)
    numpy.save("embeddings_label.npy", numpy.array(all_cropped_list))
