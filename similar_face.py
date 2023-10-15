import time
from shutil import copyfile
from insightface.app import FaceAnalysis
from os import getcwd, listdir, makedirs
from os.path import join, isdir, isfile, basename, dirname
from numpy import dot, array
from numpy.linalg import norm
from PIL import Image
from sys import argv

if argv.__len__() != 3:
    exit(1)
if not isdir(argv[2]):
    exit(1)

if not isfile(argv[1]):
    exit(1)

face_analysis = FaceAnalysis(providers=[
    # 'CUDAExecutionProvider',
    # 'CPUExecutionProvider',
    ('TensorrtExecutionProvider', {
        'trt_engine_cache_enable': True,
        'trt_engine_cache_path': join(getcwd(), 'onnx_cache'),
        'trt_fp16_enable': True,
    })
], allowed_modules=['recognition', 'detection'])
face_analysis.prepare(ctx_id=0, det_size=(160, 160))

print(argv)
collect_image = array(Image.open(join(getcwd(), argv[1])))[:, :, [2, 1, 0]]
image_files: list[str] = listdir(join(getcwd(), argv[2]))

collect_image_emb = face_analysis.get(collect_image)
if collect_image_emb.__len__() == 0:
    print("Not found face: ", argv[1])
    exit(1)

# collect_image_emb = collect_image_emb[0].embedding

dir_name = basename(dirname(argv[2]))
print(dir_name)
makedirs(join(getcwd(), dir_name, "true"), exist_ok=True)
makedirs(join(getcwd(), dir_name, "false"), exist_ok=True)

images = []
begin = time.time()
for file in image_files:
    if isfile(join(getcwd(), argv[2], file)):
        # print(join(getcwd(), argv[2], file))
        image = array(Image.open(join(getcwd(), argv[2], file)))[:, :, [2, 1, 0]]
        emb = face_analysis.get(image)
        if not emb:
            continue
        cosine = dot(emb[0].embedding, collect_image_emb[0].embedding) / \
                 (norm(emb[0].embedding) * norm(collect_image_emb[0].embedding))
        print(file, cosine)
        if cosine > 0.3:
            copyfile(join(getcwd(), argv[2], file), join(getcwd(), dir_name, "true", file))
        else:

            copyfile(join(getcwd(), argv[2], file), join(getcwd(), dir_name, "false", file))

print(f"{time.time() - begin}sec")
