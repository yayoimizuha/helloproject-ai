import time
from os import makedirs
from os.path import join, exists, basename
from shutil import rmtree, copyfile
from more_itertools import chunked
from torch import load, no_grad, device, randn, jit, float64, float32, float16
from torch.cuda import is_available
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, Resize, CenterCrop
from torchinfo import summary
from tqdm import tqdm
from settings import datadir
from concurrent.futures import ThreadPoolExecutor
from pandas import DataFrame
from seaborn import heatmap, color_palette, set_palette
from matplotlib import pyplot
from japanize_matplotlib import japanize
from torch_tensorrt import compile, Input

device = device('cuda' if is_available() else 'cpu')
# device = 'cpu'
print(f'device: {device}')
model_path: str = join(datadir(), 'artifact', 'facenet-tl_2023-10-15 14:46:51.187699', 'model.pth')
print(f'model path: {model_path}')
input_shape: int = 256
batch_size = 64
source_dir = join(datadir(), 'face_cropped')
print(f'judge file: {source_dir}')
dest_dir = join(datadir(), 'infer_all')
image_class = ImageFolder(root=join(datadir(), 'dataset', 'train')).classes
with open(join(datadir(), 'class_text'), mode='w') as f:
    f.write(str(image_class))
rmtree(dest_dir, ignore_errors=True)
makedirs(dest_dir)

transform = Compose([Resize(size=256), ToTensor()])
image_folder = ImageFolder(root=source_dir, transform=transform)
dataloader = DataLoader(image_folder, batch_size=batch_size, shuffle=False, num_workers=8)

model = load(f=model_path)
model = model.to(device)
model.eval()
for layer in model.parameters():
    layer.requires_grad = False

# summary(model=model, input_size=(batch_size, 3, input_shape, input_shape), device=device)

if exists(join(datadir(), 'infer_all_torch_trt.ts')):
    trt_model = jit.load(join(datadir(), 'infer_all_torch_trt.ts'))
else:

    example_input = randn(size=[batch_size, 3, 256, 256]).float().cuda()
    traced_script_module = jit.trace(model, example_inputs=[example_input])
    trt_model = compile(module=traced_script_module, inputs=[
        Input(
            min_shape=[1, 3, 256, 256],
            opt_shape=[batch_size, 3, 256, 256],
            max_shape=[batch_size, 3, 256, 256]
        )
    ],
                        enabled_precisions={float32},
                        truncate_long_and_double=True,
                        allow_shape_tensors=True)
    jit.save(trt_model, join(datadir(), 'infer_all_torch_trt.ts'))

# heatmap_df = DataFrame(index=image_class, columns=image_folder.classes).fillna(0)
begin = time.time()
with ThreadPoolExecutor(max_workers=60) as executor, no_grad():
    for (images, labels), fileinfo in zip(tqdm(dataloader), chunked(image_folder.imgs, n=batch_size)):
        # print(labels, fileinfo)
        res = trt_model(images.to(device))
        for name, (filename, person) in zip(res.to(device).max(1).indices.tolist(), fileinfo):
            if not exists(join(dest_dir, image_class[name])):
                makedirs(join(dest_dir, image_class[name]), exist_ok=True)
            # print(name, filename, person)
            # copyfile(src=filename,
            #          dst=join(dest_dir, image_folder.classes[name], basename(filename)))
            # if image_class[name] != image_folder.classes[person]:
            #     heatmap_df[image_folder.classes[person]][image_class[name]] += 1
            executor.submit(copyfile, filename, join(dest_dir, image_class[name], basename(filename)))
print(f"{time.time() - begin:5f}sec")
# print(heatmap_df)
# set_palette('Blues')
# pyplot.figure(figsize=(40, 40))
# heat_img = heatmap(heatmap_df, cmap='Blues', linewidths=1)
# japanize()
# heatmap_df.max()
# pyplot.savefig(join(dest_dir, 'confusion_matrix.png'))
# print(f'acc: {1 - heatmap_df.to_numpy().flatten().sum() / image_folder.__len__()}')
print(image_folder.classes)
