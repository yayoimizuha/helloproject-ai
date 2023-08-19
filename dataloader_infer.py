from os import makedirs
from os.path import join, exists, basename
from shutil import rmtree, copyfile
from more_itertools import chunked
from torch import load, no_grad, device
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

device = device('cuda' if is_available() else 'cpu')
# device = 'cpu'
print(f'device: {device}')
model_path: str = join(datadir(), 'artifact', 'facenet-tl_2023-06-03 23:48:19.808311', 'model.pth')
print(f'model path: {model_path}')
input_shape: int = 256
batch_size = 64
source_dir = join(datadir(), 'dataset', 'val')
print(f'judge file: {source_dir}')
dest_dir = join(datadir(), 'test_infer')
image_class = ImageFolder(root=join(datadir(), 'dataset', 'train')).classes
with open(join(datadir(), 'class_text'), mode='w') as f:
    f.write(str(image_class))
rmtree(dest_dir)
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

heatmap_df = DataFrame(index=image_class, columns=image_folder.classes).fillna(0)
with ThreadPoolExecutor(max_workers=60) as executor, no_grad():
    for (images, labels), fileinfo in zip(tqdm(dataloader), chunked(image_folder.imgs, n=batch_size)):
        # print(labels, fileinfo)
        res = model(images.to(device))
        for name, (filename, person) in zip(res.to(device).max(1).indices.tolist(), fileinfo):
            if not exists(join(dest_dir, image_class[name])):
                makedirs(join(dest_dir, image_class[name]), exist_ok=True)
            # print(name, filename, person)
            # copyfile(src=filename,
            #          dst=join(dest_dir, image_folder.classes[name], basename(filename)))
            if image_class[name] != image_folder.classes[person]:
                heatmap_df[image_folder.classes[person]][image_class[name]] += 1
            executor.submit(copyfile, filename, join(dest_dir, image_class[name], basename(filename)))

print(heatmap_df)
set_palette('Blues')
pyplot.figure(figsize=(40, 40))
heat_img = heatmap(heatmap_df, cmap='Blues', linewidths=1)
japanize()
heatmap_df.max()
pyplot.savefig(join(dest_dir, 'confusion_matrix.png'))
print(f'acc: {1 - heatmap_df.to_numpy().flatten().sum() / image_folder.__len__()}')
