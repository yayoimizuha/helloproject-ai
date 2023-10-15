from torch import zeros, load, no_grad, stack, float32, nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, functional, ToTensor, Resize, ConvertImageDtype
from PIL import Image
from numpy import array
from os import listdir, makedirs
from settings import datadir
from os.path import join, isdir, isfile, basename
from tqdm import tqdm
from more_itertools import chunked
from shutil import copyfile

transform = Compose([
    ToTensor(),
    #     ConvertImageDtype(float32),
    Resize(224, antialias=True)
])

image_folder = ImageFolder(root=join(datadir(), 'face_cropped'), transform=transform)
print(image_folder.__len__())
dataloader = DataLoader(image_folder, batch_size=300, shuffle=False, num_workers=14, pin_memory=True)

dest_dir = join(datadir(), 'masked_or_not', 'infer')
makedirs(dest_dir, exist_ok=True)
mask_status = ('no', 'yes')
[makedirs(join(dest_dir, p)) for p in mask_status]
[makedirs(join(dest_dir, p, name)) for p in mask_status for name in image_folder.classes]

model = load(join(datadir(), 'artifact', 'masked_or_not_2023-04-30 19:38:14.398157/model.pth')).cuda()
model.eval()
with no_grad():
    for (images, labels), fileinfo in zip(tqdm(dataloader), chunked(image_folder.imgs, n=300)):
        # print(fileinfo)
        res = model(images.cuda())
        for is_mask, (filename, person) in zip(res.max(1).indices.tolist(), fileinfo):
            # print(is_mask,image_folder.classes[person],filename)
            copyfile(filename, join(dest_dir, mask_status[is_mask], image_folder.classes[person], basename(filename)))
        # break
