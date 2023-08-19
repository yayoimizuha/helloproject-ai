from os.path import join
from matplotlib.pyplot import imshow, show, figure
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor
from numpy import ndarray, ceil, full, uint8
from torchvision.transforms import Compose, CenterCrop, RandomHorizontalFlip, GaussianBlur, RandomAutocontrast, \
    ToTensor, RandomRotation, RandomResizedCrop, RandomErasing, RandomEqualize, RandomPerspective, RandomPosterize, \
    RandomGrayscale

from settings import datadir


def plot_dataset(dataloader: DataLoader | tuple, col_len: int = 8,
                 label_text: str | None = None) -> Image.Image:
    if isinstance(dataloader, DataLoader):
        images, labels = iter(dataloader).__next__()
    else:
        images, labels = dataloader

    images: Tensor = images
    labels: Tensor = labels
    images: ndarray = images.numpy()

    if label_text is None:
        labels: list[str] = [str(i) for i in labels.tolist()]
    else:
        labels: list[str] = [label_text[i] for i in labels.tolist()]

    batch_size, _, width, height = images.shape
    rows = ceil(batch_size / col_len)
    space_y, space_x, font_size = 50, 30, 20
    shape_y, shape_x = images.shape[-2:]
    base_img = full(shape=((height + space_y) * int(rows), width * col_len + space_x * (col_len - 1), 3), dtype=uint8,
                    fill_value=255)
    for order, image in enumerate(images):
        order_y, order_x = order // col_len, order % col_len
        image = (image.transpose([1, 2, 0]) * 255).astype(uint8)
        base_img[order_y * (shape_y + space_y) + space_y:(order_y + 1) * (shape_y + space_y),
        order_x * (shape_x + space_x):(order_x + 1) * (shape_x + space_x) - space_x, :] = image
    pil_image = Image.fromarray(base_img)
    font = ImageFont.truetype(font=r'/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc', size=24)
    draw = ImageDraw.Draw(pil_image)
    pad = 5
    for order, label in enumerate(labels):
        order_y, order_x = order // col_len, order % col_len
        draw.text(((shape_x + space_x) * order_x + pad, (shape_y + space_y) * order_y + pad), label, 'black', font=font)

    return pil_image


transform = Compose([
    RandomGrayscale(p=.25),
    RandomHorizontalFlip(p=0.2),
    # GaussianBlur(kernel_size=3),
    RandomAutocontrast(),
    # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    RandomEqualize(p=.25),
    RandomPosterize(bits=4),
    ToTensor(),
    RandomRotation(degrees=30, fill=1),
    RandomPerspective(fill=1, distortion_scale=.2),
    RandomErasing(scale=(0.05, 0.1), value='random', p=.3),
    RandomResizedCrop(size=224, scale=(0.7, 1.0), ratio=(1.0, 1.0), antialias=True)
])
image_folder = ImageFolder(root=join(datadir(), 'dataset', 'train'), transform=transform)

dataloader = DataLoader(image_folder, batch_size=36, shuffle=True, num_workers=3)

figure(figsize=(10, 10), dpi=300)
imshow(plot_dataset(dataloader=dataloader, col_len=6, label_text=image_folder.classes))
show()
print(image_folder.classes)