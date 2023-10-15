from os import makedirs
import numpy
import torch.backends.cudnn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torch.nn import Linear
from torchvision.transforms import Compose, RandomResizedCrop, RandomRotation, ToTensor, \
    RandomHorizontalFlip, \
    Resize, RandomAffine, RandomAdjustSharpness, RandomAutocontrast, RandomEqualize, GaussianBlur, Normalize
from numpy import arange, ceil, full, float32, uint8, amax, amin
from torchinfo import summary
from torch.nn import CrossEntropyLoss
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp import autocast

from torch.optim import SGD, Adam, lr_scheduler
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm
from settings import datadir
from os.path import join
from torch.cuda import is_available
from torch import no_grad, save
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import matplotlib

matplotlib.use('Agg')
device = 'cuda' if is_available() else 'cpu'
torch.backends.cudnn.deterministic = True

transform = {
    'train': Compose([
        # CenterCrop(200),
        RandomHorizontalFlip(p=0.1),
        RandomAdjustSharpness(sharpness_factor=2, p=0.2),
        GaussianBlur(kernel_size=3),
        RandomAutocontrast(),
        RandomEqualize(p=0.5),
        RandomRotation(degrees=15),
        ToTensor(),
        # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        Resize(size=224, antialias=True)
    ]),
    'val': Compose([
        # CenterCrop(200),
        ToTensor(),
        # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        Resize(size=224, antialias=True)
    ])
}

image_folder = {
    'train': ImageFolder(root=join(datadir(), 'masked_or_not', 'train'), transform=transform['train']),
    'val': ImageFolder(root=join(datadir(), 'masked_or_not', 'val'), transform=transform['val'])
}

dataloader = {
    'train': DataLoader(image_folder['train'], batch_size=16, shuffle=True, num_workers=8),
    'val': DataLoader(image_folder['val'], batch_size=16, shuffle=False, num_workers=8)
}


def plot_dataset(dataloader: DataLoader | tuple, col_len: int = 8,
                 label_text: str | None = None) -> Image.Image:
    if isinstance(dataloader, DataLoader):
        images, labels = iter(dataloader).__next__()
    else:
        images, labels = dataloader

    images: torch.Tensor = images
    labels: torch.Tensor = labels
    images: numpy.ndarray = images.numpy()

    if label_text is None:
        labels: list[str] = [str(i) for i in labels.tolist()]
    else:
        labels: list[str] = [label_text[i] for i in labels.tolist()]

    batch_size, _, width, height = images.shape
    # print(batch_size, width, height)
    # print(images.dtype)
    rows = ceil(batch_size / col_len)
    # print(amax(images), amin(images))
    space_y, space_x, font_size = 50, 30, 20
    shape_y, shape_x = images.shape[-2:]
    base_img = full(shape=((height + space_y) * int(rows), width * col_len + space_x * (col_len - 1), 3), dtype=uint8,
                    fill_value=255)
    for order, image in enumerate(images):
        order_y, order_x = order // col_len, order % col_len
        image = (image.transpose([1, 2, 0]) * 255).astype(uint8)
        # print(order_y, order_x)
        # print(order_y * (shape_y + 30) + 30, (order_y + 1) * (shape_y + 30),
        #       order_x * (shape_x + 20), (order_x + 1) * (shape_x + 20) - 20)
        base_img[order_y * (shape_y + space_y) + space_y:(order_y + 1) * (shape_y + space_y),
        order_x * (shape_x + space_x):(order_x + 1) * (shape_x + space_x) - space_x, :] = image
    pil_image = Image.fromarray(base_img)
    font = ImageFont.truetype(font='NotoSansCJK-Medium.ttc', size=24)
    draw = ImageDraw.Draw(pil_image)
    pad = 5
    for order, label in enumerate(labels):
        order_y, order_x = order // col_len, order % col_len
        draw.text(((shape_x + space_x) * order_x + pad, (shape_y + space_y) * order_y + pad), label, 'black', font=font)

    return pil_image
    # pyplot.imshow((images[0].transpose([1, 2, 0]) * 255).astype(uint8))


model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)

# print(model)
summary(model=model, input_size=(3, 224, 224), device='cpu')

for layer_name, layer in model.named_parameters():
    # print(layer_name)
    layer.requires_grad = False
    if 'classifier' in layer_name:
        layer.requires_grad = True
    if 'features.12' in layer_name:
        layer.requires_grad = True
    if 'features.11' in layer_name:
        layer.requires_grad = True
    if 'features.10' in layer_name:
        layer.requires_grad = True

model.classifier[3] = Linear(in_features=1024, out_features=image_folder['train'].classes.__len__(), bias=True)

model_gpu = model.cuda()
criterion = CrossEntropyLoss().cuda()

optimizer = Adam(params=[
    {'params': model_gpu.classifier.parameters(), 'lr': 1e-3},
    {'params': model_gpu.features[12].parameters(), 'lr': 1e-4},
    {'params': model_gpu.features[11].parameters(), 'lr': 1e-4},
    {'params': model_gpu.features[10].parameters(), 'lr': 1e-4},
])

scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=5, gamma=.5)
epochs = 50

train_loss_list = list()
train_acc_list = list()
val_loss_list = list()
val_acc_list = list()

save_dir = join(datadir(), 'artifact', 'masked_or_not_' + datetime.now().__str__())
print(save_dir)
makedirs(save_dir, exist_ok=True)
makedirs(join(save_dir, 'pallets'), exist_ok=True)

scaler = GradScaler()

for epoch in range(epochs):
    train_loss = .0
    train_acc = .0
    val_loss = .0
    val_acc = .0

    model_gpu.train()
    makedirs(join(save_dir, 'pallets', str(epoch)), exist_ok=True)
    for count, (images, labels) in enumerate(tqdm(dataloader['train'])):
        image_pallets = plot_dataset(dataloader=(images, labels), col_len=4, label_text=image_folder['train'].classes)
        image_pallets.save(join(save_dir, 'pallets', str(epoch), str(count) + '.jpg'))
        optimizer.zero_grad()
        images = images.cuda()
        labels = labels.cuda()
        with autocast():
            outputs = model_gpu(images)
            loss = criterion(outputs, labels)
        train_loss += loss.item()

        # loss.backward()
        # optimizer.step()
        scaler.scale(loss).backward()
        scaler.step(optimizer=optimizer)
        scaler.update()
        predicted = outputs.max(1)[1]
        train_acc += (predicted == labels).sum()

    avg_train_loss = train_loss / dataloader['train'].dataset.__len__()
    avg_train_acc = train_acc / dataloader['train'].dataset.__len__()

    model_gpu.eval()
    with no_grad():
        for images, labels in dataloader['val']:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model_gpu(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            predicted = outputs.max(1)[1]
            val_acc += (predicted == labels).sum()
        avg_val_loss = val_loss / dataloader['val'].dataset.__len__()
        avg_val_acc = val_acc / dataloader['val'].dataset.__len__()

    print(f'Epoch [{(epoch + 1):02}/{epochs}], loss: {avg_train_loss:.5f}, '
          f'acc: {avg_train_acc:.5f}, val_loss: {avg_val_loss:.5f}, val_acc: {avg_val_acc:.5f}, '
          f'lr: {scheduler.get_last_lr()[0]:.2e}')
    scheduler.step()

    train_loss_list.append(float(avg_train_loss))
    train_acc_list.append(float(avg_train_acc))
    val_loss_list.append(float(avg_val_loss))
    val_acc_list.append(float(avg_val_acc))

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.plot(val_acc_list, label='val', lw=2, c='b')
    plt.plot(train_acc_list, label='train', lw=2, c='k')
    plt.title('learning rate')
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.grid(lw=2)
    plt.legend(fontsize=14, bbox_to_anchor=(1, 0))
    plt.xticks(arange(1, epochs, 2))
    plt.savefig(join(save_dir, 'learning_rate.png'))
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(val_loss_list, label='val', lw=2, c='b')
    plt.plot(train_loss_list, label='train', lw=2, c='k')
    plt.title('loss')
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.grid(lw=2)
    plt.legend(fontsize=14, bbox_to_anchor=(1, 1))
    plt.xticks(arange(1, epochs, 2))
    plt.savefig(join(save_dir, 'loss.png'))
    plt.close()
    del plt

save(model_gpu.cpu(), join(save_dir, 'model.pth'))
