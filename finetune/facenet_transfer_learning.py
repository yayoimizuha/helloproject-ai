from os import makedirs, environ
from torchinfo import summary
from torch.nn import Linear, Sequential, Dropout, CrossEntropyLoss, Identity, ReLU
from torchvision.transforms import Compose, RandomResizedCrop, RandomRotation, ToTensor, \
    RandomHorizontalFlip, \
    Resize, RandomAutocontrast, InterpolationMode, RandomErasing, \
    RandomEqualize, RandomPosterize, RandomPerspective, RandomGrayscale
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy import arange, ndarray, ceil, full, uint8
from torch.optim import SGD, Adam, lr_scheduler
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from settings import datadir
from os.path import join
from torch.cuda import is_available
from torch import no_grad, save, Tensor, load, device, float16
from datetime import datetime
from distutils.util import strtobool

CI = bool(strtobool(environ['CI']))
device = device('cuda' if is_available() else 'cpu')

model_path: str = join(datadir(), 'artifact', 'vggface2_facenet.pth')
input_shape: int = 256
batch_size = 32

transform = {
    'train': Compose([
        RandomGrayscale(p=.25),
        RandomHorizontalFlip(p=0.2),
        RandomAutocontrast(),
        RandomEqualize(p=.25),
        RandomPosterize(bits=4),
        ToTensor(),
        RandomRotation(degrees=30, fill=1),
        RandomPerspective(fill=1, distortion_scale=.2),
        RandomErasing(scale=(0.05, 0.1), value='random', p=.3),
        RandomResizedCrop(size=224, scale=(0.7, 1.0), ratio=(1.0, 1.0), antialias=True)
    ]),
    'val': Compose([
        # RandomAffine(scale=(0.8, 0.8), degrees=(0, 0), fill=1),
        Resize(224, antialias=True, interpolation=InterpolationMode.BILINEAR),
        ToTensor()
    ])
}
image_folder = {
    'train': ImageFolder(root=join(datadir(), 'dataset', 'train'), transform=transform['train']),
    'val': ImageFolder(root=join(datadir(), 'dataset', 'val'), transform=transform['val'])
}

dataloader = {
    'train': DataLoader(image_folder['train'], batch_size=batch_size, shuffle=True, num_workers=8),
    'val': DataLoader(image_folder['val'], batch_size=batch_size, shuffle=True, num_workers=8)
}


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


model = load(model_path)

tune = False
for name, layer in model.named_parameters():
    if 'block8' in name:
        tune = True
    layer.requires_grad = tune

model.last_linear = Identity()
model.last_bn = Identity()
model.logits = Identity()
model.dropout = Identity()
model = Sequential(model,
                   Linear(in_features=1792, out_features=2 ** 12), ReLU(inplace=True), Dropout(),
                   Linear(in_features=2 ** 12, out_features=2 ** 11), ReLU(inplace=True), Dropout(),
                   Linear(in_features=2 ** 11, out_features=image_folder['train'].classes.__len__(), bias=True),
                   )

summary(model=model, input_size=(batch_size, 3, input_shape, input_shape), device='cpu')

model_gpu = model.to(device=device)
criterion = CrossEntropyLoss()

optimizer = Adam(params=[
    {'params': model_gpu[0].block8.parameters(), 'lr': 1e-5},
    {'params': model_gpu[1].parameters(), 'lr': 1e-3},
])

scaler = GradScaler()

# model, optimizer = optimize(model=model, optimizer=optimizer)
scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.9)
epochs = 100

train_loss_list = list()
train_acc_list = list()
val_loss_list = list()
val_acc_list = list()

save_dir = join(datadir(), 'artifact', 'facenet-tl_' + datetime.now().__str__())
print(save_dir)
makedirs(save_dir, exist_ok=True)
makedirs(join(save_dir, 'pallets'), exist_ok=True)
makedirs(join(save_dir, 'checkpoints'), exist_ok=True)

for epoch in range(epochs):
    train_loss = .0
    train_acc = .0
    val_loss = .0
    val_acc = .0

    model_gpu.train()
    # makedirs(join(save_dir, 'pallets', str(epoch)), exist_ok=True)

    for count, (images, labels) in enumerate(tqdm(dataloader['train'], disable=None)):
        if count == 1:
            image_pallets = plot_dataset(dataloader=(images, labels), col_len=6,
                                         label_text=image_folder['train'].classes)
            image_pallets.save(join(save_dir, 'pallets', str(epoch) + '_train.jpg'))
        optimizer.zero_grad()
        with autocast(dtype=float16):
            images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)
        train_loss += loss.item()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer=optimizer)
        # loss.backward()
        scaler.step(optimizer=optimizer)
        scaler.update()
        # optimizer.step()

        predicted = outputs.max(1)[1]
        train_acc += (predicted == labels).sum()

    avg_train_loss = train_loss / dataloader['train'].dataset.__len__()
    avg_train_acc = train_acc / dataloader['train'].dataset.__len__()

    model_gpu.eval()
    with no_grad():
        for count, (images, labels) in enumerate(dataloader['val']):
            if count == 1:
                image_pallets = plot_dataset(dataloader=(images, labels), col_len=6,
                                             label_text=image_folder['train'].classes)
                image_pallets.save(join(save_dir, 'pallets', str(epoch) + '_val.jpg'))
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

    plt.figure(figsize=(8, 6))
    plt.plot(val_acc_list, label='val', lw=2, c='b')
    plt.plot(train_acc_list, label='train', lw=2, c='k')
    plt.title('learning rate')
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.xlim([-int(epochs * .1), int(epochs * 1.1)])
    plt.ylim([-0.1, 1.1])
    plt.grid(lw=2)
    plt.legend(fontsize=14)
    plt.xticks(arange(0, epochs + 1, 10))
    plt.yticks(arange(0, 1.1, .1))
    plt.savefig(join(save_dir, 'learning_rate.png'))
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(val_loss_list, label='val', lw=2, c='b')
    plt.plot(train_loss_list, label='train', lw=2, c='k')
    plt.title('loss')
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.xlim([-int(epochs * .1), int(epochs * 1.1)])
    plt.ylim(bottom=-0)
    plt.grid(lw=2)
    plt.legend(fontsize=14)
    plt.xticks(arange(0, epochs + 1, 10))
    plt.savefig(join(save_dir, 'loss.png'))
    plt.close()
    if (epoch + 1) % 10 == 0:
        save(model_gpu.cpu(), join(save_dir, 'checkpoints', f'{epoch + 1}.pth'))
        model.to(device=device)

save(model_gpu.cpu(), join(save_dir, 'model.pth'))
