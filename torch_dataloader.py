from torchvision.transforms import Compose
import matplotlib.pyplot as plt
from numpy import transpose
import torchvision.datasets as datasets
from torchvision import models
from PIL import Image

print(dir(datasets))


def show_image(x):
    fig = plt.figure(figsize=(10, 10))
    for s in range(len(x)):
        img = x[s].numpy()
        img = transpose(img, (1, 2, 0))
        ax1 = fig.add_subplot(1, len(x), s + 1)
        plt.axis('off')
        plt.imshow(img)


model = models.inception_v3(pretrained=True)

print(model)
