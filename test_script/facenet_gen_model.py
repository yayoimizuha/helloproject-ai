from os.path import join

from facenet_pytorch import InceptionResnetV1
from torchinfo import summary
from torch import save

from settings import datadir

model = InceptionResnetV1(pretrained='vggface2')
model.eval()
summary(model=model, input_size=(1, 3, 256, 256))
print(model)
for name, layer in model.named_parameters():
    print(name)
save(model.cpu(), f=join(datadir(), 'artifact', 'vggface2_facenet.pth'))
