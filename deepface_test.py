from PIL import Image, ImageDraw

from matplotlib import pyplot
from retinaface.pre_trained_models import get_model
from torch import save
from retinaface.predict_single import Model
from retinaface.network import RetinaFace
from numpy import array

im_path = "道重さゆみ=sayumimichishige-blog=12719215891-10.jpg"
model: Model = get_model(model_name='resnet50_2020-07-20', max_size=1024, device='cuda')
image = array(Image.open(im_path))
faces = model.predict_jsons(image, confidence_threshold=0.4, nms_threshold=0.4)

print(faces)
# print(dfs)
image = Image.open(im_path)
draw = ImageDraw.Draw(image)
if faces.__len__() == 1:
    if faces[0]['score'] == -1:
        exit()
for face in faces:
    print(face['bbox'])
    x, y, w, h = face['bbox']
    print(x, y, w, h)
    draw.rectangle((x, y, w, h))
    for order, landmark in enumerate(face['landmarks'], start=1):
        draw.point(tuple(landmark))
        draw.text(tuple(landmark), text=str(order))
#
image.save("dest.png")
# RetinaFace.detect_faces(im_path)
