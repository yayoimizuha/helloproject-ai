from cv2 import VideoCapture, getBuildInformation
from torchvision.models.mobilenetv3 import MobileNetV3
from settings import datadir

print(getBuildInformation())
sample_video = VideoCapture('/movie_processing/koi_ing.webm')
assert sample_video.isOpened()

ret = True
while ret:
    ret, frame = sample_video.read()
    # print(frame)
