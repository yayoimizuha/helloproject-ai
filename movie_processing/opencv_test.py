from cv2 import VideoCapture, getBuildInformation
from torchvision.models.mobilenetv3 import MobileNetV3

print(getBuildInformation())
sample_video = VideoCapture('/home/tomokazu/PycharmProjects/helloproject-ai/koi_ing.webm')
assert sample_video.isOpened()

ret = True
while ret:
    ret, frame = sample_video.read()
    # print(frame)
