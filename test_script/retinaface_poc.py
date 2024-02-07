from retinaface.pre_trained_models import get_model
from PIL import Image
from numpy import array

image = Image.open("./test_script/manaka_test.jpg")
model = get_model("resnet50_2020-07-20", max_size=2048)
model.eval()
print(model.predict_jsons(array(image)))
