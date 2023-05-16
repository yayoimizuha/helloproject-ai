from super_gradients.training.models import get

yolo_nas = get(model_name='yolo_nas_l', pretrained_weights='coco').cuda()

yolo_nas.predict('橋迫鈴=angerme-new=12687767841-1.jpg', conf=0.8).show()
