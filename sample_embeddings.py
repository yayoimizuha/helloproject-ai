from insightface.app import FaceAnalysis
from settings import datadir
from os import listdir
from os.path import join, isfile
from PIL import Image
from numpy import array
from json import dump

face_analysis = FaceAnalysis()
face_analysis.prepare(ctx_id=0, det_size=(160, 160))

teacher_dir = join(datadir(), 'sample_set')
teacher_files = []
teacher_embeddings = []
teacher_dict = {}
for name in listdir(teacher_dir):
    for file in listdir(join(teacher_dir, name)):
        # print(file)
        im_path = join(teacher_dir, name, file)
        if isfile(im_path):
            print(im_path)
            image = Image.open(im_path)
            embedding = face_analysis.get(array(image)[:, :, [2, 1, 0]])
            if embedding.__len__() != 0:
                teacher_embeddings.append(embedding[0].embedding.tolist())
                teacher_files.append(name)
                teacher_dict[im_path] = embedding[0].embedding.tolist()

with open(file=join(datadir(), 'sample_emb.json'), mode='w') as f:
    dump(teacher_dict, f)
