from shutil import copyfile
from time import sleep

from insightface.app import FaceAnalysis
from sklearn.neighbors import NearestNeighbors
from json import load
from os import listdir, makedirs
from os.path import join, isfile, isdir, basename, exists
from settings import datadir
from PIL import Image
from numpy import array, sqrt

with open(file=join(datadir(), 'sample_emb.json'), mode='r') as f:
    sample_dict: dict = load(f)

sample_key = [basename(s).split('=')[0] for s in sample_dict.keys()]
sample_emb = list(sample_dict.values())
del sample_dict

neighbors = NearestNeighbors(n_neighbors=15, n_jobs=-1, metric='cosine')
# print(sample_emb[0])
neighbors.fit(sample_emb)

makedirs(join(datadir(), 'NN_predict'), exist_ok=True)

face_analysis = FaceAnalysis(allowed_modules=['recognition', 'detection'])
face_analysis.prepare(ctx_id=0, det_size=(160, 160))

for name in listdir(join(datadir(), 'face_cropped')):
    for file in listdir(join(datadir(), 'face_cropped', name)):
        im_path = join(datadir(), 'face_cropped', name, file)
        if not isfile(im_path):
            continue
        image = array(Image.open(im_path))[:, :, [2, 1, 0]]
        emb = face_analysis.get(image)
        if emb.__len__() == 0:
            continue
        distances, nears = neighbors.kneighbors([emb[0].embedding], return_distance=True)
        # print(nears)
        # print(distances)
        scores = dict()
        for dist, key_id in zip(distances[0], nears[0]):
            if key_id != 0.0:
                if not sample_key[key_id] in scores.keys():
                    scores[sample_key[key_id]] = 0.0
                scores[sample_key[key_id]] += 1 / sqrt(dist)
        # print(scores)
        print(im_path)
        val_sum = sum(scores.values())
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for key, reliability in sorted_scores:
            print(f'{key:<7}\t{(reliability * 100 / val_sum):05.2f}%')

        if sorted(distances[0], reverse=False)[0] < 0.4:
            prediction = sorted_scores[0][0]
        elif sorted_scores[0][1] / val_sum > 0.6:
            prediction = sorted_scores[0][0]
        else:
            prediction = 'others'
        print(prediction)  # , sorted(distances[0], reverse=False))
        # print(sorted_scores)
        # print(prediction)

        if not exists(join(datadir(), 'NN_predict', prediction)):
            makedirs(join(datadir(), 'NN_predict', prediction))
        copyfile(im_path, join(datadir(), 'NN_predict', prediction, basename(im_path)))

        print('\n\n')
        # sleep(1)
    # exit()
