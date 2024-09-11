import cupy
import numpy
import os
import shutil

TYPE = "facenet"

DEST_ROOT = r"D:\helloproject-ai-data\similar_face"
SAMPLE_FILE = r"D:\helloproject-ai-data\face_cropped\井上春華\井上春華=morningmusume16ki=12815441476-0-0.jpg"
PICK_N = 500
PROBE = 0.6
embeddings = cupy.load(rf"C:\Users\tomokazu\PycharmProjects\helloproject-ai\embeddings_{TYPE}.npy")
embeddings_label = numpy.load(rf"C:\Users\tomokazu\PycharmProjects\helloproject-ai\embeddings_{TYPE}_label.npy")

sample_pos = numpy.argwhere(embeddings_label == os.path.basename(SAMPLE_FILE))
sample_emb = embeddings[*sample_pos, :].T
print(sample_pos)
print(embeddings_label.shape)
print(embeddings.shape)
print(sample_emb.shape)
similarity: cupy.ndarray = cupy.dot(embeddings, sample_emb).T / (
        cupy.linalg.norm(embeddings, axis=1) * cupy.linalg.norm(sample_emb))
# sample_norm = cupy.linalg.norm(sample_emb)
# similarity = cupy.apply_along_axis(lambda x: cupy.dot(x, sample_emb) / (cupy.linalg.norm(x) * sample_norm), 1,
#                                    embeddings)
print(similarity.shape)
similar_pos: cupy.ndarray = similarity.reshape((-1,)).argsort()[::-1]
print(similar_pos)
p = int(cupy.count_nonzero(cupy.argwhere(similarity > PROBE)))
# p = 300
sim_list = embeddings_label[cupy.asnumpy(similar_pos)[:p]]
print(numpy.stack([sim_list, cupy.asnumpy(similarity.reshape((-1,))[similar_pos])[:p]], axis=1))
os.makedirs(os.path.join(DEST_ROOT, os.path.splitext(os.path.basename(SAMPLE_FILE))[0] + f"_{TYPE}"), exist_ok=True)
for file in sim_list:
    shutil.copyfile(os.path.join(r"D:\helloproject-ai-data\face_cropped", file.split("=")[0], file),
                    os.path.join(DEST_ROOT, os.path.splitext(os.path.basename(SAMPLE_FILE))[0] + f"_{TYPE}", file))
print(f"Copied {p} file(s).")
