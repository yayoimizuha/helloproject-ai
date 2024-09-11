import gradio
import pathlib
import cupy
import numpy
import os

TYPE = "facenet"

embeddings = cupy.load(rf"C:\Users\tomokazu\PycharmProjects\helloproject-ai\embeddings_{TYPE}.npy")
embeddings_label = numpy.load(rf"C:\Users\tomokazu\PycharmProjects\helloproject-ai\embeddings_{TYPE}_label.npy")


def get_similar_pic(_input: pathlib.Path):
    search_key = os.path.basename(_input)
    key_pos = numpy.argwhere(embeddings_label == search_key)
    search_emb = embeddings[*key_pos, :].T
    similarity_array: cupy.ndarray = (cupy.dot(embeddings, search_emb).T /
                                      (cupy.linalg.norm(embeddings, axis=1) * cupy.linalg.norm(search_emb)))
