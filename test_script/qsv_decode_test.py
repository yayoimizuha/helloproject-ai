import ctypes
import math
import numpy
from numpy.lib._stride_tricks_impl import as_strided
from matplotlib import pyplot
from test_ext import decode

with open(file=r"C:\Users\tomokazu\すぐ消す\friends-4385686.jpg", mode="rb") as f:
    ptr, height, width, pitch = decode(f.read())

pitch_h = math.ceil(height / 2) * 2
pitch_w = math.ceil(width / 2) * 2

print(height, width, pitch)
y_arr = numpy.frombuffer((ctypes.c_uint8 * (pitch_h * pitch)).from_address(ptr), dtype=numpy.uint8,
                         count=pitch_h * pitch)
print(f"{y_arr=}")
uv_arr = numpy.frombuffer((ctypes.c_uint8 * (int(pitch_h * 1.5) * pitch)).from_address(ptr),
                          dtype=numpy.uint8,
                          count=int(pitch_h / 2) * pitch, offset=pitch_h * pitch)
print(f"{uv_arr=}")

y_plane = as_strided(y_arr, (pitch_h, pitch_w), (pitch, 1))
uv_plane = as_strided(uv_arr, (int(pitch_h / 2), int(pitch_w / 2), 2), (pitch, 2, 1))
yuv_plane = numpy.stack([y_plane,
                         uv_plane[:, :, 0].repeat(2, axis=0).repeat(2, axis=1),
                         uv_plane[:, :, 1].repeat(2, axis=0).repeat(2, axis=1)])
# print(y_plane.shape)
# print(y_plane.strides)
# print(uv_plane.shape)
# print(uv_plane.strides)
# print(uv_plane[:, :, 0].shape)
print(yuv_plane.shape)
print(yuv_plane.strides)
# print(yuv_plane[:, : 4, : 4])
# print(yuv_plane.transpose(1, 2, 0)[:4, :4, :])
pyplot.figure(figsize=(20, 20), dpi=150)
pyplot.imshow(yuv_plane[0, :, :])
pyplot.show()
pyplot.close("all")
pyplot.figure(figsize=(20, 20), dpi=150)
pyplot.imshow(yuv_plane[1, :, :])
pyplot.show()
pyplot.close("all")
pyplot.figure(figsize=(20, 20), dpi=150)
pyplot.imshow(yuv_plane[2, :, :])
pyplot.show()
pyplot.close("all")
ycbcr_mat = yuv_plane.transpose((1, 2, 0)).reshape((-1, 3)) - [0, 128, 128]
# print(ycbcr_mat)
transform_matrix = numpy.array([
    [1.0, 0.0, 1.5748],
    [1.0, -0.1873, -0.4681],
    [1.0, 1.8556, 0.0]
])
rgb_plane = (numpy.clip(numpy.dot(ycbcr_mat, transform_matrix.T), 0, 255)
             .reshape(pitch_h, pitch_w, 3).astype(numpy.uint8))
pyplot.figure(figsize=(20, 20), dpi=150)
pyplot.imshow(rgb_plane)
pyplot.show()
pyplot.close("all")
