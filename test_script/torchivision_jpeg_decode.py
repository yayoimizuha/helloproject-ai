import os
from concurrent.futures.process import ProcessPoolExecutor
from multiprocessing import shared_memory
from io import BytesIO
from os import listdir, path, pathsep, makedirs
from pprint import pprint
import more_itertools
import numpy as np
import tqdm
from PIL import Image
from numpy import ndarray
from onnxruntime import InferenceSession
from torch import tensor
import aiofiles
import numpy
import torch
from torchvision.io import decode_jpeg
from asyncio import run, gather, Semaphore
from site import getsitepackages
from rust_retinaface_post_processor import resnet_post_process

os.environ["Path"] = path.join(getsitepackages()[-1], "tensorrt_libs") + pathsep + os.environ["Path"]
root_dir = r"D:\helloproject-ai-data\blog_images"
model_path = r"C:\Users\tomokazu\build\retinaface\retinaface_only_nn_fp16.onnx"
# makedirs("memmap", exist_ok=True)

files = []
files_data: dict[str, numpy.ndarray | None] = {}
chunk_size = 32
image_size = 640
device = torch.device("cuda")


async def async_read(path: str, semaphore: Semaphore):
    async with semaphore:
        async with aiofiles.open(file=path, mode="rb") as fp:
            return await fp.read()


async def gather_runner(l: list, fn):
    sem = Semaphore(2048)
    return await gather(*[fn(p, sem) for p in l])


def post_processor(outputs, batch_size, image_size):
    # print("aaa", flush=True)
    outputs = [numpy.ascontiguousarray(output.astype(numpy.float32)) for output in outputs]
    res = resnet_post_process([output.__array_interface__["data"][0] for output in outputs], batch_size, image_size)
    return res


def post_processor_memmap(tmp_filename, sizes, batch_size, image_size):
    # print("aaa", flush=True)
    outputs = [
        numpy.memmap(filename=path.join("memmap", tmp_filename + str(order)), dtype=numpy.float16, mode="r", shape=size)
        for order, size in enumerate(sizes)]
    outputs = [numpy.ascontiguousarray(output.astype(numpy.float32)) for output in outputs]
    res = resnet_post_process([output.__array_interface__["data"][0] for output in outputs], batch_size, image_size)
    return res


def post_processor_shm(shm_name, sizes, batch_size, image_size):
    shms = [shared_memory.SharedMemory(name=shm_name + "_" + str(i)) for i in range(3)]
    outputs = \
        [numpy.ascontiguousarray(numpy.ndarray(shape=size, dtype=numpy.float16, buffer=shm.buf).astype(numpy.float32))
         for size, shm in zip(sizes, shms)]
    res = resnet_post_process([output.__array_interface__["data"][0] for output in outputs], batch_size, image_size)
    [shm.close() for shm in shms]
    return res


if __name__ == '__main__':
    from kornia.augmentation import LongestMaxSize, PadTo, Normalize
    from kornia.constants import Resample

    longest_max_size = LongestMaxSize(max_size=640,resample=Resample.NEAREST)
    pad_to = PadTo(size=(640, 640), pad_value=1.)
    normalize = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    session = InferenceSession(
        path_or_bytes=model_path,
        providers=[
            ('TensorrtExecutionProvider', {
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': 'trt_cache',
                'trt_fp16_enable': True,
                'trt_profile_min_shapes': f'input:1x3x{image_size}x{image_size}',
                'trt_profile_max_shapes': f'input:{chunk_size}x3x{image_size}x{image_size}',
                'trt_profile_opt_shapes': f'input:{chunk_size}x3x{image_size}x{image_size}',
            }),
            'CUDAExecutionProvider',
            'CPUExecutionProvider'
        ]
    )

    for name in listdir(root_dir):
        with (ProcessPoolExecutor(max_workers=4) as executor):
            if name != "下井谷幸穂":
                # continue
                pass
            file_names = listdir(path.join(root_dir, name))
            name_files = [path.join(root_dir, name, file_name) for file_name in file_names]
            files_data = {file_name: numpy.frombuffer(dat, dtype=numpy.uint8) for file_name, dat in
                          zip(file_names, run(gather_runner(name_files, async_read)))}

            # print(k_1)
            for cnk in more_itertools.chunked(tqdm.tqdm(files_data.items(), desc=name), n=chunk_size):
                stack = []
                tmp_file_name = ""
                for file, dat in cnk:
                    tmp_file_name = file
                    try:
                        decoded_image = decode_jpeg(tensor(dat), device=device)
                    except:
                        decoded_image = tensor(numpy.array(Image.open(BytesIO(dat.tobytes()))).transpose([2, 0, 1])).to(
                            device)
                    decoded_image = decoded_image.to(torch.float16) / 255
                    decoded_image = normalize(decoded_image)
                    decoded_image_resized = longest_max_size(decoded_image)
                    decoded_image_padded = pad_to(decoded_image_resized)
                    stack.append(decoded_image_padded.squeeze())
                [stack.append(torch.zeros(size=[3, 640, 640], dtype=torch.float16, device=device)) for _ in
                 range(32 - stack.__len__())]
                stacked = torch.stack(stack).contiguous()
                # print(stacked.shape)
                io_binding = session.io_binding()
                io_binding.bind_input(
                    name="input",
                    device_type=stacked.device.type,
                    device_id=stacked.device.index,
                    element_type='float16',
                    shape=tuple(stacked.shape),
                    buffer_ptr=stacked.data_ptr()
                )
                io_binding.bind_output("landmark")
                io_binding.bind_output("confidence")
                io_binding.bind_output("bbox")
                session.run_with_iobinding(iobinding=io_binding)
                outputs: list[numpy.ndarray] = io_binding.copy_outputs_to_cpu()
                # [numpy.memmap(filename=path.join("memmap", tmp_file_name + str(order)), dtype=numpy.float16,
                #               mode="w+", shape=output.shape) for order, output in enumerate(outputs)]
                shared_array: list[shared_memory.SharedMemory] = \
                    [shared_memory.SharedMemory(name=tmp_file_name + "_" + str(order), create=True, size=output.nbytes)
                     for order, output in enumerate(outputs)]
                shared_ndarray = [numpy.ndarray(shape=output.shape, dtype=numpy.float16, buffer=shm.buf)
                                  for shm, output in zip(shared_array, outputs, strict=True)]
                for shm, output in zip(shared_ndarray, outputs, strict=True):
                    shm[:] = output[:]
                future = executor.submit(post_processor_shm, tmp_file_name, [output.shape for output in outputs],
                                         chunk_size, [image_size, image_size])
                # print(future.result())
                # future.add_done_callback(pprint)
                # exit(0)
