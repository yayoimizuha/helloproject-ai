import json
import os
import warnings

warnings.filterwarnings("ignore", lineno=6, category=UserWarning)
from concurrent.futures.process import ProcessPoolExecutor
from itertools import chain
from multiprocessing import shared_memory
from io import BytesIO
from os import listdir, path, pathsep, makedirs
from pprint import pprint
import more_itertools
import msgspec
import pandas.io.json
import tqdm
from PIL import Image
from uuid import uuid4
from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel
from torch import tensor
import aiofiles
import numpy
import torch
from torchvision.io import decode_jpeg
from asyncio import run, gather, Semaphore
from site import getsitepackages
from rust_retinaface_post_processor import resnet_post_process

USE_OPENVINO = True
if USE_OPENVINO:
    import openvino

    ov_core = openvino.Core()
os.environ["Path"] = path.join(getsitepackages()[-1], "tensorrt_libs") + pathsep + os.environ["Path"]
root_dir = r"E:\helloproject-ai-data\blog_images"
model_path = r"C:\Users\tomokazu\build\retinaface\retinaface_only_nn_fp16.onnx"
# makedirs("memmap", exist_ok=True)

files = []
files_data: dict[str, numpy.ndarray | None] = {}
chunk_size = 16
image_size = 640
device = torch.device("cpu") if torch.xpu.is_available() else exit(-1)


async def async_read(path: str, semaphore: Semaphore):
    async with semaphore:
        async with aiofiles.open(file=path, mode="rb") as fp:
            return await fp.read()


async def gather_runner(l: list, fn):
    sem = Semaphore(2048)
    return await gather(*[fn(p, sem) for p in l])


# def post_processor(outputs, batch_size, image_size):
#     # print("aaa", flush=True)
#     outputs = [numpy.ascontiguousarray(output.astype(numpy.float32)) for output in outputs]
#     res = resnet_post_process([output.__array_interface__["data"][0] for output in outputs], batch_size, image_size)
#     return res
#
#
# def post_processor_memmap(tmp_filename, sizes, batch_size, image_size): # print("aaa", flush=True) outputs = [
# numpy.memmap(filename=path.join("memmap", tmp_filename + str(order)), dtype=numpy.float16, mode="r", shape=size)
# for order, size in enumerate(sizes)] outputs = [numpy.ascontiguousarray(output.astype(numpy.float32)) for output in
# outputs] res = resnet_post_process([output.__array_interface__["data"][0] for output in outputs], batch_size,
# image_size) return res


def post_processor_shm(shm_name, sizes, batch_size, image_size):
    shms = [shared_memory.SharedMemory(name=shm_name + "_" + str(i)) for i in range(3)]
    outputs = \
        [numpy.ascontiguousarray(numpy.ndarray(shape=size, dtype=numpy.float16, buffer=shm.buf).astype(numpy.float32))
         for size, shm in zip(sizes, shms)]
    res = resnet_post_process([output.__array_interface__["data"][0] for output in outputs], batch_size, image_size)
    # print(res)
    return res


def dec_jpg(f, fn):
    _decoded_image = tensor(numpy.array(Image.open(BytesIO(f.tobytes()))).transpose([2, 0, 1]))
    _decoded_image = _decoded_image.to(device, torch.float16) / 255
    _decoded_image = fn[2](_decoded_image)
    _decoded_image_resized = fn[0](_decoded_image)
    return fn[1](_decoded_image_resized)


if __name__ == '__main__':
    from kornia.augmentation import LongestMaxSize, PadTo, Normalize
    from kornia.constants import Resample

    longest_max_size = LongestMaxSize(max_size=640, resample=Resample.NEAREST)
    pad_to = PadTo(size=(640, 640), pad_value=1.)
    normalize = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    if USE_OPENVINO:
        onnx_model = ov_core.read_model(model_path)
        onnx_model.reshape([chunk_size, 3, image_size, image_size])
        onnx_model = ov_core.compile_model(onnx_model, device_name='GPU')

    else:
        session_options = SessionOptions()
        session_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
        # session_options.optimized_model_filepath = GraphOptimizationLevel = "onnx_cache"
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
                ('OpenVINOExecutionProvider', {
                    'device_type': 'GPU.0',
                    'precision': 'FP16',
                    'cache_dir': 'openvino_cache'
                }),
                'CUDAExecutionProvider',
                'CPUExecutionProvider'
            ],
            sess_options=session_options
        )
    if os.path.exists("faces.jsonl"):
        with open(file="faces.jsonl", mode="r", encoding="utf-8") as fp:
            already = {list(msgspec.json.decode(line).keys())[0] for line in fp.read().removesuffix("\n").split("\n")}
    else:
        already = set()
    pbar = tqdm.tqdm(
        total=(set().union(*[listdir(path.join(root_dir, name)) for name in listdir(root_dir)]) - already).__len__())

    # print(len(already))
    # exit(0)

    for name in listdir(root_dir):
        with (ProcessPoolExecutor(max_workers=16) as executor):
            pbar.set_description_str(desc=name, refresh=True)
            if name != "ブログ":
                # continue
                pass
            file_names = listdir(path.join(root_dir, name))
            file_names_set = set(file_names) - already
            file_names = list(file_names_set)
            name_files = [path.join(root_dir, name, file_name) for file_name in file_names]
            files_data = {file_name: numpy.frombuffer(dat, dtype=numpy.uint8) for file_name, dat in
                          zip(file_names, run(gather_runner(name_files, async_read)))}
            if files_data.__len__() == 0:
                continue
            futures = []
            shms = []
            namess = []
            # print(k_1)
            for cnk in more_itertools.chunked(files_data.items(), n=chunk_size):
                stack = []
                names = []
                if USE_OPENVINO:
                    fn_pack = [longest_max_size, pad_to, normalize]
                    submits = []
                    for file, dat in cnk:
                        submits.append(executor.submit(dec_jpg, dat, fn_pack))
                        names.append(file)
                    for submit in submits:
                        stack.append(submit.result().squeeze())
                else:
                    for file, dat in cnk:
                        try:
                            decoded_image = decode_jpeg(tensor(dat), device=device)
                        except:
                            decoded_image = tensor(
                                numpy.array(Image.open(BytesIO(dat.tobytes()))).transpose([2, 0, 1]))
                        decoded_image = decoded_image.to(device, torch.float16) / 255
                        decoded_image = normalize(decoded_image)
                        decoded_image_resized = longest_max_size(decoded_image)
                        decoded_image_padded = pad_to(decoded_image_resized)
                        stack.append(decoded_image_padded.squeeze())
                        names.append(file)
                namess.append(names)
                [stack.append(torch.zeros(size=[3, 640, 640], dtype=torch.float16, device=device)) for _ in
                 range(chunk_size - stack.__len__())]
                stacked = torch.stack(stack).contiguous()
                # print(stacked.shape)
                if USE_OPENVINO:
                    _outputs = onnx_model([stacked])
                    # print(_outputs[onnx_model.output(0)])
                    outputs = [_outputs[onnx_model.output(i)] for i in range(2, -1, -1)]
                    # print(outputs)
                else:
                    io_binding = session.io_binding()
                    io_binding.bind_input(
                        name="input",
                        device_type=stacked.device.type,
                        device_id=stacked.device.index if stacked.device.index is not None else 0,
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
                uuid = uuid4().__str__()
                shared_array: list[shared_memory.SharedMemory] = \
                    [shared_memory.SharedMemory(name=uuid + "_" + str(order), create=True, size=output.nbytes)
                     for order, output in enumerate(outputs)]
                shared_ndarray = [numpy.ndarray(shape=output.shape, dtype=numpy.float16, buffer=shm.buf)
                                  for shm, output in zip(shared_array, outputs, strict=True)]
                for shm, output in zip(shared_ndarray, outputs, strict=True):
                    shm[:] = output[:]
                future = executor.submit(post_processor_shm, uuid, [output.shape for output in outputs],
                                         chunk_size, [image_size, image_size])
                futures.append(future)
                shms.extend(shared_array)
                # exit(0)
                pbar.update(n=cnk.__len__())
            # result_dict = dict()
            with open("faces.jsonl", mode="a", encoding="utf-8") as fp:
                futures_results = [future.result() for future in futures]
                # pprint(futures_results)
                for names, futures_result in zip(namess, futures_results):

                    for name, results in zip(names, futures_result):
                        results_list = []
                        if results:
                            # print(name)
                            for result in results:
                                # [print(int(a), end=" ") for a in result[0]]
                                # print(*result[1], end=" ")
                                # [print(int(a), end=" ") for a in result[2]]
                                # print()
                                # results_list.append(list(chain.from_iterable([result])))
                                fp.write(
                                    pandas.io.json.ujson_dumps({name: [result[0], result[1][0], result[2]]},
                                                               ensure_ascii=False, double_precision=5) + "\n")
                                pass
                        else:
                            fp.write(
                                pandas.io.json.ujson_dumps({name: None}, ensure_ascii=False) + "\n")

                            # print(name, [])
                            pass
                            # result_dict[name] = results_list
                            # pprint(result_dict)
            [shm.close() for shm in shms]
