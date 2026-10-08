# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import ctypes
import time
import logging

import numpy as np
import torch

# DALI imports
import nvidia.dali as dali
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.types import to_numpy_type


class COCOPipeline(Pipeline):
    def __init__(self, batch_size, file_root, annotations_file, default_boxes,
                 device_id, num_shards,
                 output_fp16=False, output_nhwc=False, pad_output=False,
                 num_threads=1, seed=15):
        super(COCOPipeline, self).__init__(batch_size=batch_size,
                                           device_id=device_id,
                                           num_threads=num_threads,
                                           seed=seed)

        if torch.distributed.is_initialized():
            shard_id = torch.distributed.get_rank()
        else:
            shard_id = 0

        # Data loader and image decoder
        self.input = dali.ops.readers.COCO(file_root=file_root,
                                           annotations_file=annotations_file,
                                           shard_id=shard_id,
                                           num_shards=num_shards,
                                           ratio=True,
                                           ltrb=True,
                                           shuffle_after_epoch=True,
                                           skip_empty=True)
        self.decode_slice = dali.ops.decoders.ImageSlice(device="cpu",
                                                         output_type=dali.types.RGB)

        # Augumentation techniques
        ## Random crop
        self.crop = dali.ops.RandomBBoxCrop(device="cpu",
                                            aspect_ratio=[0.5, 2.0],
                                            thresholds=[0, 0.1, 0.3, 0.5, 0.7, 0.9],
                                            scaling=[0.3, 1.0],
                                            bbox_layout="xyXY",
                                            allow_no_crop=True,
                                            num_attempts=1)
        ## Color twist
        self.hsv = dali.ops.Hsv(device="gpu",
                                dtype=dali.types.FLOAT)  # use float to avoid clipping and quantizing the intermediate result
        self.bc = dali.ops.BrightnessContrast(device="gpu",
                                              contrast_center=128,  # input is in the [0, 255] range
                                              dtype=dali.types.UINT8)
        ## Cropping and normalization
        dtype = dali.types.FLOAT16 if output_fp16 else dali.types.FLOAT
        output_layout = dali.types.NHWC if output_nhwc else dali.types.NCHW
        self.normalize = dali.ops.CropMirrorNormalize(
            device="gpu",
            crop=(300, 300),
            mean=[0.0, 0.0, 0.0],
            std=[255.0, 255.0, 255.0],
            mirror=0,
            dtype=dtype,
            output_layout=output_layout,
            pad_output=pad_output)
        self.gaussian_blur = dali.ops.GaussianBlur(device="gpu", window_size=51, sigma=10.0)
        self.jpeg_compress = dali.ops.JpegCompressionDistortion(device="gpu", quality=10)
        self.jpeg_decode = dali.ops.ImageDecoder(device="gpu", output_type=dali.types.RGB)
        self.warp = dali.ops.WarpAffine(device="gpu", matrix=[0.5, 0.5, 0, 0, 0.5, 0], fill_value=255)
        self.rotate = dali.ops.Rotate(device="gpu", angle=90)




        ## Flipping
        self.flip = dali.ops.Flip(device="cpu")
        self.bbflip = dali.ops.BbFlip(device="cpu", ltrb=True)

        # Resize
        self.resize = dali.ops.Resize(device="cpu",
                                      resize_x=300,
                                      resize_y=300)
               

        # Random variables
        self.rng1 = dali.ops.random.Uniform(range=[0.5, 1.5])
        self.rng2 = dali.ops.random.Uniform(range=[0.875, 1.125])
        self.rng3 = dali.ops.random.Uniform(range=[-0.5, 0.5])
        self.flip_coin = dali.ops.random.CoinFlip(probability=0.5)

        # bbox encoder
        self.anchors = default_boxes(order='ltrb').cpu().numpy().flatten().tolist()
        self.box_encoder = dali.ops.BoxEncoder(device="cpu",
                                               criteria=0.5,
                                               anchors=self.anchors)

    def define_graph(self):
        saturation = self.rng1()
        contrast = self.rng1()
        brightness = self.rng2()
        hue = self.rng3()
        coin_rnd = self.flip_coin()

        inputs, bboxes, labels = self.input(name="Reader")
        crop_begin, crop_size, bboxes, labels = self.crop(bboxes, labels)
        images = self.decode_slice(inputs, crop_begin, crop_size)
        images = self.flip(images, horizontal=coin_rnd)
        bboxes = self.bbflip(bboxes, horizontal=coin_rnd)
        images = self.resize(images)
   
        images = images.gpu()
        images = self.hsv(images, hue=hue, saturation=saturation)
        images = self.bc(images, brightness=brightness, contrast=contrast)
        images = self.normalize(images)
        bboxes, labels = self.box_encoder(bboxes, labels)

    
        # import torch
        # import torchvision.transforms.functional as F
        # from PIL import Image
        # import numpy as np

        # # Step 1: Generate a synthetic image (e.g., a random image)
        # # Let's create a random image of size 128x128 with 3 color channels (RGB)
        # image_array = np.random.rand(128, 128, 3)  # Random values in the range [0, 1]

        # # Step 2: Convert the NumPy array to a PIL image
        # image = Image.fromarray((image_array * 255).astype(np.uint8))  # Convert to [0, 255] and to uint8

        # # Step 3: Loop to apply noise to the image 50 times
        # for i in range(50):
        #     # Step 4: Generate random noise with the same size as the image (3 channels)
        #     x = torch.randn(3, image.size[1], image.size[0]) * 0.1  # Random noise for each color channel
            
        #     # Step 5: Convert the image to a tensor
        #     image_tensor = F.to_tensor(image)  # Convert PIL image to Tensor (range [0, 1])
            
        #     # Step 6: Add the noise to the image tensor
        #     image_cl = torch.clamp(image_tensor + x, 0, 1)  # Add noise and clamp to valid pixel range [0, 1]
            
        #     # Step 7: Convert the image tensor back to a PIL image
        #     image = F.to_pil_image(image_cl)  # Convert back to PIL image
        

    

        # bboxes and images and labels on GPU
        return (images, bboxes.gpu(), labels.gpu())

to_torch_type = {
    np.float32 : torch.float32,
    np.float64 : torch.float64,
    np.float16 : torch.float16,
    np.uint8   : torch.uint8,
    np.int8    : torch.int8,
    np.int16   : torch.int16,
    np.int32   : torch.int32,
    np.int64   : torch.int64
}

def feed_ndarray(dali_tensor, arr):
    """
    Copy contents of DALI tensor to pyTorch's Tensor.

    Parameters
    ----------
    `dali_tensor` : nvidia.dali.backend.TensorCPU or nvidia.dali.backend.TensorGPU
                    Tensor from which to copy
    `arr` : torch.Tensor
            Destination of the copy
    """
    assert dali_tensor.shape() == list(arr.size()), \
            ("Shapes do not match: DALI tensor has size {0}"
            ", but PyTorch Tensor has size {1}".format(dali_tensor.shape(), list(arr.size())))
    #turn raw int to a c void pointer
    c_type_pointer = ctypes.c_void_p(arr.data_ptr())
    dali_tensor.copy_to_external(c_type_pointer)
    return arr

class DALICOCOIterator(object):
    """
    COCO DALI iterator for pyTorch.

    Parameters
    ----------
    pipelines : list of nvidia.dali.pipeline.Pipeline
                List of pipelines to use
    size : int
           Epoch size.
    """
    def __init__(self, pipelines, size):
        if not isinstance(pipelines, list):
            pipelines = [pipelines]

        self._num_gpus = len(pipelines)
        assert pipelines is not None, "Number of provided pipelines has to be at least 1"
        self.batch_size = pipelines[0].max_batch_size
        self._size = size
        print("Epoch size: ", self._size)
        self._pipes = pipelines

        # Build all pipelines
        for p in self._pipes:
            p.build()

        # Use double-buffering of data batches
        self._data_batches = [[None, None, None, None] for i in range(self._num_gpus)]
        self._counter = 0
        self._current_data_batch = 0
        self.output_map = ["image", "bboxes", "labels"]

        # We need data about the batches (like shape information),
        # so we need to run a single batch as part of setup to get that info
        self._first_batch = None
        self._first_batch = self.next()

    def __next__(self):
        torch.cuda.synchronize()
        start_time = time.time()

        if self._first_batch is not None:
            batch = self._first_batch
            self._first_batch = None
            return batch
        if self._counter > self._size:
            raise StopIteration

        # Gather outputs
        outputs = []
        for p in self._pipes:
            p._prefetch()
        for p in self._pipes:
            outputs.append(p.share_outputs())
        for i in range(self._num_gpus):
            dev_id = self._pipes[i].device_id
            out_images = []
            bboxes = []
            labels = []
            # segregate outputs into image/labels/bboxes entries
            for j, out in enumerate(outputs[i]):
                if self.output_map[j] == "image":
                    out_images.append(out)
                elif self.output_map[j] == "bboxes":
                    bboxes.append(out)
                elif self.output_map[j] == "labels":
                    labels.append(out)

            # Change DALI TensorLists into Tensors
            images = [x.as_tensor() for x in out_images]
            images_shape = [x.shape() for x in images]

            # Prepare bboxes shapes
            bboxes_shape = []
            for j in range(len(bboxes)):
                bboxes_shape.append([])
                for k in range(len(bboxes[j])):
                    bboxes_shape[j].append(bboxes[j][k].shape())

            # Prepare labels shapes and offsets
            labels_shape = []
            bbox_offsets = []

            torch.cuda.synchronize()
            for j in range(len(labels)):
                labels_shape.append([])
                bbox_offsets.append([0])
                for k in range(len(labels[j])):
                    lshape = labels[j][k].shape()
                    bbox_offsets[j].append(bbox_offsets[j][k] + lshape[0])
                    labels_shape[j].append(lshape)

            # We always need to alocate new memory as bboxes and labels varies in shape
            images_torch_type = to_torch_type[to_numpy_type(images[0].dtype)]
            bboxes_torch_type = to_torch_type[to_numpy_type(bboxes[0][0].dtype)]
            labels_torch_type = to_torch_type[to_numpy_type(labels[0][0].dtype)]

            torch_gpu_device = torch.device('cuda', dev_id)
            torch_cpu_device = torch.device('cpu')

            pyt_images = [torch.zeros(shape, dtype=images_torch_type, device=torch_gpu_device) for shape in images_shape]
            pyt_bboxes = [[torch.zeros(shape, dtype=bboxes_torch_type, device=torch_gpu_device) for shape in shape_list] for shape_list in bboxes_shape]
            pyt_labels = [[torch.zeros(shape, dtype=labels_torch_type, device=torch_gpu_device) for shape in shape_list] for shape_list in labels_shape]
            pyt_offsets = [torch.zeros(len(offset), dtype=torch.int32, device=torch_cpu_device) for offset in bbox_offsets]

            self._data_batches[i][self._current_data_batch] = (pyt_images, pyt_bboxes, pyt_labels, pyt_offsets)

            # Copy data from DALI Tensors to torch tensors
            for j, i_arr in enumerate(images):
                feed_ndarray(i_arr, pyt_images[j])

            for j, b_list in enumerate(bboxes):
                for k in range(len(b_list)):
                    if (pyt_bboxes[j][k].shape[0] != 0):
                        feed_ndarray(b_list[k], pyt_bboxes[j][k])
                pyt_bboxes[j] = torch.cat(pyt_bboxes[j])

            for j, l_list in enumerate(labels):
                for k in range(len(l_list)):
                    if (pyt_labels[j][k].shape[0] != 0):
                        feed_ndarray(l_list[k], pyt_labels[j][k])
                pyt_labels[j] = torch.cat(pyt_labels[j])

            for j in range(len(pyt_offsets)):
                pyt_offsets[j] = torch.IntTensor(bbox_offsets[j])

        for p in self._pipes:
            p.release_outputs()
            p.schedule_run()

        copy_db_index = self._current_data_batch
        # Change index for double buffering
        self._current_data_batch = (self._current_data_batch + 1) % 2
        self._counter += self._num_gpus * self.batch_size
        torch.cuda.synchronize()
        end_time = time.time()
        print(f"[DALI] Preprocessing time: {(end_time - start_time)*1000:.2f} ms")
        with open("dali_time.txt", "a") as f:
            f.write(f"{(end_time - start_time) * 1000:.2f}\n")

        return [db[copy_db_index] for db in self._data_batches]

    def next(self):
        """
        Returns the next batch of data.
        """
        return self.__next__();

    def __iter__(self):
        return self

    def reset(self):
        """
        Resets the iterator after the full epoch.
        DALI iterators do not support resetting before the end of the epoch
        and will ignore such request.
        """
        if self._counter > self._size:
            self._counter = self._counter % self._size
        else:
            logging.warning("DALI iterator does not support resetting while epoch is not finished. Ignoring...")