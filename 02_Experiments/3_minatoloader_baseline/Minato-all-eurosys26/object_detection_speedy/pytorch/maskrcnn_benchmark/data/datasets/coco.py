# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints
from maskrcnn_benchmark.utils.mlperf_logger import log_event
from mlperf_logging import mllog
from mlperf_logging.mllog import constants

import time

from pycocotools.coco import COCO

min_keypoints_per_image = 10

######################
#try:
#    with open("time_preprocess.txt", "w") as f:
#        f.write("Preprocessing Time.\n")
#except FileExistsError:
#    print("File not created.")

size = 0
ann_file = "/projects/I20240005/coco/annotations/instances_train2017.json"
coco=COCO(ann_file)
img_name = ""

def get_img_name(img_id):
    # print("get_image_name ", type(image_id))
    # coco=COCO(ann_file)
    img_info = coco.loadImgs(int(img_id))
    return img_info[0]['file_name']

#######################
def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False


class COCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None
    ):
        ann_file = "/projects/I20240005/coco/annotations/instances_train2017.json"
        root = "/projects/I20240005/coco/train2017"

        super(COCODataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        # filter images without detection annotations
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.ids = ids

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms

    def __getitem__(self, idx):
        img, anno = super(COCODataset, self).__getitem__(idx)

        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]
        
        img_id = self.ids[idx]
        global img_name
        img_name = get_img_name(img_id)
        global size
        size = img.size
        #print("speedyloader - getitem", img_id)

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        masks = [obj["segmentation"] for obj in anno]
        masks = SegmentationMask(masks, img.size)
        target.add_field("masks", masks)

        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = PersonKeypoints(keypoints, img.size)
            target.add_field("keypoints", keypoints)

        target = target.clip_to_image(remove_empty=True)

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data

#####################################################

# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random
from PIL import Image

import multiprocessing
import sys
import threading

import torch
import torchvision
from torchvision.transforms import functional as F

filelock = multiprocessing.Lock()

# ##pay attention this is for pecan
# class Compose(object):
#     def __init__(self, transforms):
#         self.transforms = transforms

#     def __call__(self, image, target):
#         resize_transform = None
#         for i, t in enumerate(self.transforms):
#             if isinstance(t, Resize):
#                 resize_transform = t
#                 break

#         # If resize is found, check if it's shrinking or expanding
#         if resize_transform is not None:
#             if isinstance(image, Image.Image):  
#                 w, h = image.size  # PIL gives (W, H), so swap them
#                 image_size = (h, w)
#             if isinstance(image, torch.Tensor):  
#                 h, w = image.shape[-2:]  # Extract (H, W)
#                 image_size = (h, w)
#             print("image_size", image_size)


#             resize_transform.get_size(image_size)  # Determine if shrinking or expanding

#             if resize_transform.shrinking:
#                 # Leave the transforms in original order if shrinking
#                 print("self.transforms original", self.transforms)

#                 for t in self.transforms:
#                     image, target = t(image, target)
#             else:
#                 # Move Resize to the end if expanding
#                 self.transforms.remove(resize_transform)  # Remove Resize from the list
#                 self.transforms.insert(-2, resize_transform)  
#                 # Apply the transformations in the new order
#                 print("self.transforms reordered", self.transforms)
#                 for t in self.transforms:
#                     image, target = t(image, target)

#         else:
#             # If no Resize, just apply the transforms in order
#             for t in self.transforms:
#                 image, target = t(image, target)

#         return image, target

#     def __repr__(self):
#         format_string = self.__class__.__name__ + "("
#         for t in self.transforms:
#             format_string += "\n"
#             format_string += "    {0}".format(t)
#         format_string += "\n)"
#         return format_string




# class Resize(object):
#     def __init__(self, min_size, max_size):
#         if not isinstance(min_size, (list, tuple)):
#             min_size = (min_size,)
#         self.min_size = min_size
#         self.max_size = max_size
#         self.shrinking = False

#     def get_size(self, image_size):
#         """Determines the new size and whether resizing shrinks the image."""
#         h, w = image_size  # Expecting (H, W) for both PIL images and tensors
#         size = random.choice(self.min_size)
#         max_size = self.max_size

#         if max_size is not None:
#             min_original_size = float(min(w, h))
#             max_original_size = float(max(w, h))
#             if max_original_size / min_original_size * size > max_size:
#                 size = int(round(max_size * min_original_size / max_original_size))
#                 self.shrinking = True
#             else:
#                 self.shrinking = False            

#         if (w <= h and w == size) or (h <= w and h == size):
#             return h, w

#         if w < h:
#             ow = size
#             oh = int(size * h / w)
#         else:
#             oh = size
#             ow = int(size * w / h)

#         return oh, ow

#     def __call__(self, image, target):
#         """Resizes both PIL and tensor images."""
#         start = time.time()

#         if isinstance(image, torch.Tensor):
#             # Tensor shape: (C, H, W), extract (H, W)
#             h, w = image.shape[-2:]  
#             image_size = (h, w)
#         elif isinstance(image, Image.Image):
#             # PIL image: use .size, which is (W, H), so reverse it to (H, W)
#             w, h = image.size  
#             image_size = (h, w)
#         else:
#             raise TypeError("Unsupported image type. Must be PIL Image or Tensor.")

#         size = self.get_size((h, w))

       
#                 # Convert tensor to PIL Image if needed
#         if isinstance(image, torch.Tensor):
#             image = F.to_pil_image(image)  # Convert tensor to PIL Image

#         image = F.resize(image, size)
#         target = target.resize(image.size)
        

#         duration = time.time() - start
#         print(f"speedyloader - Transform.Resize: size {size}, time(ms) {duration * 1000:.2f}")

#         return image, target


#FOR PYTORCH 
# class Compose(object):
#     def __init__(self, transforms):
#         self.transforms = transforms

#     def __call__(self, image, target):
#         for t in self.transforms:
#             image, target = t(image, target)
#         return image, target

#     def __repr__(self):
#         format_string = self.__class__.__name__ + "("
#         for t in self.transforms:
#             format_string += "\n"
#             format_string += "    {0}".format(t)
#         format_string += "\n)"
#         return format_string
import time

class Compose:
    def __init__(self, transforms, timeouts=None):
        self.transforms = transforms
        self.timeouts = timeouts or [None] * len(transforms)  # None = no timeout

    def __call__(self, image, target):
        # timing + tag
        if "preproc_start" not in target:
            target["preproc_start"] = time.perf_counter()
        target.setdefault("is_long", False)

        for i, (t, timeout) in enumerate(zip(self.transforms, self.timeouts)):
            if timeout is None:
                # normal path
                image, target = t(image, target)
                continue

            # 1) try with timeout (detect long)
            result, status = run_with_timeout(t, image, target, timeout)

            if status == "Success":
                image, target = result

            elif "timeout" in status:
                # 2) mark long and finish **synchronously** (no preemption, no queue)
                target["is_long"] = True
                image, target = t(image, target)

            else:  # error
                target["preproc_end"] = time.perf_counter()
                target["preproc_dur"] = float(target["preproc_end"] - target["preproc_start"])
                target["error"] = f"transform_{i}:{t}"
                return None, None  # or raise

        # done
        target["preproc_end"] = time.perf_counter()
        target["preproc_dur"] = float(target["preproc_end"] - target["preproc_start"])
        return image, target


class Resize(object):
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image, target):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        target = target.resize(image.size)
        return image, target
class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        #size = image.size
        start = time.time()
        if random.random() < self.prob:
            image = F.hflip(image)
            target = target.transpose(0)
        duration = time.time() - start
        print("speedyloader - Transform.RandomHorizontalFlip img", img_name, "size ", size, "time(s)", duration*1000)
#        log_event(key=constants.TRAIN_SAMPLES, value={"speedyloader - Transform.RandomHorizontalFlip img " : img_name, "size " : size, " time(ms) " : duration*1000})
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        #size = image.size
        start = time.time()
        tmp = F.to_tensor(image)
        duration = time.time() - start
        print("speedyloader - Transform.ToTensor img", img_name, "size ", size, "time(s)", duration*1000)
#        log_event(key=constants.TRAIN_SAMPLES, value={"speedyloader - Transform.ToTensor img " : img_name, "size " : size, " time(ms) " : duration*1000})
        return tmp, target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target):
        #size = image.size
        start = time.time()
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        duration = time.time() - start
        print("speedyloader - Transform.Normalize img ", img_name, "size ", size, " time(s)", duration*1000)
#        log_event(key=constants.TRAIN_SAMPLES, value={"speedyloader - Transform.Normalize img " : img_name, " size " : size, " time(ms) " : duration*1000})
        #filelock.acquire()
        #with open("time_preprocess.txt","a") as f:
        #    f.write("speedyloader - Transform.Normalize img " + repr(get_img_name(img_id)) + " time(ms) " + repr(duration*1000) + "\n")
        #    f.close()
        #filelock.release()
        return image, target

