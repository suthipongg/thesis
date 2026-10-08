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

#img_id = 0
#size = 0

def get_img_name(img_id):
    # print("get_image_name ", type(image_id))
    ann_file = "/datasets/coco/annotations/instances_train2017.json"
    coco=COCO(ann_file)
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
        
#        global img_id
#        img_id = self.ids[idx]
#        global size 
#        size = img.size
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

import multiprocessing
import sys
import threading

import torch
import torchvision
from torchvision.transforms import functional as F

filelock = multiprocessing.Lock()

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


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
        #size = image.size
        start = time.time()
        size = self.get_size(image.size)
        image = F.resize(image, size)
        target = target.resize(image.size)
        duration = time.time() - start
#        filelock.acquire()
#        print("speedyloader - Transform.Resize img", get_img_name(img_id), "time(ms)", duration*1000)
#        log_event(key=constants.TRAIN_SAMPLES, value={"speedyloader - Transform.Resize img " : repr(get_img_name(img_id)), "size " : size, " time(ms) " : repr(duration*1000)})
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
#        print("speedyloader - Transform.RandomHorizontalFlip img", get_img_name(img_id), "time(s)", (time.time() - start))
        duration = time.time() - start
#        log_event(key=constants.TRAIN_SAMPLES, value={"speedyloader - Transform.RandomHorizontalFlip img " : repr(get_img_name(img_id)), "size " : size, " time(ms) " : repr(duration*1000)})
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        #size = image.size
        start = time.time()
        tmp = F.to_tensor(image)
#        print("speedyloader - Transform.ToTensor img", get_img_name(img_id), "time(s)", (time.time() - start))
        duration = time.time() - start
#        log_event(key=constants.TRAIN_SAMPLES, value={"speedyloader - Transform.ToTensor img " : repr(get_img_name(img_id)), "size " : size, " time(ms) " : repr(duration*1000)})
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
#        print("speedyloader - Transform.Normalize img", get_img_name(img_id), "time(s)", (time.time() - start))
        duration = time.time() - start
#        log_event(key=constants.TRAIN_SAMPLES, value={"speedyloader - Transform.Normalize img " : repr(get_img_name(img_id)), " size " : size, " time(ms) " : repr(duration*1000)})
        #filelock.acquire()
        #with open("time_preprocess.txt","a") as f:
        #    f.write("speedyloader - Transform.Normalize img " + repr(get_img_name(img_id)) + " time(ms) " + repr(duration*1000) + "\n")
        #    f.close()
        #filelock.release()
        return image, target

