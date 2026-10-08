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
from maskrcnn_benchmark.config import cfg

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
root = "/projects/I20240005/coco/train2017"
coco=COCO(ann_file)
img_name = ""



def get_img_name(img_id):
    # print("get_image_name ", type(image_id))
    # ann_file = "/datasets/coco/annotations/instances_train2017.json"
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
        print("speedyloader - getitem", img_id)

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
            batch = self._transforms(img, target)
            if batch is None:
                return 
            else: 
                img, target = batch

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




import sys
import multiprocessing
import queue
import signal
import time

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Transform execution exceeded the timeout!")

def run_transform(transform, image, target, result_queue):
    """
    Executes the transform and puts the result into the queue.
    """
    try:
        # print("Image input before transform:", image)
        # print(f"Transform {transform} started.")
        result = transform(image, target)
        if result == (None, "Timeout"):
            return result_queue.put((None, "timeout"))
        
        result_queue.put((result, None))  # Enqueue result and no error
        # print(f"Transform {transform} completed and result enqueued {result}.")
    except Exception as e:
        print(f"Transform {transform} encountered an error: {e}")
        result_queue.put((None, e))  # Enqueue no result and the error

def run_with_timeout(transform, image, target, timeout):
    """
    Runs the transform with a timeout using signals.
    Ensures success status always has a valid result.
    """
    result_queue = queue.Queue(maxsize=1)  # Standard library queue for simplicity

    def wrapper():
        run_transform(transform, image, target, result_queue)

    # Set up signal handler for timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout)  # Start timer with sub-second precision

    try:
        wrapper()  # Run the transform function
    except TimeoutException as e:
        print(f"Timeout occurred: {e}")
        return None, "timeout"  
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None, "Error" 
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)  # Cancel the timer in all cases

    # Process result from the queue
    try:
        result, error = result_queue.get_nowait()  # Get the result from the queue
        if error and result is None:
            print(f"Transform {transform} failed with error: {error}")
            return None, f"Error: {error}"  
        
        # print(f"Transform {transform} succeeded with result: {result}")
        return result, "Success"
    except queue.Empty:
        # print(f"Transform {transform} completed but did not enqueue any result.")
        return None, "No Result"  



import torch.multiprocessing as mp
import time


class Compose(object):
    def __init__(self, transforms, queue_timeout=None, timeouts=[0.5, 0.5, 0.5, 0.5]):
        """
        Initialize Compose with a list of transforms and an optional timeout.
        """
        self.transforms = transforms
        self.queue_timeout = queue_timeout 
        self.timeouts = timeouts  # Timeout in seconds
        self.task_queue = multiprocessing.Queue()
        self.timeout_process = None

    def start_timeout_process(self):
        if self.timeout_process is None or not self.timeout_process.is_alive():
            self.timeout_process = mp.Process(
                target=self.process_remaining_transforms)
            self.timeout_process.start()



    def __call__(self, image, target):
       
        for i, (t, timeout) in enumerate(zip(self.transforms, self.timeouts)):
            result, status = run_with_timeout(t, image, target, timeout)
            print("Result in Compose:", result, "Status:", status)
            if status == "Success":
                image, target = result

            elif "timeout" in status:
                print(f"Transform {i} {t} for image exceeded timeout. Spawning parallel process for remaining transforms.")
                self.start_timeout_process()  # Ensure the process is running when needed
                self.task_queue.put((image, target, i))
                return None, None
                

            elif "Error" in status:
                print(f"Transform {t} failed with an error.")
                return None, None
        
        else:
            image , target = t(image, target)

        print("the return statement ", type(image))    
        return image, target


        

    def process_remaining_transforms(self):
        """
        Apply remaining transforms and put results into the queue.
        Signal completion using the done_event.
        """
        while True:
            
            item = self.task_queue.get()
            if item is None:
                print("[timeout_worker] Received sentinel => shutting down.")
                break

            # item should be (image, target)
            print("task queue dequeued")
            image, target, i = item
            try:
                print("self.transforms", self.transforms)
                for t in self.transforms[i:]:
                    print(f"Applying remaining transform: {t}, image: {type(image)}, target: {type(target)}")
                    image, target = t(image, target)
                    print(f"After transform {t}: image: {type(image)}, target: {type(target)}")

                # image = image.numpy()
                # target = {key: value.numpy() for key, value in target.items()}
                
                
                batch = image, target, i
                while self.queue_timeout.qsize() >= 190 :  # Maximum queue size
                    print(f"Timeout Queue full (size: {self.queue_timeout.qsize()}). Pausing...")
                    time.sleep(0.05)  # Wait for 0.5 seconds before checking again
                self.queue_timeout.put(batch)
                print("Adding image and target to the timeout queue.", self.queue_timeout.qsize())

            except Exception as e:
                print(f"Error in process_remaining_transforms: {e}")
        def __repr__(self):
            format_string = self.__class__.__name__ + "("
            for t in self.transforms:
                format_string += "\n"
                format_string += "    {0}".format(t)
            format_string += "\n)"
            return format_string

        


# class Compose(object):
#     def __init__(self, transforms, queue_timeout):
#         self.transforms = transforms

#     def __call__(self, image, target):
#         for t in self.transforms:
#             print("transformations in compose", t)
#             image, target = t(image, target)
#         return image, target

#     def __repr__(self):
#         format_string = self.__class__.__name__ + "("
#         for t in self.transforms:
#             format_string += "\n"
#             format_string += "    {0}".format(t)
#         format_string += "\n)"
#         return format_string
import random
import time
import torch
import torchvision.transforms.functional as F

class Resize(object):
    def __init__(self, min_size, max_size, seed=42):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        self.seed = seed

    def get_size(self, image, image_size):
        # Ensure deterministic behavior
        if self.seed is not None:
            random.seed(self.seed)  # Set the seed for reproducibility

        w, h = image_size
        size = random.choice(self.min_size)
        factor = 0.5
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        # for i in range(1):
        #     x = torch.randn(image.size[1], image.size[0]) * factor
        #     image_tensor = F.to_tensor(image)
        #     image_cl = torch.clamp(image_tensor + x, 0, 1)
        #     image = F.to_pil_image(image_cl)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image, target):
        start = time.time()
        size = self.get_size(image, image.size)
        image = F.resize(image, size)
        target = target.resize(image.size)
        duration = time.time() - start
        print("speedyloader - Transform.Resize time(ms)", duration * 1000)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5, seed=42):
        self.prob = prob
        self.seed = seed

    def __call__(self, image, target):
        start = time.time()

        # Ensure deterministic behavior
        if self.seed is not None:
            random.seed(self.seed)  # Set the seed for reproducibility

        if random.random() < self.prob:
            image = F.hflip(image)
            target = target.transpose(0)
        
        duration = time.time() - start
        print("speedyloader - Transform.RandomHorizontalFlip time(ms)", duration * 1000)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        start = time.time()
        tmp = F.to_tensor(image)
        duration = time.time() - start
        print("speedyloader - Transform.ToTensor time(ms)", duration * 1000)
        return tmp, target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True, seed=42):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255
        self.seed = seed

    def __call__(self, image, target):
        start = time.time()

        # Ensure deterministic behavior
        if self.seed is not None:
            random.seed(self.seed)  # Set the seed for reproducibility

        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)

        duration = time.time() - start
        print("speedyloader - Transform.Normalize time(ms)", duration * 1000)
        return image, target
