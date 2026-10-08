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
from maskrcnn_benchmark.structures.image_list import to_image_list


class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        print("type of batch inside collate", type(batch))
        transposed_batch = list(zip(*batch))
        print("type of transdposed batch", type(transposed_batch[0]))
        print("size divisible", self.size_divisible)
        images = to_image_list(transposed_batch[0], self.size_divisible)
        for img in transposed_batch[0]:
            print("img shape  the batch before collate", img.shape)
        
        targets = transposed_batch[1]
        img_ids = transposed_batch[2]
        return images, targets, img_ids
