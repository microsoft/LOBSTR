"""
Copyright (C) 2023 Microsoft Corporation
"""
import random
from collections import defaultdict
import os
import json

import PIL
from PIL import Image, ImageFilter
import torch
from fitz import Rect
import numpy as np
from torchvision import transforms

# Project imports
from detr.datasets import transforms as R

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class RandomTightCrop(object):
    def __init__(self, prob, min_left_padding, max_left_padding,
                 min_top_padding, max_top_padding,
                 min_right_padding, max_right_padding,
                 min_bottom_padding, max_bottom_padding):
        self.prob = prob
        self.min_left_padding = min_left_padding
        self.max_left_padding = max_left_padding
        self.min_top_padding = min_top_padding
        self.max_top_padding = max_top_padding
        self.min_right_padding = min_right_padding
        self.max_right_padding = max_right_padding
        self.min_bottom_padding = min_bottom_padding
        self.max_bottom_padding = max_bottom_padding

    def __call__(self, image, target):
        if random.random() < self.prob:
            width, height = image.size
            
            min_left = int(round(min([bbox[0].item() for bbox in target['boxes']])))
            min_top = int(round(min([bbox[1].item() for bbox in target['boxes']])))
            max_right = int(round(max([bbox[2].item() for bbox in target['boxes']])))
            max_bottom = int(round(max([bbox[3].item() for bbox in target['boxes']])))
            
            left = random.randint(max(0, min_left-self.max_left_padding), max(0, min_left-self.min_left_padding))
            top = random.randint(max(0, min_top-self.max_top_padding), max(0, min_top-self.min_top_padding))
            right = random.randint(min(width, max_right+self.min_right_padding), min(width, max_right+self.max_right_padding))
            bottom = random.randint(min(height, max_bottom+self.min_bottom_padding), min(height, max_bottom+self.max_bottom_padding))         
            
            cropped_image = image.crop((left, top, right, bottom))
            cropped_bboxes = []
            cropped_labels = []
            cropped_idxs = []
            for idx, bbox, label in zip(range(len(target['boxes'])), target["boxes"], target["labels"]):
                bbox = [max(bbox[0], left) - left,
                        max(bbox[1], top) - top,
                        min(bbox[2], right) - left,
                        min(bbox[3], bottom) - top]
                if bbox[0] <= bbox[2] and bbox[1] <= bbox[3]:
                    cropped_idxs.append(idx)
                    cropped_bboxes.append(bbox)
                    cropped_labels.append(label)
                         
            if len(cropped_bboxes) > 0:
                adjacencies = target["adjacencies"].numpy()
                adjacencies = adjacencies[np.ix_(cropped_idxs, cropped_idxs)]
                
                target["crop_orig_size"] = torch.tensor([bottom-top, right-left])
                target["crop_orig_offset"] = torch.tensor([left, top]) 
                target["boxes"] = torch.as_tensor(cropped_bboxes, dtype=torch.float32)
                target["labels"] = torch.as_tensor(cropped_labels, dtype=torch.int64)
                target["adjacencies"] = torch.as_tensor(adjacencies, dtype=torch.int64)
                return cropped_image, target

        print("not cropping")
        target["crop_orig_size"] = torch.tensor([height, width])
        target["crop_orig_offset"] = torch.tensor([0, 0]) 
        return image, target

    
class RandomDilation(object):
    def __init__(self, probability=0.5, size=3):
        self.probability = probability
        self.filter = ImageFilter.RankFilter(size, int(round(0 * size * size))) # 0 is equivalent to a min filter

    def __call__(self, img: PIL.Image.Image, target: dict):
        r = random.random()
        
        if r <= self.probability:
            img = img.filter(self.filter)
        
        return img, target
        
class RandomErosion(object):
    def __init__(self, probability=0.5, size=3):
        self.probability = probability
        self.filter = ImageFilter.RankFilter(size, int(round(0.6 * size * size))) # Almost a median filter

    def __call__(self, img: PIL.Image.Image, target: dict):
        r = random.random()
        
        if r <= self.probability:
            img = img.filter(self.filter)
        
        return img, target
    
class RandomMaxResize(object):
    def __init__(self, min_max_size, max_max_size):
        self.min_max_size = min_max_size
        self.max_max_size = max_max_size

    def __call__(self, image, target):
        width, height = image.size
        current_max_size = max(width, height)
        target_max_size = random.randint(self.min_max_size, self.max_max_size)
        scale = target_max_size / current_max_size
        resized_image = image.resize((int(round(scale*width)), int(round(scale*height))))
        resized_bboxes = []
        for bbox in target["boxes"]:
            bbox = [scale*elem for elem in bbox]
            resized_bboxes.append(bbox)

        target["boxes"] = torch.as_tensor(resized_bboxes, dtype=torch.float32)
        
        return resized_image, target
    
    
class RandomGaussianBlur(object):
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, img: PIL.Image.Image, target: dict):
        if random.random() <= self.probability:
            rad = random.randint(1,3)
            filt = PIL.ImageFilter.GaussianBlur(radius=rad)
            img = img.filter(filt)
        
        return img, target


def get_transform(split):
    """
    Returns the appropriate transforms.
    """

    normalize = R.Compose([
        R.ToTensor(),
        R.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if split == 'train':
        return R.Compose([
            RandomTightCrop(1, -5, 200, -5, 200, -5, 200, -5, 200),
            RandomMaxResize(600, 1100),
            RandomMaxResize(900, 1100),
            RandomGaussianBlur(0.2),
            RandomDilation(0.05, 3),
            RandomErosion(0.05, 3),
            normalize
        ])

    if split == 'val' or split == 'test':
        return R.Compose([
            RandomTightCrop(1, 100, 100, 100, 100, 100, 100, 100, 100),
            RandomMaxResize(800, 800),
            normalize
        ])
    
    if split == 'val_no_norm' or 'test_no_norm':
        return R.Compose([
            RandomTightCrop(1, 100, 100, 100, 100, 100, 100, 100, 100),
            RandomMaxResize(800, 800)
        ])

    raise ValueError(f'unknown {split}')


# TODO: Add code for converting to COCO
class ScatterPlotDataset(torch.utils.data.Dataset):
    def __init__(self, data_directory, transforms=None, max_size=None, do_crop=True,
                 include_original=False, exclude_classes=set()):
        self.data_directory = data_directory
        self.transforms = transforms
        self.do_crop=do_crop
        self.include_original = include_original
        self.exclude_classes = exclude_classes
        
        self.file_ids = [elem.replace(".jpg", "") for elem in os.listdir(data_directory) if elem.endswith(".jpg")]

    def __getitem__(self, idx):
        file_id = self.file_ids[idx]
        
        img = Image.open(os.path.join(self.data_directory, file_id + ".jpg"))
        
        with open(os.path.join(self.data_directory, file_id + ".json"), 'r') as json_file:
            data = json.load(json_file)
        bboxes = data['bboxes']
        classes = data['classes']
        clusters = data['clusters']
        
        overall_bbox = []
        for class_num, bbox in zip(classes, bboxes):
            if class_num == 0:
                overall_bbox = bbox
        
        keep_bboxes = []
        keep_indices = []
        for node_idx, bbox in enumerate(bboxes):
            if classes[node_idx] in self.exclude_classes:
                continue
            
            if bbox[0] < 0 or bbox[2] > img.width or bbox[1] < 0 or bbox[3] > img.height:
                continue

            keep_bboxes.append(bbox)
            keep_indices.append(node_idx)
                
        classes = [classes[node_idx] for node_idx in keep_indices]
        clusters = [clusters[node_idx] for node_idx in keep_indices]
        
        # Add a data point bbox for every cluster
        rect_by_cluster = defaultdict(Rect)
        for cluster, class_idx, bbox in zip(clusters, classes, keep_bboxes):
            if class_idx == 1:
                bbox = [bbox[0]-1, bbox[1]-1, bbox[2]+1, bbox[3]+1]
                rect_by_cluster[cluster].include_rect(list(bbox))
        for cluster, rect in rect_by_cluster.items():
            bbox = list(rect)
            bbox = [bbox[0]+1, bbox[1]+1, bbox[2]-1, bbox[3]-1]
            keep_bboxes.append(bbox)
            classes.append(12)
            clusters.append(cluster)
        
        num_objects = len(keep_bboxes)
        adjacency_matrix = np.zeros((num_objects, num_objects))
        for object1 in range(num_objects):
            for object2 in range(num_objects):
                class_num1 = classes[object1]
                class_num2 = classes[object2]
                cluster1 = clusters[object1]
                cluster2 = clusters[object2]

                if (cluster1 == cluster2 and not cluster1 == 'None' and not class_num1 == class_num2):
                    adjacency_matrix[object1, object2] = 1
            
        # Convert to Torch Tensor
        if len(classes) > 0:
            bboxes = torch.as_tensor(keep_bboxes, dtype=torch.float32)
            classes = torch.as_tensor(classes, dtype=torch.int64)
        else:
            bboxes = torch.empty((0, 4), dtype=torch.float32)
            classes = torch.empty((0,), dtype=torch.int64)

        #num_objs = bboxes.shape[0]

        # Create target
        target = {}
        target["boxes"] = bboxes
        target["labels"] = classes
        target["orig_size"] = torch.as_tensor([int(img.height), int(img.width)])
        target["size"] = torch.as_tensor([int(img.height), int(img.width)])
        target["adjacencies"] = torch.as_tensor(adjacency_matrix)
        target["overall_bbox"] = torch.as_tensor(overall_bbox)
        
        if self.transforms is not None:
            img_tensor, target = self.transforms(img, target)
        else:
            return img, target

        if self.include_original:
            return img_tensor, target, img
        else:
            return img_tensor, target

    def __len__(self):
        return len(self.file_ids)