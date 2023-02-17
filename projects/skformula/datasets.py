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
import torchvision.transforms.functional as L

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

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = L.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "coords" in target and len(target["coords"]) > 0:
            coords = target["coords"]
            coords = coords / torch.tensor([w, h], dtype=torch.float32)
            target["coords"] = coords
        return image, target

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
            
            min_left = int(round(min([coord[0].item() for coord in target['coords']])))
            min_top = int(round(min([coord[1].item() for coord in target['coords']])))
            max_right = int(round(max([coord[0].item() for coord in target['coords']])))
            max_bottom = int(round(max([coord[1].item() for coord in target['coords']])))
            
            left = random.randint(max(0, min_left-self.max_left_padding), max(0, min_left-self.min_left_padding))
            top = random.randint(max(0, min_top-self.max_top_padding), max(0, min_top-self.min_top_padding))
            right = random.randint(min(width, max_right+self.min_right_padding), min(width, max_right+self.max_right_padding))
            bottom = random.randint(min(height, max_bottom+self.min_bottom_padding), min(height, max_bottom+self.max_bottom_padding))         
            
            cropped_image = image.crop((left, top, right, bottom))
            cropped_coords = []
            cropped_labels = []
            for coord, label in zip(target["coords"], target["labels"]):
                coord = [max(coord[0], left) - left,
                        max(coord[1], top) - top]

                cropped_coords.append(coord)
                cropped_labels.append(label)
                         
            if len(cropped_coords) > 0:
                target["coords"] = torch.as_tensor(cropped_coords, dtype=torch.float32)
                target["labels"] = torch.as_tensor(cropped_labels, dtype=torch.int64)
                return cropped_image, target

        return image, target
        
class RandomErasingWithTarget(object):
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=255, inplace=False):
        self.transform = transforms.RandomErasing(p=p,
                                                  scale=scale,
                                                  ratio=ratio,
                                                  value=value,
                                                  inplace=False)

    def __call__(self, img: PIL.Image.Image, target: dict):
        img = self.transform(img)

        return img, target
        
class ToPILImageWithTarget(object):
    def __init__(self):
        self.transform = transforms.ToPILImage()

    def __call__(self, img: PIL.Image.Image, target: dict):
        img = self.transform(img)

        return img, target
    
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
        resized_coords = []
        for coord in target["coords"]:
            coord = [scale*elem for elem in coord]
            resized_coords.append(coord)

        target["coords"] = torch.as_tensor(resized_coords, dtype=torch.float32)
        
        return resized_image, target
    
class RandomBlackAndWhite(object):
    def __init__(self, probability=0.5, threshold=200):
        self.probability = probability
        self.threshold = threshold
        self.fn = lambda x : 255 if x > threshold else 0

    def __call__(self, img: PIL.Image.Image, target: dict):
        r = random.random()
        if r <= self.probability:
            img = img.convert('L').point(self.fn, mode='1')
            img = img.convert('RGB')
        
        return img, target
    
    
class RandomSaltAndPepper(object):
    def __init__(self, probability=0.5, salt_amount=0.1, pepper_amount=0.0001):
        self.probability = probability
        self.salt_amount = salt_amount
        self.pepper_amount = pepper_amount

    def __call__(self, img: PIL.Image.Image, target: dict):
        r = random.random()
        if r <= self.probability:
            img = np.array(img)
            img = util.random_noise(img,mode='salt',amount=self.salt_amount)
            img = util.random_noise(img,mode='pepper',amount=self.pepper_amount)
            img = Image.fromarray(np.uint8(img*255))
        
        return img, target


def get_transform(split):
    """
    Returns the appropriate transforms.
    """

    normalize = R.Compose([
        R.ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    random_erasing = R.Compose([
        R.ToTensor(),
        RandomErasingWithTarget(p=0.2, scale=(0.003, 0.03), ratio=(0.1, 0.3), value='random'),
        RandomErasingWithTarget(p=0.2, scale=(0.003, 0.03), ratio=(0.3, 1), value='random'),
        ToPILImageWithTarget()
    ])

    if split == 'train':
        return R.Compose([
            RandomMaxResize(350, 900),
            RandomDilation(0.18, 3),
            RandomMaxResize(700, 900),
            random_erasing,
            normalize
        ])

    if split == 'val' or split == 'test':
        return R.Compose([
            RandomMaxResize(800, 800),
            normalize
        ])
        
    if split == 'val_no_norm':
        return R.Compose([
            RandomMaxResize(800, 800)
        ])

    raise ValueError(f'unknown {split}')


# TODO: Add code for converting to COCO
class SkeletalFormulaDataset(torch.utils.data.Dataset):
    def __init__(self, data_directory, transforms=None, max_size=None, do_crop=True,
                 include_original=False, include_smiles=False):
        self.transforms = transforms
        self.do_crop=do_crop
        self.data_directory = data_directory
        self.include_smiles = include_smiles

        files = os.listdir(data_directory)
        img_file_stems = [elem.replace(".jpg", "") for elem in files if elem.endswith(".jpg")]
        json_file_stems = [elem.replace(".json", "") for elem in files if elem.endswith(".json")]
        self.file_stems = list(set(img_file_stems).intersection(set(json_file_stems)))
            
        sample_ids = range(len(self.file_stems))
        self.sample_ids = sample_ids
        
        if not max_size is None:
            self.sample_ids = random.sample(self.sample_ids, max_size)

        self.include_original = include_original

    def __getitem__(self, idx):
        # load images ad masks
        file_stem = self.file_stems[idx]
        
        img = Image.open(os.path.join(self.data_directory, file_stem + ".jpg"))
        with open(os.path.join(self.data_directory, file_stem + ".json"), 'r') as f:
            data = json.load(f)

        w, h = img.size
            
        # Convert to Torch Tensor
        if len(data['labels']) > 0:
            coords = torch.as_tensor(data['coords'], dtype=torch.float32)
            labels = torch.as_tensor(data['labels'], dtype=torch.int64)
            hydrogens_labels = torch.as_tensor(data['hydrogens'], dtype=torch.int64)
            formal_charge_labels = torch.as_tensor(data['formal_charges'], dtype=torch.int64)
        else:
            print("NO LABELS")
            coords = torch.empty((0, 2), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)
            hydrogens_labels = torch.empty((0,), dtype=torch.int64)
            formal_charge_labels = torch.empty((0,), dtype=torch.int64)

        num_objs = coords.shape[0]

        # Create target
        target = {}
        if self.include_smiles:
            target["smiles"] = data['smiles']
        target["coords"] = coords
        target["labels"] = labels
        target["hydrogens_labels"] = hydrogens_labels
        target["formal_charge_labels"] = formal_charge_labels
        target["image_id"] = torch.as_tensor([idx])
        target["iscrowd"] = torch.zeros((num_objs,), dtype=torch.int64)
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        target["adjacencies"] = torch.as_tensor(data['bond_types'])
        target["bond_directions"] = torch.as_tensor(data['bond_directions'])
        
        if self.transforms is not None:
            img_tensor, target = self.transforms(img, target)

        if self.include_original:
            return img_tensor, target, img
        else:
            return img_tensor, target

    def __len__(self):
        return len(self.sample_ids)
    
    def getImgIds(self):
        return range(len(self.sample_ids))