import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import json


class BeagleDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transforms=None):
        self.data_dir = data_dir
        self.transforms = transforms
        # load the annotations file, it also contain information of image names
        # load annotations
        annotations1 = json.load(open(os.path.join(data_dir, "via_region_data.json")))
        # print(annotations1)
        self.annotations = list(annotations1.values())  # don't need the dict keys
        # self.annotations = [a for a in annotations if a['regions']]
        

    def __getitem__(self, idx):

        # load images ad masks
        img_name = self.annotations[idx]["filename"]
        img_path = os.path.join(self.data_dir, img_name)
        # mask_path = os.path.join(self.data_dir, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        
        # first id is the background, objects count from 1
        obj_ids = np.array(range(len(self.annotations[idx]["regions"]))) +1
        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        # if num_objs > 0:
        for i in range(num_objs):
            xmin = np.min(self.annotations[idx]["regions"][i]["shape_attributes"]["all_points_x"])
            xmax = np.max(self.annotations[idx]["regions"][i]["shape_attributes"]["all_points_x"])
            ymin = np.min(self.annotations[idx]["regions"][i]["shape_attributes"]["all_points_y"])
            ymax = np.max(self.annotations[idx]["regions"][i]["shape_attributes"]["all_points_y"])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.annotations)