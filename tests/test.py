from PIL import Image

import torchvision.transforms as transforms

from torchcv.transforms import resize
from torchcv.transforms import random_flip

from torchcv.datasets import ListDataset
from torchcv.visualizations import vis_image
from torchcv.models.retinanet import BoxCoder


def transform(img, boxes):
    img, boxes = resize(img, boxes, size=600)
    img, boxes = random_flip(img, boxes)
    img = transforms.ToTensor()(img)
    return img, boxes

box_coder = BoxCoder()

dataset = ListDataset(root='/mnt/hgfs/D/mscoco/2017/val2017',
                      list_file='torchcv/datasets/mscoco/coco17_val.txt',
                      transform=transform, box_coder=box_coder)

img, boxes, labels = dataset[0]
print(img.size())
print(boxes.size())
print(labels.size())
