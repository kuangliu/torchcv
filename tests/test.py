from PIL import Image

import torchvision.transforms as transforms

from torchcv.transforms.resize import resize
from torchcv.transforms.random_flip import random_flip

from torchcv.datasets.listdataset import ListDataset
from torchcv.visualizations.vis_image import vis_image


def transform(img, boxes):
    img, boxes = resize(img, boxes, 600)
    img, boxes = random_flip(img, boxes)
    img = transforms.ToTensor()(img)
    return img, boxes

dataset = ListDataset(root='/mnt/hgfs/D/mscoco/2017/val2017',
                      list_file='torchcv/datasets/mscoco/coco17_val.txt',
                      transform=transform)

img, boxes, labels = dataset[0]
vis_image(img, boxes)
