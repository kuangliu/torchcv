from PIL import Image

import torchvision.transforms as transforms

from torchcv.transforms import resize
from torchcv.transforms import random_flip
from torchcv.transforms import random_paste
from torchcv.transforms import scale_jitter

from torchcv.datasets import ListDataset
from torchcv.visualizations import vis_image


def transform(img, boxes, labels):
    img, boxes = resize(img, boxes, size=600)
    img, boxes = random_flip(img, boxes)
    img, boxes = random_paste(img, boxes, (1000,1000), left_top=True)
    img, boxes = scale_jitter(img, boxes, sizes=(600,700,800,900,1000), max_size=1400)
    img = transforms.ToTensor()(img)
    return img, boxes, labels

dataset = ListDataset(root='/mnt/hgfs/D/mscoco/2017/val2017',
                      list_file='torchcv/datasets/mscoco/coco17_val.txt',
                      transform=transform)

img, boxes, labels = dataset[0]
vis_image(img, boxes)
