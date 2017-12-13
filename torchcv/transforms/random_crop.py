'''This random crop strategy is described in paper:
   [1] SSD: Single Shot MultiBox Detector
'''
import torch
import random

from PIL import Image
from torchcv.utils.box import box_iou, box_clamp


def random_crop(img, boxes, labels):
    '''Randomly crop PIL image.

    Args:
      img: (PIL.Image) image.
      boxes: (tensor) bounding boxes, sized [#obj, 4].
      labels: (tensor) bounding box labels, sized [#obj,].

    Returns:
      img: (PIL.Image) cropped image.
      selected_boxes: (tensor) selected boxes.
      selected_labels: (tensor) selected box labels.
    '''
    imw, imh = img.size
    while True:
        '''
        When min_iou is:
          1. None: use original image
          2. 0: randomly sample a patch
          3. 0.1~0.9: minimal iou > min_iou
        '''
        min_iou = random.choice([None, 0, 0.1, 0.3, 0.5, 0.7, 0.9])
        if min_iou is None:
            return img, boxes, labels

        for _ in range(100):
            w = random.randrange(int(0.1*imw), imw)
            h = random.randrange(int(0.1*imh), imh)

            if h > 2*w or w > 2*h:
                continue

            x = random.randrange(imw - w)
            y = random.randrange(imh - h)
            roi = torch.Tensor([[x, y, x+w, y+h]])

            center = (boxes[:,:2] + boxes[:,2:]) / 2              # [N,2]
            roi2 = roi.expand(len(center), 4)                     # [N,4]
            mask = (center > roi2[:,:2]) & (center < roi2[:,2:])  # [N,2]
            mask = mask[:,0] & mask[:,1]                          # [N,]
            if not mask.any():
                continue

            selected_boxes = boxes[mask.nonzero().squeeze(),:]
            iou = box_iou(selected_boxes, roi)
            if iou.max() < min_iou:
                continue

            img = img.crop((x,y,x+w,y+h))
            selected_boxes = selected_boxes - torch.Tensor([x,y,x,y])
            selected_boxes = box_clamp(selected_boxes, 0,0,w,h)
            return img, selected_boxes, labels[mask]
