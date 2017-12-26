'''Encode object boxes and labels.'''
import math
import torch
import itertools

from torchcv.utils import meshgrid
from torchcv.utils.box import box_iou, box_nms, change_box_order


class SSDBoxCoder:
    def __init__(self):
        self.steps = (8, 16, 32, 64, 100, 300)
        self.box_sizes = (30, 60, 111, 162, 213, 264, 315)  # default bounding box sizes for each feature map.
        self.aspect_ratios = ((2,), (2,3), (2,3), (2,3), (2,), (2,))
        self.fm_sizes = (38, 19, 10, 5, 3, 1)
        self.default_boxes = self._get_default_boxes()

    def _get_default_boxes(self):
        boxes = []
        for i, fm_size in enumerate(self.fm_sizes):
            for h, w in itertools.product(range(fm_size), repeat=2):
                cx = (w + 0.5) * self.steps[i]
                cy = (h + 0.5) * self.steps[i]

                s = self.box_sizes[i]
                boxes.append((cx, cy, s, s))

                s = math.sqrt(self.box_sizes[i] * self.box_sizes[i+1])
                boxes.append((cx, cy, s, s))

                s = self.box_sizes[i]
                for ar in self.aspect_ratios[i]:
                    boxes.append((cx, cy, s * math.sqrt(ar), s / math.sqrt(ar)))
                    boxes.append((cx, cy, s / math.sqrt(ar), s * math.sqrt(ar)))
        return torch.Tensor(boxes)  # xywh

    def encode(self, boxes, labels):
        '''Encode target bounding boxes and class labels.

        SSD coding rules:
          tx = (x - anchor_x) / (variance[0]*anchor_w)
          ty = (y - anchor_y) / (variance[0]*anchor_h)
          tw = log(w / anchor_w) / variance[1]
          th = log(h / anchor_h) / variance[1]

        Args:
          boxes: (tensor) bounding boxes of (xmin,ymin,xmax,ymax), sized [#obj, 4].
          labels: (tensor) object class labels, sized [#obj,].

        Returns:
          loc_targets: (tensor) encoded bounding boxes, sized [#anchors,4].
          cls_targets: (tensor) encoded class labels, sized [#anchors,].

        Reference:
          https://github.com/chainer/chainercv/blob/master/chainercv/links/model/ssd/multibox_coder.py
        '''
        def argmax(x):
            v, i = x.max(0)
            j = v.max(0)[1][0]
            return (i[j], j)

        default_boxes = self.default_boxes  # xywh
        default_boxes = change_box_order(default_boxes, 'xywh2xyxy')

        ious = box_iou(default_boxes, boxes)  # [#anchors, #obj]
        index = torch.LongTensor(len(default_boxes)).fill_(-1)
        masked_ious = ious.clone()
        while True:
            i, j = argmax(masked_ious)
            if masked_ious[i,j] < 1e-6:
                break
            index[i] = j
            masked_ious[i,:] = 0
            masked_ious[:,j] = 0

        mask = (index<0) & (ious.max(1)[0]>=0.5)
        if mask.any():
            index[mask] = ious[mask.nonzero().squeeze()].max(1)[1]

        boxes = boxes[index.clamp(min=0)]  # negative index not supported
        boxes = change_box_order(boxes, 'xyxy2xywh')
        default_boxes = change_box_order(default_boxes, 'xyxy2xywh')

        variances = (0.1, 0.2)
        loc_xy = (boxes[:,:2]-default_boxes[:,:2]) / default_boxes[:,2:] / variances[0]
        loc_wh = torch.log(boxes[:,2:]/default_boxes[:,2:]) / variances[1]
        loc_targets = torch.cat([loc_xy,loc_wh], 1)
        cls_targets = 1 + labels[index.clamp(min=0)]
        cls_targets[index<0] = 0
        return loc_targets, cls_targets

    def decode(self, loc_preds, cls_preds):
        '''Transform predicted loc/cls back to real box locations and class labels.

        Args:
          loc_preds: (tensor) predicted loc, sized [8732,4].
          cls_preds: (tensor) predicted conf, sized [8732,21].

        Returns:
          boxes: (tensor) bbox locations, sized [#obj, 4].
          labels: (tensor) class labels, sized [#obj,1].
        '''
        variances = (0.1, 0.2)
        wh = torch.exp(loc[:,2:]*variances[1]) * self.default_boxes[:,2:]
        cxcy = loc[:,:2] * variances[0] * self.default_boxes[:,2:] + self.default_boxes[:,:2]
        boxes = torch.cat([cxcy-wh/2, cxcy+wh/2], 1)  # [8732,4]

        max_conf, labels = conf.max(1)  # [8732,1]
        ids = labels.squeeze(1).nonzero().squeeze(1)  # [#boxes,]

        keep = box_nms(boxes[ids], max_conf[ids].squeeze(1))
        return boxes[ids][keep], labels[ids][keep], max_conf[ids][keep]
