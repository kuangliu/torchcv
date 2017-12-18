from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torchcv.utils import one_hot_embedding


class SSDLoss(nn.Module):
    def __init__(self, num_classes):
        super(SSDLoss, self).__init__()
        self.num_classes = num_classes

    def _hard_negative_mining(self, cls_loss, cls_targets):
        '''Return negative indices that is 3x the number as postive indices.

        Args:
          cls_loss: (tensor) cross entroy loss between cls_preds and cls_targets, sized [N*#anchors,].
          cls_targets: (tensor) encoded target labels, sized [N, #anchors].

        Return:
          (tensor) negative indices, sized [N,#anchors].
        '''
        batch_size, num_anchors = cls_targets.size()
        loss = cls_loss.view(batch_size, -1).clone()  # [N,#anchors], clone is for the following in-place operation
        loss[cls_targets!=0] = 0                # set pos and ignore loss to 0, the rest are neg cls_loss

        _, idx = loss.sort(1, descending=True)  # sort by neg cls_loss
        _, rank = idx.sort(1)                   # [N,#anchors]

        pos = cls_targets > 0                   # [N,#anchors]
        num_pos = pos.long().sum(1)             # [N,1]
        num_neg = torch.clamp(3*num_pos, max=num_anchors-1)  # [N,]

        neg = rank < num_neg[:,None]  # [N,#anchors]
        return neg

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
        '''Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).

        Args:
          loc_preds: (tensor) predicted locations, sized [N, #anchors, 4].
          loc_targets: (tensor) encoded target locations, sized [N, #anchors, 4].
          cls_preds: (tensor) predicted class confidences, sized [N, #anchors, #classes].
          cls_targets: (tensor) encoded target labels, sized [N, #anchors].

        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + CrossEntropyLoss(cls_preds, cls_targets).
        '''
        pos = cls_targets > 0  # [N,#anchors]
        num_pos = pos.data.long().sum()

        #===============================================================
        # loc_loss = SmoothL1Loss(pos_loc_preds, pos_loc_targets)
        #===============================================================
        mask = pos.unsqueeze(2).expand_as(loc_preds)       # [N,#anchors,4]
        masked_loc_preds = loc_preds[mask].view(-1,4)      # [#pos,4]
        masked_loc_targets = loc_targets[mask].view(-1,4)  # [#pos,4]
        loc_loss = F.smooth_l1_loss(masked_loc_preds, masked_loc_targets, size_average=False)

        #===============================================================
        # cls_loss = CrossEntropyLoss(cls_preds, cls_targets)
        #===============================================================
        cls_loss = F.cross_entropy(cls_preds.view(-1,self.num_classes), \
                                   cls_targets.view(-1), reduce=False)  # [N*#anchors,]
        neg = self._hard_negative_mining(cls_loss, cls_targets)         # [N,#anchors]
        pos_neg = pos | neg
        cls_loss = cls_loss[pos_neg.view(-1)].sum()

        print('loc_loss: %.3f | cls_loss: %.3f' % (100*loc_loss.data[0]/num_pos, cls_loss.data[0]/num_pos), end=' | ')
        loss = (100*loc_loss+cls_loss)/num_pos
        return loss
