import torch
import torch.nn as nn

from fpn import FPN50


class FPNSSD512(nn.Module):
    num_anchors = 9

    def __init__(self, num_classes):
        super(FPNSSD512, self).__init__()
        self.fpn = FPN50()
        self.num_classes = num_classes
        self.loc_head = self._make_head(self.num_anchors*4)
        self.cls_head = self._make_head(self.num_anchors*self.num_classes)

    def forward(self, x):
        loc_preds = []
        cls_preds = []
        fms = self.fpn(x)
        for fm in fms:
            loc_pred = self.loc_head(fm)
            loc_pred = loc_pred.permute(0,2,3,1)
            loc_pred = loc_pred.reshape(loc_pred.size(0), -1, 4)
            loc_preds.append(loc_pred)

            cls_pred = self.cls_head(fm)
            cls_pred = cls_pred.permute(0,2,3,1)
            cls_pred = cls_pred.reshape(cls_pred.size(0), -1, self.num_classes)
            cls_preds.append(cls_pred)

        loc_preds = torch.cat(loc_preds, 1)
        cls_preds = torch.cat(cls_preds, 1)
        return loc_preds, cls_preds

    def _make_head(self, out_planes):
        layers = []
        for _ in range(4):
            layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(256, out_planes, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)


def test():
    net = FPNSSD512(21)
    loc_preds, cls_preds = net(torch.randn(1,3,512,512))
    print(loc_preds.size(), cls_preds.size())

# test()
