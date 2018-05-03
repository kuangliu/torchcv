import os
import torch
from torchcv.models.fpnssd import FPNSSD512


model_dir = './examples/fpnssd/model'
params = torch.load(os.path.join(model_dir, 'resnet50-19c8e357.pth'))

net = FPNSSD512(num_classes=9)
net.fpn.load_state_dict(params, strict=False)
torch.save(net.state_dict(), os.path.join(model_dir, 'fpnssd512_resnet50.pth'))
