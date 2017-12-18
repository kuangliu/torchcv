from __future__ import print_function

import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from PIL import Image
from torch.autograd import Variable

from torchcv.datasets import ListDataset
from torchcv.transforms import resize, random_flip, random_paste
from torchcv.models.loss import FocalLoss
from torchcv.models.retinanet import BoxCoder, RetinaNet


parser = argparse.ArgumentParser(description='PyTorch RetinaNet Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--model', default='./examples/retinanet/model/net.pth', type=str, help='pretrained model path')
parser.add_argument('--checkpoint', default='./examples/retinanet/checkpoint/ckpt.pth', type=str, help='checkpoint path')
args = parser.parse_args()


# Dataset
print('==> Preparing dataset..')
def transform_train(img, boxes, labels):
    img, boxes = resize(img, boxes, size=600)
    img, boxes = random_flip(img, boxes)
    return img, boxes, labels

def transform_test(img, boxes, labels):
    img, boxes = resize(img, boxes, size=600)
    return img, boxes, labels

trainset = ListDataset(root='/mnt/hgfs/D/mscoco/2017/train2017',
                       list_file='torchcv/datasets/mscoco/coco17_train.txt',
                       transform=transform_train)
testset = ListDataset(root='/mnt/hgfs/D/mscoco/2017/val2017',
                      list_file='torchcv/datasets/mscoco/coco17_val.txt',
                      transform=transform_test)

# DataLoader
box_coder = BoxCoder()
def collate_fn(batch):
    imgs = [x[0] for x in batch]  # [PIL.Image]
    boxes = [x[1] for x in batch]
    labels = [x[2] for x in batch]

    num_imgs = len(imgs)
    max_w = max([im.size[0] for im in imgs])
    max_h = max([im.size[1] for im in imgs])
    inputs = torch.zeros(num_imgs, 3, max_h, max_w)

    loc_targets = []
    cls_targets = []
    for i in range(num_imgs):
        img = imgs[i]
        boxes_i = boxes[i].clone()
        labels_i = labels[i].clone()

        img, boxes_i = random_paste(img, boxes_i, (max_w,max_h), True)
        inputs[i] = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
        ])(img)

        loc_target, cls_target = box_coder.encode(boxes_i, labels_i, (max_w,max_h))
        loc_targets.append(loc_target)
        cls_targets.append(cls_target)
    return inputs, torch.stack(loc_targets), torch.stack(cls_targets)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=4, collate_fn=collate_fn)
testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False, num_workers=4, collate_fn=collate_fn)

# Model
net = RetinaNet(num_classes=90)
net.load_state_dict(torch.load(args.model))  # load pretrained model

best_loss = float('inf')  # best test loss
start_epoch = 0  # start from epoch 0 or last epoch
if args.resume:
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(args.checkpoint)
    net.load_state_dict(checkpoint['net'])
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']

net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
net.cuda()

criterion = FocalLoss(num_classes=90)
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    net.module.freeze_bn()
    train_loss = 0
    for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(trainloader):
        inputs = Variable(inputs.cuda())
        loc_targets = Variable(loc_targets.cuda())
        cls_targets = Variable(cls_targets.cuda())

        optimizer.zero_grad()
        loc_preds, cls_preds = net(inputs)
        loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        print('train_loss: %.3f | avg_loss: %.3f [%d/%d]'
              % (loss.data[0], train_loss/(batch_idx+1), batch_idx+1, len(trainloader)))

# Test
def test(epoch):
    print('\nTest')
    net.eval()
    test_loss = 0
    for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(testloader):
        inputs = Variable(inputs.cuda(), volatile=True)
        loc_targets = Variable(loc_targets.cuda())
        cls_targets = Variable(cls_targets.cuda())

        loc_preds, cls_preds = net(inputs)
        loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        test_loss += loss.data[0]
        print('test_loss: %.3f | avg_loss: %.3f [%d/%d]'
              % (loss.data[0], test_loss/(batch_idx+1), batch_idx+1, len(testloader)))

    # Save checkpoint
    global best_loss
    test_loss /= len(testloader)
    if test_loss < best_loss:
        print('Saving..')
        state = {
            'net': net.module.state_dict(),
            'loss': test_loss,
            'epoch': epoch,
        }
        if not os.path.isdir(os.path.dirname(args.checkpoint)):
            os.mkdir(os.path.dirname(args.checkpoint))
        torch.save(state, args.checkpoint)
        best_loss = test_loss


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
