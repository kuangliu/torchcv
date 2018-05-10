'''RetinaNet train on COCO.'''
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

from torchcv.loss import FocalLoss
from torchcv.datasets import ListDataset
from torchcv.models.retinanet import RetinaBoxCoder, RetinaNet
from torchcv.transforms import pad, resize, random_flip, random_paste


parser = argparse.ArgumentParser(description='PyTorch RetinaNet Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--model', default='./examples/retinanet/model/net.pth', type=str, help='pretrained model path')
parser.add_argument('--checkpoint', default='./examples/retinanet/checkpoint/ckpt.pth', type=str, help='checkpoint path')
args = parser.parse_args()


# Dataset
print('==> Preparing dataset..')
img_size = 640
box_coder = RetinaBoxCoder()
def transform_train(img, boxes, labels):
    img, boxes = random_flip(img, boxes)
    img, boxes = resize(img, boxes, size=img_size, max_size=img_size)
    img = pad(img, (img_size, img_size))
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])(img)
    boxes, labels = box_coder.encode(boxes, labels)
    return img, boxes, labels

def transform_test(img, boxes, labels):
    img, boxes = resize(img, boxes, size=img_size, max_size=img_size)
    img = pad(img, (img_size, img_size))
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])(img)
    boxes, labels = box_coder.encode(boxes, labels)
    return img, boxes, labels

trainset = ListDataset(root='/home/liukuang/data/coco/train2017',
                       list_file='torchcv/datasets/mscoco/coco17_train.txt',
                       transform=transform_train)
testset = ListDataset(root='/home/liukuang/data/coco/val2017',
                      list_file='torchcv/datasets/mscoco/coco17_val.txt',
                      transform=transform_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=8)
testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False, num_workers=8)

# Model
print('==> Building model..')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = RetinaNet(num_classes=80).to(device)
# net.load_state_dict(torch.load(args.model))
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

best_loss = float('inf')  # best test loss
start_epoch = 0  # start from epoch 0 or last epoch
if args.resume:
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(args.checkpoint)
    net.load_state_dict(checkpoint['net'])
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']

criterion = FocalLoss(num_classes=80)
# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
optimizer = optim.Adam(net.parameters(), lr=args.lr, amsgrad=True)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(trainloader):
        inputs = inputs.to(device)
        loc_targets = loc_targets.to(device)
        cls_targets = cls_targets.to(device)

        optimizer.zero_grad()
        loc_preds, cls_preds = net(inputs)
        loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        print('train_loss: %.3f | avg_loss: %.3f [%d/%d]'
              % (loss.item(), train_loss/(batch_idx+1), batch_idx+1, len(trainloader)))

def test(epoch):
    print('\nTest')
    net.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(testloader):
            inputs = inputs.to(device)
            loc_targets = loc_targets.to(device)
            cls_targets = cls_targets.to(device)

            loc_preds, cls_preds = net(inputs)
            loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
            test_loss += loss.item()
            print('test_loss: %.3f | avg_loss: %.3f [%d/%d]'
                  % (loss.item(), test_loss/(batch_idx+1), batch_idx+1, len(testloader)))

    # Save checkpoint
    global best_loss
    test_loss /= len(testloader)
    if test_loss < best_loss:
        print('Saving..')
        state = {
            'net': net.state_dict(),
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
