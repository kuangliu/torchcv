from __future__ import print_function

import os
import random
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

from torchcv.models.fpnssd import FPNSSD512
from torchcv.models.ssd import SSD300, SSD512, SSDBoxCoder

from torchcv.loss import SSDLoss
from torchcv.datasets import ListDataset
from torchcv.transforms import resize, random_flip, random_paste, random_crop, random_distort


parser = argparse.ArgumentParser(description='PyTorch SSD Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--model', default='./examples/ssd/model/ssd512_vgg16.pth', type=str, help='initialized model path')
parser.add_argument('--checkpoint', default='./examples/ssd/checkpoint/ckpt.pth', type=str, help='checkpoint path')
args = parser.parse_args()

# Model
print('==> Building model..')
# net = SSD512(num_classes=21)
net = FPNSSD512(num_classes=21)
net.load_state_dict(torch.load(args.model))
best_loss = float('inf')  # best test loss
start_epoch = 0  # start from epoch 0 or last epoch
if args.resume:
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(args.checkpoint)
    net.load_state_dict(checkpoint['net'])
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']

# Dataset
print('==> Preparing dataset..')
box_coder = SSDBoxCoder(net)
img_size = 512
def transform_train(img, boxes, labels):
    img = random_distort(img)
    if random.random() < 0.5:
        img, boxes = random_paste(img, boxes, max_ratio=4, fill=(123,116,103))
    img, boxes, labels = random_crop(img, boxes, labels)
    img, boxes = resize(img, boxes, size=(img_size,img_size), random_interpolation=True)
    img, boxes = random_flip(img, boxes)
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])(img)
    boxes, labels = box_coder.encode(boxes, labels)
    return img, boxes, labels

trainset = ListDataset(root='/search/odin/liukuang/data/voc_all_images',
                       list_file=['torchcv/datasets/voc/voc07_trainval.txt',
                                  'torchcv/datasets/voc/voc12_trainval.txt'],
                       transform=transform_train)

def transform_test(img, boxes, labels):
    img, boxes = resize(img, boxes, size=(img_size,img_size))
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])(img)
    boxes, labels = box_coder.encode(boxes, labels)
    return img, boxes, labels

testset = ListDataset(root='/search/odin/liukuang/data/voc_all_images',
                      list_file='torchcv/datasets/voc/voc07_test.txt',
                      transform=transform_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=8)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=8)

net.cuda()
net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
cudnn.benchmark = True

criterion = SSDLoss(num_classes=21)
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
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
