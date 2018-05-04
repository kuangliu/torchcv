import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from PIL import Image, ImageDraw
from torchcv.models.fpnssd import FPNSSD512, FPNSSDBoxCoder


print('Loading model..')
net = FPNSSD512(num_classes=9).to('cuda')
net = torch.nn.DataParallel(net)
net.load_state_dict(torch.load('./examples/fpnssd/checkpoint/17.pth'))
net.eval()

print('Loading image..')
img = Image.open('/home/liukuang/data/kitti/training/image_2/000000.png')
ow = oh = 512
img = img.resize((ow,oh))

print('Predicting..')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))])
x = transform(img)
loc_preds, cls_preds = net(x.unsqueeze(0))

print('Decoding..')
box_coder = FPNSSDBoxCoder()
loc_preds = loc_preds.squeeze().cpu()
cls_preds = F.softmax(cls_preds.squeeze(), dim=1).cpu()
boxes, labels, scores = box_coder.decode(loc_preds, cls_preds)

draw = ImageDraw.Draw(img)
for box in boxes:
    draw.rectangle(list(box), outline='red')
img.show()
