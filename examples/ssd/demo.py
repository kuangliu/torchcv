import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image, ImageDraw
from torch.autograd import Variable
from torchcv.models.fpnssd import FPNSSD512
from torchcv.models.ssd import SSDBoxCoder


print('Loading model..')
net = FPNSSD512(num_classes=21)
net.load_state_dict(torch.load('./examples/ssd/checkpoint/params.pth'))
net.eval()

print('Loading image..')
img = Image.open('./image/000020.jpg')
ow = oh = 512
img = img.resize((ow,oh))

print('Predicting..')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])
x = transform(img)
x = Variable(x, volatile=True)
loc_preds, cls_preds = net(x.unsqueeze(0))

print('Decoding..')
box_coder = SSDBoxCoder(net)
boxes, labels, scores = box_coder.decode(
    loc_preds.data.squeeze(), F.softmax(cls_preds.squeeze(), dim=1).data)
print(labels)
print(scores)

draw = ImageDraw.Draw(img)
for box in boxes:
    draw.rectangle(list(box), outline='red')
img.show()
