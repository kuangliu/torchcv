# TorchCV: a PyTorch vision library mimics ChainerCV


## Detection
| Model                | Original Paper | ChainerCV  | TorchCV*  |
| -------------------- | -------------- | ---------- | -------   |
| SSD300@voc07_test    | 74.3%          | 77.8%      |  76.68%   |
| SSD512@voc07_test    | 76.8%          | 79.2%      |  78.89%   |
| FPNSSD512@voc07_test | -              | -          |  81.46%   |

The accuracy of TorchCV SSD is ~1% lower than ChainerCV. This is because the VGG base model I use performs slightly worse.  
I did the experiment by replacing [pytorch/vision](https://github.com/pytorch/vision) VGG16 model with the [model](https://github.com/chainer/chainercv/blob/master/chainercv/links/model/ssd/ssd_vgg16.py#L298) used in ChainerCV, the SSD512 model got __79.85%__ accuracy.

FPNSSD512 is created by replacing SSD VGG16 network with FPN50, the rest is the same. It beats all SSD models.


## TODO
- [x] SSD300
- [x] SSD512
- [x] FPNSSD512
- [ ] YOLOV2
- [ ] RetinaNet
