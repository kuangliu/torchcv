# TorchCV: a PyTorch vision library mimics ChainerCV


## Detection
| Model             | Original Paper | ChainerCV  | TorchCV  |
| ----------------- | -------------- | ---------- | -------  |
| SSD300@voc07_test | 74.3%          | 77.8%      |  76.68%  |
| SSD512@voc07_test | 76.8%          | 79.2%      |  78.89%* |

\* I did another experiment by replacing [pytorch/vision](https://github.com/pytorch/vision) VGG16 model with the [model](https://github.com/chainer/chainercv/blob/master/chainercv/links/model/ssd/ssd_vgg16.py#L298) used in ChainerCV. And I got 79.85% accuracy.

## Todo
- [x] SSD300
- [x] SSD512
- [ ] YOLOV2
- [ ] RetinaNet
