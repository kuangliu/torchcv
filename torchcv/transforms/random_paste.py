import torch
import random

from PIL import Image


def random_paste(img, boxes, out_size, left_top=False):
    '''Randomly paste the input image on a larger canvas.

    If boxes is not None, adjust boxes accordingly.

    Args:
      img: (PIL.Image) image to be flipped.
      boxes: (tensor) object boxes, sized [#obj,4].
      out_size: (tuple) output size of (w,h).
      left_top: (bool) place image on the left-top corner.

    Returns:
      canvas: (PIL.Image) canvas with image pasted.
      boxes: (tensor) adjusted object boxes.
    '''
    w, h = out_size
    canvas = Image.new('RGB', (w,h))

    x = 0 if left_top else random.randint(0, w - img.size[0])
    y = 0 if left_top else random.randint(0, h - img.size[1])
    canvas.paste(img, (x,y))

    if boxes is not None:
        boxes = boxes + torch.Tensor([x,y,x,y])
    return canvas, boxes
