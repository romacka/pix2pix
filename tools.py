import cv2
import torch
import numpy as np
from typing import Tuple
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def resize_with_pad(image: np.array,
                    new_shape: Tuple[int, int],
                    padding_color: Tuple[int] = (255, 255, 255)) -> np.array:
    original_shape = (image.shape[1], image.shape[0])
    ratio = float(max(new_shape))/max(original_shape)
    new_size = tuple([int(x*ratio) for x in original_shape])
    image = cv2.resize(image, new_size)
    delta_w = new_shape[0] - new_size[0]
    delta_h = new_shape[1] - new_size[1]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
    return image

def get_image(image, gen):
    gen.eval()
    with torch.no_grad():
        res = gen(image[None].to(device))
        res = res * 0.5 + 0.5  # remove normalization#
    res = np.rollaxis(res[0].cpu().detach().numpy(), 0, 3)
    return res