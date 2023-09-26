import os

import numpy as np
import torch
from typing import List
import matplotlib.pyplot as plt

def getColors():
    return np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255], [255, 0, 255],
            [176, 48, 96], [46, 139, 87], [160, 32, 240], [255, 127, 80], [127, 255, 212],
            [218, 112, 214], [160, 82, 45], [127, 255, 0], [216, 191, 216], [128, 0, 0], [0, 128, 0],
            [0, 0, 128]])

def getClassificationMap(label: torch.tensor or np.ndarray, unknown=[]):
    colors = getColors()
    image = np.zeros((*label.shape, 3), dtype='uint8')
    for cls in range(1, label.max() + 1):
        image[np.where(label == cls)] = colors[cls - 1]

    for cls in unknown:
        image[np.where(label == cls)] = [255, 255, 255]

    return image

def clearBackground(info, image):
    from scipy.io import loadmat
    dataset_path = os.path.join('./datasets', info.path)
    gt = loadmat(os.path.join(dataset_path, info.gt_file_name))[info.gt_mat_name].astype(np.int64)

    image[np.where(gt == 0)] = [0, 0, 0]

    return image

def parsePredictionLabel(label: List[torch.tensor], H):
    label = torch.cat(label) + 1
    return label.reshape(H, -1)

def drawPredictionMap(label: List[torch.tensor], args, info, draw_background=True):
    label = parsePredictionLabel(label, info.image_width)
    image = getClassificationMap(label, unknown=[label.max()])
    
    if draw_background is False:
        image = clearBackground(info, image)
    saveImage(image, args.log_name, 'map')

def saveImage(image, name, path='map'):
    if not os.path.exists(path):
        os.makedirs(path)

    plt.imsave(f'{os.path.join(path, name)}.png', image)