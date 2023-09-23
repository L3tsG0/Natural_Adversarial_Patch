import os
import matplotlib.pyplot as plt
import pathlib
import numpy as np


def plot_images(data):
    total = len(data)
    if total % 2 == 0:
        rows = total // 2
    else:
        rows = total // 2 + 1

    plt.figure(figsize=(10, 8))
    for i, img in enumerate(data):
        img = plt.imread(img)
        plt.subplot(rows, 2, i + 1)
        plt.imshow(img)
        plt.xlabel(img.shape)
    plt.show()


def rescale_mask(img):
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img[img >= 0.5] = 1
    img[img < 0.5] = 0
    return img


img_dir = pathlib.Path("data/images")
mask_data_dir = pathlib.Path("data/segmentations")
img_path_list = list(img_dir.iterdir())
mask_path_list = list(mask_data_dir.iterdir())

first_img_path = img_path_list[0]
first_img_mask_path = mask_path_list[0]

plot_images([first_img_path, first_img_mask_path])
