"""
©2022 Technology Innovation Institute. All rights reserved.
Author(s): saurabh.pathak@tii.ae
Created: 31 March, 2022

Purpose: Adversarial patch creation based on bounding box
"""
import math

import cv2
import numpy as np
import tifffile


class AdversarialPatch:

    def __init__(self, *, scale, h=512, w=512, patch_file=None):
        if patch_file is not None:
            self._patch_img = tifffile.imread(patch_file).astype('uint8')
        else:
            self._patch_img = (np.random.rand(h, w, 3) * 255).astype('uint8')
        self.scale = scale

    def _create(self, img, bbox):
        ymin, xmin, ymax, xmax = bbox
        h, w = ymax - ymin, xmax - xmin

        area = h * w

        patch_w = int(math.sqrt(area * self.scale))
        patch_h = patch_w

        orig_y = ymin + h / 2.
        orig_x = xmin + w / 2.

        ymin_patch = max(orig_y - patch_h / 2., 0.)
        xmin_patch = max(orig_x - patch_w / 2., 0.)

        img_h, img_w, _ = img.shape
        if ymin_patch + patch_h > img_h:
            ymin_patch = img_h - patch_h

        if xmin_patch + patch_w > img_w:
            xmin_patch = img_w - patch_w

        return list(map(int, (ymin_patch, xmin_patch, patch_h, patch_w)))

    def add_adv_to_img(self, img: np.ndarray, bboxes):
        img = img.copy()
        for bbox in bboxes:
            ymin_patch, xmin_patch, patch_h, patch_w = self._create(img, bbox)
            patch = cv2.resize(self._patch_img, (patch_w, patch_h))

            img[ymin_patch: ymin_patch + patch_h, xmin_patch: xmin_patch + patch_w] = patch
        return img


def main():
    import matplotlib.pyplot as plt
    from matplotlib import patches
    from PIL import Image

    im = np.asarray(Image.open('burj_khalifa_sunset.jpg'))
    adv_patch = AdversarialPatch(scale=.5)
    bbox = ymin, xmin, ymax, xmax = 50, 125, 400, 200
    im = adv_patch.add_adv_to_img(im, [bbox])
    plt.imshow(im)
    rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=1, edgecolor='r', facecolor='none')
    plt.gca().add_patch(rect)
    plt.show()


if __name__ == '__main__':
    main()
