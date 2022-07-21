"""
©2022 Technology Innovation Institute. All rights reserved.
Author(s): saurabh.pathak@tii.ae
Created: 31 March, 2022

Purpose: Adversarial patch creation based on bounding box
"""
import math

import cv2
import numpy as np
import scipy.ndimage
from PIL import Image


class AdversarialPatch:

    def __init__(self, *, scale, h=640, w=640, patch_file=None):
        if patch_file is not None:
            self._patch_img = np.asarray(Image.open(patch_file).convert('RGB'))
        else:
            self._patch_img = (np.random.rand(h, w, 3) * 255).astype('uint8')
        self.scale = scale
        self._printed = False
        self.mean_rgb = 127.
        self.stddev_rgb = 128.
        # Image.fromarray(self._patch_img).show()
        self._patch_img = self.print_patch()
        # Image.fromarray(self._patch_img).show()
        self.output_size = h, w

    def print_patch(self):
        if self._printed:
            return self._patch_img

        patch = self._patch_img - self.mean_rgb
        patch /= self.stddev_rgb

        patch *= .5

        patch *= self.stddev_rgb
        patch += self.mean_rgb
        self._printed = True
        return np.clip(patch, 0., 255.).astype('uint8')

    def _create(self, img, bbox):
        img_h, img_w, _ = img.shape
        ymin, xmin, ymax, xmax = bbox
        h, w = ymax - ymin, xmax - xmin

        long_side = max(h, w)

        patch_w = int(long_side * self.scale)
        patch_h = patch_w

        diag = min((2. ** .5) * patch_w, img_w)

        orig_y = ymin + h / 2.
        orig_x = xmin + w / 2.

        ymin_patch = max(orig_y - patch_h / 2., 0.)
        xmin_patch = max(orig_x - patch_w / 2., 0.)

        if ymin_patch + diag > img_h:
            ymin_patch = img_h - diag

        if xmin_patch + diag > img_w:
            xmin_patch = img_w - diag

        return list(map(int, (ymin_patch, xmin_patch, patch_h, patch_w, diag)))

    def add_gray(self, image):
        h, w, c = image.shape
        image_scale_y = self.output_size[0] / h
        image_scale_x = self.output_size[1] / w
        image_scale = min(image_scale_x, image_scale_y)
        scaled_height = int(h * image_scale)
        scaled_width = int(w * image_scale)

        scaled_image = cv2.resize(image, [scaled_width, scaled_height])
        output_image = 127 + np.zeros((*self.output_size, c), dtype='uint8')
        output_image[:scaled_height, :scaled_width, :] = scaled_image
        return output_image

    def brightness_match(self, tgt):
        tgt = self.add_gray(tgt)
        tgt = cv2.cvtColor(tgt, cv2.COLOR_RGB2YUV)
        src = cv2.cvtColor(self._patch_img, cv2.COLOR_RGB2YUV)

        source, target = src[:, :, 0], tgt[:, :, 0]
        source_mean = np.mean(source)
        target_mean = np.mean(target)
        res = np.clip(source - source_mean + target_mean, 0., 255.)

        src[:, :, 0] = res.astype('uint8')
        res = cv2.cvtColor(src, cv2.COLOR_YUV2RGB)
        return res

    def random_brightness(self, tgt, delta):
        delta = np.random.uniform(-delta, delta)
        return np.clip(tgt + delta, -1., 1.)

    def random_noise(self, tgt, delta):
        delta = np.random.uniform(low=-delta, high=delta, size=tgt.shape)
        return np.clip(tgt + delta, -1., 1.)

    def resize(self, patch, patch_h, patch_w):
        h, w, _ = patch.shape
        if h > patch_h:
            patch = cv2.resize(patch, (patch_w, patch_h), interpolation=cv2.INTER_AREA)
        elif h < patch_h:
            patch = cv2.resize(patch, (patch_w, patch_h), interpolation=cv2.INTER_CUBIC)
        return patch

    def get_transformed_patch(self, img, patch_h, patch_w):
        patch = self.brightness_match(img)
        patch = self.resize(patch, patch_h, patch_w)
        patch = patch - self.mean_rgb
        patch /= self.stddev_rgb
        patch = self.random_noise(patch, .01)
        # patch = self.random_brightness(patch, .3)
        patch *= self.stddev_rgb
        patch += self.mean_rgb
        # return patch
        return np.clip(patch, None, 255.).astype(int)

    def add_adv_to_img(self, img: np.ndarray, bboxes):
        img = img.copy()
        for bbox in bboxes:
            ymin_patch, xmin_patch, patch_h, patch_w, diag = self._create(img, bbox)
            ymax_patch = ymin_patch + diag
            xmax_patch = xmin_patch + diag
            patch = self.get_transformed_patch(img, patch_h, patch_w)

            patch = scipy.ndimage.rotate(patch, 5, cval=-256)
            patch_h, patch_w, _ = patch.shape
            pads = (diag - patch_h) / 2
            top = left = math.floor(pads)
            bottom = right = math.ceil(pads)
            pads = [[top, bottom], [left, right], [0, 0]]

            patch = np.pad(patch, pads, constant_values=-256)
            patch_bg = img[ymin_patch: ymax_patch, xmin_patch: xmax_patch].copy()
            img_affected = img[ymin_patch: ymax_patch, xmin_patch: xmax_patch] = patch
            img_affected = np.where(img_affected < 0, patch_bg, img_affected)
            img[ymin_patch: ymax_patch, xmin_patch: xmax_patch] = img_affected
        return img.astype('uint8')


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
