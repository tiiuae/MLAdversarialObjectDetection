"""
©2022 Technology Innovation Institute. All rights reserved.
Author(s): saurabh.pathak@tii.ae
Created: 31 March, 2022

Purpose: Adversarial patch creation based on bounding box
"""
import cv2
import numpy as np
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

        w = np.random.normal(.5, .1, size=(1, 1, 3))
        b = np.random.normal(0., .01, size=(1, 1, 3))
        patch = w * patch + b

        patch *= self.stddev_rgb
        patch += self.mean_rgb
        self._printed = True
        return np.clip(patch, 0., 255.).astype('uint8')

    def _create(self, img, bbox):
        ymin, xmin, ymax, xmax = bbox
        h, w = ymax - ymin, xmax - xmin

        long_side = max(h, w)

        patch_w = int(long_side * self.scale)
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

    def get_transformed_patch(self, img, patch_h, patch_w):
        patch = self.brightness_match(img)
        patch = cv2.resize(patch, (patch_w, patch_h), interpolation=cv2.INTER_AREA)
        patch = patch - self.mean_rgb
        patch /= self.stddev_rgb
        patch = self.random_noise(patch, .01)
        # patch = self.random_brightness(patch, .3)
        patch *= self.stddev_rgb
        patch += self.mean_rgb
        return np.clip(patch, 0., 255.).astype('uint8')

    def add_adv_to_img(self, img: np.ndarray, bboxes):
        img = img.copy()
        for bbox in bboxes:
            ymin_patch, xmin_patch, patch_h, patch_w = self._create(img, bbox)
            patch = self.get_transformed_patch(img, patch_h, patch_w)
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
