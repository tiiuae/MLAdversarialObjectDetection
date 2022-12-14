"""
©2022 Technology Innovation Institute. All rights reserved.
Author(s): saurabh.pathak@tii.ae
Created: 31 March, 2022

Purpose: Adversarial patch addition to persons based on bounding box. This adds adversarial patch to an image.
It uses numpy and opencv and is executed on the CPU. This class is suited for inference time and should only be used
for that. During training the attacker module already implements everything in this class inside a tf graph which can be
placed on the GPU and can execute much faster during training
"""
import cv2
import numpy as np
from PIL import Image


class AdversarialPatch:
    """add adversarial patch to an image"""

    def __init__(self, *, scale, h=640, w=640, patch_file=None):
        """
        init
        :param scale: scale relative to bounding box length
        :param h: height of original image
        :param w: width of original image
        :param patch_file: patch to be added or none if random patch
        """
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
        """
        simulate variations incurred during printing and reimaging the patch, were it a physical patch. Here using
        deterministic values for inference time instead of probabilistic implementation of the tensorflow version
        :return: "printed and reimaged" patch
        """
        if self._printed:
            return self._patch_img

        # normalize
        patch = self._patch_img - self.mean_rgb
        patch /= self.stddev_rgb

        patch *= .5  # tf version uses a linear MoG centered around this value

        # denormalize
        patch *= self.stddev_rgb
        patch += self.mean_rgb
        self._printed = True
        return np.clip(patch, 0., 255.).astype('uint8')

    def _create(self, img, bbox):
        """
        creates the coordinates of the area to be patched based on the person bounding box.
        patch rotation not implemented as in tf version. TODO
        :param img: image
        :param bbox: person bounding box in the image
        :return: patch coordinates
        """
        ymin, xmin, ymax, xmax = bbox
        h, w = ymax - ymin, xmax - xmin

        long_side = max(h, w)

        patch_w = int(long_side * self.scale)
        patch_h = patch_w

        # center patch on bounding box
        orig_y = ymin + h / 2.
        orig_x = xmin + w / 2.

        # ensure patch dimensions are within image
        ymin_patch = max(orig_y - patch_h / 2., 0.)
        xmin_patch = max(orig_x - patch_w / 2., 0.)

        img_h, img_w, _ = img.shape
        if ymin_patch + patch_h > img_h:
            ymin_patch = img_h - patch_h

        if xmin_patch + patch_w > img_w:
            xmin_patch = img_w - patch_w

        return list(map(int, (ymin_patch, xmin_patch, patch_h, patch_w)))

    def rescale(self, image):
        """
        helper function to rescale and add grey values to image to assist in brightness matching that has similar
        results as done in the tf version where the images are in rescaled form and have grey bands
        :param image: image
        :return: rescaled image
        """
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
        """
        brightness match implementation to match intensity on the Y channel of the YUV color scale in both patch and
        target image. Images are rescaled and converted to YUV for intensity matching on Y channel and then converted
        back to RGB. similar to tf implementation
        :param tgt:
        :return:
        """
        tgt = self.rescale(tgt)
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
        """
        random brightness
        :param tgt: target patch
        :param delta: delta intensity
        :return: brightness altered patch
        """
        delta = np.random.uniform(-delta, delta)
        return np.clip(tgt + delta, -1., 1.)

    def random_noise(self, tgt, delta):
        """
        random noise to simulate reimaging of a physical patch
        :param tgt: target patch
        :param delta: delta noise range
        :return: noise added patch
        """
        delta = np.random.uniform(low=-delta, high=delta, size=tgt.shape)
        return np.clip(tgt + delta, -1., 1.)

    def resize(self, patch, patch_h, patch_w):
        """
        rescale patch to area coordinates within the target image
        :param patch: patch
        :param patch_h: target patch height in the image
        :param patch_w: target patch width in the image
        :return: rescale patch
        """
        h, w, _ = patch.shape
        if h > patch_h:
            # use area interpolation when target area is smaller than the original patch
            patch = cv2.resize(patch, (patch_w, patch_h), interpolation=cv2.INTER_AREA)
        elif h < patch_h:
            # use bicubic interpolation when target area is larger than the original patch
            patch = cv2.resize(patch, (patch_w, patch_h), interpolation=cv2.INTER_CUBIC)
        return patch

    def get_transformed_patch(self, img, patch_h, patch_w):
        """
        apply patch transformations
        :param img: image
        :param patch_h: target patch height in the image
        :param patch_w: target patch width in the image
        :return: transformed patch
        """
        patch = self.brightness_match(img)
        patch = self.resize(patch, patch_h, patch_w)
        patch = patch - self.mean_rgb
        patch /= self.stddev_rgb
        patch = self.random_noise(patch, .01)
        patch = self.random_brightness(patch, .3)
        patch *= self.stddev_rgb
        patch += self.mean_rgb
        return np.clip(patch, 0., 255.).astype('uint8')

    def add_adv_to_img(self, img: np.ndarray, bboxes):
        """
        called by outside class that wants to add patches to an image
        :param img: image
        :param bboxes: all the persons in the image as bounding boxes
        :return: patched image
        """
        img = img.copy()
        for bbox in bboxes:
            ymin_patch, xmin_patch, patch_h, patch_w = self._create(img, bbox)
            patch = self.get_transformed_patch(img, patch_h, patch_w)
            img[ymin_patch: ymin_patch + patch_h, xmin_patch: xmin_patch + patch_w] = patch
        return img


def test():
    """test only"""
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
    test()
