"""
©2022 Technology Innovation Institute. All rights reserved.
Author(s): saurabh.pathak@tii.ae
Created: 31 March, 2022

Purpose: Adversarial patch creation based on bounding box
"""
import math

import cv2
import numpy as np
from PIL import Image


def rotate_point(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    :param angle: Angle in radians. Positive angle is counterclockwise.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


def rotate_annotation(origin, annotation, degree):
    """
    Rotates an annotation's bounding box by `degree` counterclockwise about `origin`.
    Assumes cropping from center to preserve image dimensions.
    :param origin: down is positive
    :param annotation: dict
    :param degree: degrees by which to rotate (positive is counterclockwise)
    :return: annotation after rotation
    """
    # Don't mutate annotation
    # new_annotation = copy.deepcopy(annotation)
    new_annotation = annotation

    angle = math.radians(degree)
    origin_x, origin_y = origin
    origin_y *= -1

    x = annotation["x"]
    y = annotation["y"]

    new_x, new_y = map(lambda x: round(x * 2) / 2, rotate_point(
        (origin_x, origin_y), (x, -y), angle)
                       )

    new_annotation["x"] = new_x
    new_annotation["y"] = -new_y

    width = annotation["width"]
    height = annotation["height"]

    left_x = x - width / 2
    right_x = x + width / 2
    top_y = y - height / 2
    bottom_y = y + height / 2

    c1 = (left_x, top_y)
    c2 = (right_x, top_y)
    c3 = (right_x, bottom_y)
    c4 = (left_x, bottom_y)

    c1 = rotate_point(origin, c1, angle)
    c2 = rotate_point(origin, c2, angle)
    c3 = rotate_point(origin, c3, angle)
    c4 = rotate_point(origin, c4, angle)

    x_coords, y_coords = zip(c1, c2, c3, c4)
    new_annotation["width"] = round(max(x_coords) - min(x_coords))
    new_annotation["height"] = round(max(y_coords) - min(y_coords))

    return new_annotation


class AdversarialPatch:

    def __init__(self, h, w):
        self._patch_img = (np.random.rand(h, w, 3) * 255).astype('uint8')

    @staticmethod
    def create(img, bbox, *, aspect=1., scale=.5):
        ymin, xmin, h, w = bbox

        patch_w = h * scale
        patch_h = aspect * patch_w

        orig_y = ymin + h / 2.
        orig_x = xmin + w / 2.

        ymin_patch = max(orig_y - patch_h / 2., 0.)
        xmin_patch = max(orig_x - patch_w / 2., 0.)

        img_h, img_w, _ = img.shape
        if ymin_patch + patch_h > img_h:
            ymin_patch = img_h - patch_h

        if xmin_patch + patch_w > img_w:
            ymin_patch = img_w - patch_w

        return list(map(int, (ymin_patch, xmin_patch, patch_h, patch_w)))

    def add_adv_to_img(self, img: np.ndarray, bbox):
        ymin_patch, xmin_patch, patch_h, patch_w = self.create(img, bbox)
        patch = cv2.resize(self._patch_img, (patch_w, patch_h))

        img[ymin_patch: ymin_patch + patch_h, xmin_patch: xmin_patch + patch_w] = patch
        return img


def main():
    import matplotlib.pyplot as plt
    from matplotlib import patches

    im = np.asarray(Image.open('burj_khalifa_sunset.jpg'))
    adv_patch = AdversarialPatch(100, 100)
    bbox = ymin, xmin, h, w = 50, 125, 400, 200
    im = adv_patch.add_adv_to_img(im, bbox)
    plt.imshow(im)
    rect = patches.Rectangle((xmin, ymin), w, h, linewidth=1, edgecolor='r', facecolor='none')
    plt.gca().add_patch(rect)
    plt.show()


if __name__ == '__main__':
    main()
