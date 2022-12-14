"""
©2022 Technology Innovation Institute. All rights reserved.
Author(s): saurabh.pathak@tii.ae
Created: June 17, 2022

Purpose: filter visDrone by car and van
"""
import os
import pathlib
import shutil

from PIL import Image


def _parse_line(line):
    return list(map(int, line.strip().split(',')[:6]))


def main():
    root_dir = pathlib.Path('../data/VisDrone2019-DET-train')
    src_dir = root_dir.joinpath('images')
    src_labels_dir = root_dir.joinpath('annotations')

    tgt_dir = root_dir.joinpath('images_filt')
    tgt_labels_dir = root_dir.joinpath('annotations_filt')

    os.makedirs(tgt_dir)
    os.makedirs(tgt_labels_dir)

    for filename in os.listdir(src_labels_dir):
        txt_filename = src_labels_dir.joinpath(filename)
        with open(txt_filename) as f:
            write_file = None
            for line in f.readlines():
                xmin, ymin, bw, bh, _, cat = _parse_line(line)
                if cat in {4, 5}:
                    if not write_file:
                        write_filename = tgt_labels_dir.joinpath(filename)
                        write_file = open(write_filename, 'w')

                        prefix = os.path.splitext(filename)[0] + '.jpg'
                        image_filename = src_dir.joinpath(prefix)
                        shutil.copyfile(image_filename, tgt_dir.joinpath(prefix))
                        im = Image.open(image_filename)
                        w, h = im.size

                    x, y = xmin + bw // 2, ymin + bh // 2
                    write_file.writelines(f'0 {x/w} {y/h} {bw/w} {bh/h}\n')
            if write_file:
                write_file.close()


if __name__ == '__main__':
    main()
