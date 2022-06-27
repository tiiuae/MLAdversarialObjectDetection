"""
©2022 Technology Innovation Institute. All rights reserved.
Author(s): saurabh.pathak@tii.ae
Created: June 17, 2022

Purpose: filter visDrone by person
"""
import os
import pathlib
import shutil


def _parse_line(line):
    return list(map(int, line.strip().split(',')[:6]))


def main():
    root_dir = pathlib.Path('VisDrone2019-DET-train')
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
                xmin, ymin, xmax, ymax, _, cat = _parse_line(line)
                if cat in {1, 2}:
                    if not write_file:
                        write_filename = tgt_labels_dir.joinpath(filename)
                        write_file = open(write_filename, 'w')
                    write_file.writelines(f'{ymin},{xmin},{ymax},{xmax},{cat}\n')
            if write_file:
                write_file.close()
        if write_file:
            prefix = os.path.splitext(filename)[0] + '.jpg'
            image_filename = src_dir.joinpath(prefix)
            shutil.copyfile(image_filename, tgt_dir.joinpath(prefix))


if __name__ == '__main__':
    main()