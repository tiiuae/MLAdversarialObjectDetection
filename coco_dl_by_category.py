"""
©2022 Technology Innovation Institute. All rights reserved.
Author(s): saurabh.pathak@tii.ae
Created: April 01, 2022

Purpose: async parallel download image files and annotations from COCO containing only a given category (eg. persons)
and save labels in [ymin, xmin, ymax, xmax] txt format
"""
import asyncio
import os

import aiofiles as aiofiles
import aiohttp as aiohttp
from pycocotools.coco import COCO


def truncate(n, decimals=0):
    """Truncates numbers to N decimals"""
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier


async def get_image(session, im_def, filename):
    async with session.get(im_def['coco_url']) as resp:
        if resp.status == 200:
            async with aiofiles.open(filename, mode='wb') as f:
                await f.write(await resp.read())
            print('downloaded', filename)
        else:
            print('failed to write to', filename)


async def main():
    """Download instances_train2017.json from the COCO website and put in the same directory as this script"""
    coco = COCO('instances_val2017.json')
    # cats = coco.loadCats(coco.getCatIds())
    # nms = [cat['name'] for cat in cats]
    # print('COCO categories: \n{}\n'.format(' '.join(nms)))

    # We are interested in persons
    cat = "person"
    cat_ids = coco.getCatIds(catNms=[cat])
    img_ids = coco.getImgIds(catIds=cat_ids)
    images = coco.loadImgs(img_ids)
    download_dir = 'downloaded_images_val'
    labels_dir = 'labels_val'

    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Comment this entire section out if you don't want to download the images
    filenames = [os.path.join(download_dir, im['file_name']) for im in images]
    dl_images = [im for im, filename in zip(images, filenames)
                 if not os.path.isfile(filename) or not os.path.getsize(filename)]
    print('will download', len(dl_images), 'images')
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=50)) as session:
        await asyncio.gather(*[get_image(session, im, os.path.join(download_dir, im['file_name']))
                               for im in dl_images])

    for im in images:
        ann_ids = coco.getAnnIds(imgIds=im['id'], catIds=cat_ids, iscrowd=None)
        anns = coco.loadAnns(ann_ids)

        filename = os.path.join(labels_dir, im['file_name'].replace(".jpg", ".txt"))
        if os.path.isfile(filename):
            continue

        print(filename)
        with open(filename, "a") as myfile:
            for i in range(len(anns)):
                xmin = anns[i]["bbox"][0]
                ymin = anns[i]["bbox"][1]
                xmax = anns[i]["bbox"][2] + anns[i]["bbox"][0]
                ymax = anns[i]["bbox"][3] + anns[i]["bbox"][1]

                # Note: This forms a single-category dataset, and thus the "0" at the beginning of each line.
                myfile.write(f"0 {ymin} {xmin} {ymax} {xmax}")
                myfile.write("\n")

        myfile.close()


if __name__ == '__main__':
    asyncio.run(main())
