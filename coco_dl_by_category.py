"""
©2022 Technology Innovation Institute. All rights reserved.
Author(s): saurabh.pathak@tii.ae
Created: April 01, 2022

Purpose: async parallel download files and annotations from COCO containing only a given category (eg. persons)
and save in YOLOv3 txt format
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
    coco = COCO('instances_train2017.json')
    # cats = coco.loadCats(coco.getCatIds())
    # nms = [cat['name'] for cat in cats]
    # print('COCO categories: \n{}\n'.format(' '.join(nms)))

    # We are interested in persons
    cat = "person"
    cat_ids = coco.getCatIds(catNms=[cat])
    img_ids = coco.getImgIds(catIds=cat_ids)
    images = coco.loadImgs(img_ids)

    os.makedirs('downloaded_images', exist_ok=True)
    os.makedirs('labels', exist_ok=True)

    # Comment this entire section out if you don't want to download the images
    filenames = [os.path.join('downloaded_images', im['file_name']) for im in images]
    dl_images = [im for im, filename in zip(images, filenames)
                 if not os.path.isfile(filename) or not os.path.getsize(filename)]
    print('will download', len(dl_images), 'images')
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=50)) as session:
        await asyncio.gather(*[get_image(session, im, os.path.join('downloaded_images', im['file_name']))
                               for im in dl_images])

    for im in images:
        dw = 1. / im['width']
        dh = 1. / im['height']

        ann_ids = coco.getAnnIds(imgIds=im['id'], catIds=cat_ids, iscrowd=None)
        anns = coco.loadAnns(ann_ids)

        filename = os.path.join('labels', im['file_name'].replace(".jpg", ".txt"))
        if os.path.isfile(filename):
            continue

        print(filename)
        with open(filename, "a") as myfile:
            for i in range(len(anns)):
                xmin = anns[i]["bbox"][0]
                ymin = anns[i]["bbox"][1]
                xmax = anns[i]["bbox"][2] + anns[i]["bbox"][0]
                ymax = anns[i]["bbox"][3] + anns[i]["bbox"][1]

                x = (xmin + xmax)/2
                y = (ymin + ymax)/2

                w = xmax - xmin
                h = ymax-ymin

                x = x * dw
                w = w * dw
                y = y * dh
                h = h * dh

                # Note: This assumes a single-category dataset, and thus the "0" at the beginning of each line.
                mystring = str("0 " + str(truncate(x, 7)) +
                               " " + str(truncate(y, 7)) +
                               " " + str(truncate(w, 7)) +
                               " " + str(truncate(h, 7)))
                myfile.write(mystring)
                myfile.write("\n")

        myfile.close()


if __name__ == '__main__':
    asyncio.run(main())
