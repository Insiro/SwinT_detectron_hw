import json
import os
from itertools import cycle
from os import path
from typing import Any, Dict

import matplotlib.image as imgl
import matplotlib.pyplot as plt
from matplotlib import patches
from tqdm import tqdm

ROOT = os.getcwd()
COCO_PATH = path.join(ROOT, "/media/young/backup/sunghoon/dataset/coco.json")
IMG_PATH = path.join(ROOT, "/media/young/backup/sunghoon/dataset")
OUTPUT = path.join(ROOT, "./output/inference/coco_instances_results.json")


def load():
    with open(OUTPUT, "r") as pred:
        predicted = json.load(pred)
    with open(COCO_PATH, "r") as cocos:
        coco: Dict[str, Any] = json.load(cocos)
    imgs = coco["images"]
    categories = coco["categories"]
    cycol = cycle("bgrcmk")
    categories = [
        (categories[index], next(cycol)) for index in range(categories.__len__())
    ]
    anns = coco["annotations"]
    return imgs, anns, categories, cycol


def annotate(index, imgs, anns, categories):
    fig, ax = plt.subplots()
    img = imgl.imread(path.join(IMG_PATH, imgs[index]["file_name"]))
    plt.imshow(img)
    # x, y, w, h

    lend: list[str] = []

    for ann in anns:
        if int(ann["image_id"]) != int(imgs[index]["id"]):
            continue
        ids = [
            cate for cate in categories if int(cate[0]["id"]) == int(ann["category_id"])
        ][0]
        ax.add_patch(
            patches.Rectangle(
                (float(ann["bbox"][0]), float(ann["bbox"][1])),
                float(ann["bbox"][2]) - float(ann["bbox"][0]),
                float(ann["bbox"][3]) - float(ann["bbox"][1]),
                color=ids[1],
                fill=False
                # facecolor='lightgray',
            )
        )
        if not (ids[0]["name"] in lend):
            lend.append(ids[0]["name"])
    plt.legend(lend, bbox_to_anchor=(1.05, 1.0), loc="upper left")


if __name__ == "__main__":
    imgs, anns, categories, cycol = load()
    for i in tqdm(range(imgs.__len__())):
        annotate(i, imgs, anns, categories)
        im_path = imgs[i]["file_name"]
        dirs = im_path.split("/")

        plt.savefig(f"./predicted/{dirs[-2]}_{dirs[-1]}")
