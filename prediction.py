import os
from os import path

import cv2
import matplotlib.image as imgl
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.engine.defaults import DefaultPredictor
from matplotlib import patches
from matplotlib import pyplot as plt

from swint import add_swint_config


def setup_pred(args):
    cfg = get_cfg()
    add_swint_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
    cfg.MODEL.WEIGHTS = args.model

    cfg.freeze()

    default_setup(cfg, args)
    return cfg


def predict(args):
    cfg = setup_pred(args)
    predictor = DefaultPredictor(cfg)
    predicted = []
    for img in args.imgs:
        img = cv2.imread(img)
        outputs = predictor(img)
        predicted.append(outputs)
    for idx, instance in enumerate(predicted):
        instance = outputs["instances"].to("cpu")
        print(instance.get_fields().keys())
        print("-------------------------------")
        print(instance.get_fields()["scores"].tolist())
        print("-------------------------------")
        print(instance.get_fields()["pred_classes"].tolist())
        print("-------------------------------")
        print(instance.get_fields()["pred_boxes"].tensor.tolist())
        display(args.imgs[idx], instance.get_fields()["pred_boxes"].tensor.tolist())


def display(imgPath, annots):
    fig, ax = plt.subplots()
    img = imgl.imread(path.join(imgPath))
    plt.imshow(img)
    # x, y, w, h
    w = 0
    for ann in annots:
        y1 = float(ann[1])
        y2 = float(ann[3])
        x1 = float(ann[0])
        x2 = float(ann[2])
        print(ann)
        w = abs(x2 - x1) - w
        ax.add_patch(
            patches.Rectangle(
                (x1, y1),
                x2,
                y2,
                color="r",
                fill=False
                # facecolor='lightgray',
            )
        )
    plt.show()


dataset_dir = "/media/young/backup/sunghoon/dataset"

if __name__ == "__main__":
    test_img = [
        "dataset1_snack/label/snack000000.jpg",
        "dataset2_snack2/label/05-315.jpg",
        "dataset2_snack2/label/05-449.jpg",
        "dataset2_snack2/label/05-554.jpg",
        "dataset1_snack/label/snack001298.jpg",
        "dataset2_snack2/label/05-337.jpg",
    ]
    test_img = [os.path.join(dataset_dir, item) for item in test_img]
    args = default_argument_parser().parse_args()
    args.imgs = test_img
    args.model = "./output/model_0044999.pth"
    args.config_file = "./configs/SwinT/retinanet_swint_T_FPN_3x_.yaml"
    register_coco_instances(
        "drawing",
        {},
        "/media/young/backup/sunghoon/dataset/coco.json",
        "/media/young/backup/sunghoon/dataset",
    )

    print("Command Line Args:", args)
    launch(
        predict,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
