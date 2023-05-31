import os
import platform
from os import path

import cv2
from tqdm import tqdm

from coco import CoCoLabeler

bins = [
    # "dataset1_snack",
    "dataset2_snack2",
    # "dataset3_sauce",
    # "dataset4_processed1",
    # # "dataset5_processed2", # label missing
    # "dataset6_myeon",
    # "dataset7_can",
]
out_bin = "merged_dataset2"


class PathLoader:
    def __init__(self) -> None:
        self.system = platform.system()

    def convert(self, dir: str, force=True):
        if force or self.system == "Linux":
            dir = dir.replace("\\", "/")
            dirs = dir.split(":/")
            dir = f"/mnt/{dirs[0].lower}/{dirs[1]}" if len(dirs) == 2 else dir
            return dir
        return dir


pathloader = PathLoader()


class DataSetMerger:
    def __init__(self, saved_coco_path=None) -> None:
        self.labels: dict[str, int] = {}
        self.len_labels = 0
        if saved_coco_path is not None:
            self.load_saved_coco(saved_coco_path)
        else:
            self.coco: CoCoLabeler = CoCoLabeler("coco2.json")
        pass

    def load_saved_labels(self, folder):
        classes = []
        labels = {}
        with open(path.join(folder, "obj.names"), "r", encoding="UTF8") as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                class_name = line.strip()
                if len(class_name) == 0:
                    continue
                classes.append(class_name)
                labels[class_name] = idx
        len_labels = len(classes)
        return classes, labels, len_labels

    def load_saved_coco(self, file_path):
        self.coco = CoCoLabeler.load_saved(file_path)
        for category in self.coco.categories:
            self.labels[category.name] = category.id
        self.len_labels = len(self.labels)

    def merge_dataset(self, src):
        src_folder = path.join(src, "label")
        # load labels
        classes, labels, _ = self.load_saved_labels(src_folder)

        # copy files
        for file in tqdm(os.listdir(src_folder), desc=f"merging {src}"):
            if file.endswith(".jpg"):
                img_path = path.join(src_folder, file)
                label_file = file.replace(".jpg", ".txt")

                origin = path.join(src_folder, label_file)
                if not os.path.exists(origin):
                    continue

                # add image to coco
                shape = cv2.imread(path.join(src_folder, file)).shape[:2]
                img_id = self.coco.add_image(pathloader.convert(img_path), shape)

                # convert labels
                self.convert_label(img_id, origin, classes, shape)

    @staticmethod
    def yolo_to_coco(yolo_bbox, shape):
        height, width = shape
        x, y, w, h = yolo_bbox
        x = x * width
        y = y * height

        w = w * width
        h = h * height
        min_x = int(x - w / 2)
        min_y = int(y - h / 2)
        return [min_x, min_y, w, h]

    def convert_label(self, img_id, src, classes, shape):
        with open(src, "r", encoding="UTF8") as f:
            lines = f.readlines()
            for line in lines:
                class_idx, *rest = line.split(" ")
                class_name = classes[int(class_idx)]

                if class_name not in self.labels:
                    self.labels[class_name] = self.len_labels
                    self.coco.add_category(class_name, self.len_labels)
                    self.len_labels += 1

                bbox = self.yolo_to_coco([float(item) for item in rest], shape)
                self.coco.add_annotation(img_id, bbox, self.labels[class_name], 0)

    def merge_all(self, bins):
        for bin in bins:
            self.merge_dataset(bin)
        self.save_labels()

    def save_labels(self):
        self.coco.save()


# DataSetMerger().merge_all(bins)

annots = [
    {
        "id": 1,
        "image_id": 1,
        "bbox": [910, 157, 1147, 479],
        "iscrowd": 0,
        "category_id": 0,
        "area": 0,
        "segmentation": [],
    },
    {
        "id": 2,
        "image_id": 1,
        "bbox": [972, 37, 1238, 122],
        "iscrowd": 0,
        "category_id": 0,
        "area": 0,
        "segmentation": [],
    },
    {
        "id": 3,
        "image_id": 1,
        "bbox": [545, 102, 791, 556],
        "iscrowd": 0,
        "category_id": 0,
        "area": 0,
        "segmentation": [],
    },
    {
        "id": 4,
        "image_id": 1,
        "bbox": [108, 26, 428, 351],
        "iscrowd": 0,
        "category_id": 0,
        "area": 0,
        "segmentation": [],
    },
    {
        "id": 5,
        "image_id": 1,
        "bbox": [7, 46, 79, 474],
        "iscrowd": 0,
        "category_id": 0,
        "area": 0,
        "segmentation": [],
    },
]
im_path = "./dataset2_snack2/label/05-315.jpg"


# def visualization_test():
#     from matplotlib import pyplot as plt

#     img = cv2.imread(im_path)
#     for annot in annots:
#         bbox = annot["bbox"]
#         cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 3)

#     plt.imshow(img)

#     plt.show()


# visualization_test()

DataSetMerger().merge_all(bins)
