import os
import platform
from os import path

import cv2
from coco import CoCoLabeler
from tqdm import tqdm



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


if __name__ == "__main__":
    bins = [
    "dataset1_snack",
    "dataset2_snack2",
    ]

    DataSetMerger().merge_all(bins)
