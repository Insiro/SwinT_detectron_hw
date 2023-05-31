from typing import TypedDict
import json


class CoCoLabeler:
    def __init__(self, file_path) -> None:
        self.path = file_path
        self.images: list[CoCoImage] = []
        self.categories: list[CoCoCategory] = []
        self.annotations: list[CoCoAnnotation] = []
        self.index = 1
        self.annotIndex = 1

    @classmethod
    def load_saved(cls, file_path):
        labeler = CoCoLabeler(file_path)
        with open(file_path, "r") as f:
            data = json.load(f)
            labeler.images = data["images"]
            labeler.annotations = data["annotations"]
            labeler.categories = data["categories"]
            labeler.index = data["index"]
            labeler.annotIndex = data["annotIndex"]
        return labeler

    def add_category(self, name, id):
        self.categories.append(CoCoCategory(id=id, name=name))

    def save(self):
        with open(self.path, "w") as f:
            json.dump(
                {
                    "images": self.images,
                    "annotations": self.annotations,
                    "categories": self.categories,
                    "index": self.index,
                    "annotIndex": self.annotIndex,
                },
                f,
            )

    def add_image(self, img_path, shape2d):
        img = CoCoImage(
            file_name=img_path,
            height=shape2d[0],
            width=shape2d[1],
            id=self.index,
        )
        self.images.append(img)

        self.index += 1
        return img["id"]

    def add_annotation(self, image_id: int, bbox, category_id, area):
        annotation = CoCoAnnotation(
            id=self.annotIndex,
            image_id=image_id,
            bbox=bbox,
            iscrowd=0,
            category_id=category_id,
            area=area,
            segmentation=[],
        )
        self.annotations.append(annotation)
        self.annotIndex += 1
        return annotation["id"]


class CoCoCategory(TypedDict):
    id: int
    name: str


class CoCoAnnotation(TypedDict):
    id: int
    image_id: int
    bbox: list[float]
    iscrowd: int
    category_id: int
    area: float
    segmentation: list[float]


class CoCoImage(TypedDict):
    id: int
    file_name: str
    height: int
    width: int
