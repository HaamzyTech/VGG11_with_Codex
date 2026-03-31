"""Oxford-IIIT Pet dataset loader for classification/localization/segmentation."""

from pathlib import Path
from typing import Callable, Optional, Tuple
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet dataset loader.

    This loader reads official split files from ``annotations``:
    - ``trainval.txt`` for training/validation
    - ``test.txt`` for test

    Supported targets:
    - ``classification``: returns breed label in [0, 36]
    - ``localization``: returns bbox [x_center, y_center, width, height] normalized to [0, 1]
    - ``segmentation``: returns trimap mask in class ids [0, 1, 2]
    """

    def __init__(
        self,
        root: str,
        split: str = "trainval",
        target_type: str = "classification",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        self.root = Path(root)
        self.images_dir = self.root / "images"
        self.annotations_dir = self.root / "annotations"
        self.xml_dir = self.annotations_dir / "xmls"
        self.trimaps_dir = self.annotations_dir / "trimaps"
        self.transform = transform
        self.target_transform = target_transform
        self.target_type = target_type

        if split not in {"trainval", "test"}:
            raise ValueError(f"Unsupported split: {split}. Use 'trainval' or 'test'.")
        if self.target_type not in {"classification", "localization", "segmentation"}:
            raise ValueError("target_type must be one of {'classification', 'localization', 'segmentation'}")

        split_file = self.annotations_dir / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(
                f"Could not find split file: {split_file}. "
                "Download and extract the Oxford-IIIT Pet dataset first."
            )

        self.samples = []
        with split_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                name, class_id, *_ = line.split()
                image_path = self.images_dir / f"{name}.jpg"
                label = int(class_id) - 1

                if self.target_type == "classification":
                    target = label
                elif self.target_type == "localization":
                    target = self._load_normalized_bbox(name)
                else:
                    target = self._load_trimap(name)

                self.samples.append((image_path, target))

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found in split file: {split_file}")

    def _load_normalized_bbox(self, image_id: str):
        xml_path = self.xml_dir / f"{image_id}.xml"
        if not xml_path.exists():
            raise FileNotFoundError(f"Missing XML annotation for {image_id}: {xml_path}")

        root = ET.parse(xml_path).getroot()
        size = root.find("size")
        width = float(size.find("width").text)
        height = float(size.find("height").text)

        bnd = root.find("object").find("bndbox")
        xmin = float(bnd.find("xmin").text)
        ymin = float(bnd.find("ymin").text)
        xmax = float(bnd.find("xmax").text)
        ymax = float(bnd.find("ymax").text)

        box_w = max(xmax - xmin, 1.0)
        box_h = max(ymax - ymin, 1.0)
        x_center = xmin + box_w / 2.0
        y_center = ymin + box_h / 2.0

        return [x_center / width, y_center / height, box_w / width, box_h / height]

    def _load_trimap(self, image_id: str):
        mask_path = self.trimaps_dir / f"{image_id}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing trimap annotation for {image_id}: {mask_path}")
        trimap = np.array(Image.open(mask_path), dtype=np.int64)
        # Dataset trimap labels are {1,2,3}; convert to {0,1,2}.
        return trimap - 1

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[object, object]:
        image_path, target = self.samples[index]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target
