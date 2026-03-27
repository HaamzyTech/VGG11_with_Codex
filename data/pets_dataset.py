"""Oxford-IIIT Pet dataset loader for classification."""

from pathlib import Path
from typing import Callable, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset


class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet classification dataset loader.

    This loader reads official split files from ``annotations``:
    - ``trainval.txt`` for training/validation
    - ``test.txt`` for test

    Labels are converted from 1..37 (official) to 0..36.
    """

    def __init__(
        self,
        root: str,
        split: str = "trainval",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        self.root = Path(root)
        self.images_dir = self.root / "images"
        self.annotations_dir = self.root / "annotations"
        self.transform = transform
        self.target_transform = target_transform

        if split not in {"trainval", "test"}:
            raise ValueError(f"Unsupported split: {split}. Use 'trainval' or 'test'.")

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
                target = int(class_id) - 1
                self.samples.append((image_path, target))

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found in split file: {split_file}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[object, int]:
        image_path, target = self.samples[index]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target
