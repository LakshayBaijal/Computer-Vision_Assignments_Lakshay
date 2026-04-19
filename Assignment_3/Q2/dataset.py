import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEPTH_EXTS = IMAGE_EXTS.union({".npy"})


class MultiTaskDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str,
        image_size: int = 256,
        val_ratio: float = 0.25,
        seed: int = 42,
        augment: bool = False,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.val_ratio = val_ratio
        self.seed = seed
        self.augment = augment

        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split: {split}")

        self.samples = self._build_samples()
        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found for split={split} under {self.root_dir}")

    def _find_dir(self, candidates: List[Path]) -> Path:
        for candidate in candidates:
            if candidate.exists() and candidate.is_dir():
                return candidate
        return None

    def _list_files(self, folder: Path, valid_exts: set) -> List[Path]:
        if folder is None:
            return []
        files = [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in valid_exts]
        return sorted(files)

    def _match_by_stem(self, image_path: Path, target_files: List[Path]) -> Path:
        image_stem = image_path.stem
        exact = [path for path in target_files if path.stem == image_stem]
        if exact:
            return exact[0]

        relaxed = [path for path in target_files if image_stem in path.stem or path.stem in image_stem]
        if relaxed:
            return relaxed[0]

        raise FileNotFoundError(f"Could not match {image_path.name} in targets")

    def _collect_split_files(self, split: str) -> Tuple[List[Path], List[Path], List[Path]]:
        if split in {"train", "val"}:
            image_dir = self._find_dir(
                [
                    self.root_dir / "train" / "images",
                    self.root_dir / "train" / "imgs",
                    self.root_dir / "images",
                    self.root_dir / "img",
                ]
            )
            mask_dir = self._find_dir(
                [
                    self.root_dir / "train" / "labels",
                    self.root_dir / "train" / "masks",
                    self.root_dir / "train" / "segmentation",
                    self.root_dir / "labels",
                    self.root_dir / "masks",
                    self.root_dir / "mask",
                    self.root_dir / "segmentation",
                ]
            )
            depth_dir = self._find_dir(
                [
                    self.root_dir / "train" / "depth",
                    self.root_dir / "depth",
                    self.root_dir / "depths",
                ]
            )
        else:
            image_dir = self._find_dir(
                [
                    self.root_dir / "test" / "images",
                    self.root_dir / "test" / "imgs",
                    self.root_dir / "images_test",
                    self.root_dir / "test_images",
                ]
            )
            mask_dir = self._find_dir(
                [
                    self.root_dir / "test" / "labels",
                    self.root_dir / "test" / "masks",
                    self.root_dir / "test" / "segmentation",
                    self.root_dir / "labels_test",
                    self.root_dir / "masks_test",
                    self.root_dir / "test_masks",
                ]
            )
            depth_dir = self._find_dir(
                [
                    self.root_dir / "test" / "depth",
                    self.root_dir / "depth_test",
                    self.root_dir / "test_depth",
                ]
            )

        images = self._list_files(image_dir, IMAGE_EXTS)
        masks = self._list_files(mask_dir, IMAGE_EXTS)
        depths = self._list_files(depth_dir, DEPTH_EXTS)
        return images, masks, depths

    def _build_samples(self) -> List[Dict[str, Path]]:
        if self.split == "test":
            images, masks, depths = self._collect_split_files("test")
            samples = []
            for image_path in images:
                try:
                    mask_path = self._match_by_stem(image_path, masks)
                    depth_path = self._match_by_stem(image_path, depths)
                except FileNotFoundError:
                    continue
                samples.append({"image": image_path, "mask": mask_path, "depth": depth_path})
            return samples

        images, masks, depths = self._collect_split_files("train")
        all_samples = []
        for image_path in images:
            try:
                mask_path = self._match_by_stem(image_path, masks)
                depth_path = self._match_by_stem(image_path, depths)
            except FileNotFoundError:
                continue
            all_samples.append({"image": image_path, "mask": mask_path, "depth": depth_path})

        rng = random.Random(self.seed)
        rng.shuffle(all_samples)

        split_index = int((1.0 - self.val_ratio) * len(all_samples))
        split_index = max(1, min(split_index, len(all_samples) - 1))

        if self.split == "train":
            return all_samples[:split_index]
        return all_samples[split_index:]

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: Path) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        tensor = TF.to_tensor(image)
        return tensor

    def _load_mask(self, path: Path) -> torch.Tensor:
        mask = Image.open(path).convert("L")
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        mask_np = np.array(mask, dtype=np.int64)
        return torch.from_numpy(mask_np)

    def _load_depth(self, path: Path) -> torch.Tensor:
        if path.suffix.lower() == ".npy":
            depth_np = np.load(path).astype(np.float32)
            if depth_np.ndim == 3:
                depth_np = depth_np[..., 0]
        else:
            depth_img = Image.open(path).convert("F")
            depth_img = depth_img.resize((self.image_size, self.image_size), Image.BILINEAR)
            depth_np = np.array(depth_img, dtype=np.float32)

        if depth_np.max() > 1.0:
            depth_np = depth_np / 255.0

        depth_np = np.clip(depth_np, 0.0, 1.0)

        depth = torch.from_numpy(depth_np).unsqueeze(0)
        return depth

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = self._load_image(sample["image"])
        mask = self._load_mask(sample["mask"])
        depth = self._load_depth(sample["depth"])

        if self.augment and random.random() < 0.5:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[1])
            depth = torch.flip(depth, dims=[2])

        return {
            "image": image,
            "mask": mask,
            "depth": depth,
            "image_path": str(sample["image"]),
        }
