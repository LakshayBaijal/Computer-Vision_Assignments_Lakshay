from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from plyfile import PlyData
from torch.utils.data import Dataset


@dataclass
class PointCloudSample:
    points: np.ndarray
    label: int
    file_path: Path


class ModelNet10Dataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str,
        num_points: int = 1024,
        use_normals: bool = False,
        augment: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.num_points = num_points
        self.use_normals = use_normals
        self.augment = augment

        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test")

        train_dir = self.root_dir / "train"
        test_dir = self.root_dir / "test"
        if not train_dir.exists() or not test_dir.exists():
            raise FileNotFoundError(f"Expected train/test folders inside {self.root_dir}")

        self.class_names = sorted([item.name for item in train_dir.iterdir() if item.is_dir()])
        self.class_to_idx: Dict[str, int] = {name: idx for idx, name in enumerate(self.class_names)}

        train_files = self._collect_split_files(train_dir)
        test_files = self._collect_split_files(test_dir)

        if split == "test":
            self.samples = test_files
        else:
            rng = np.random.default_rng(seed=42)
            indices = np.arange(len(train_files))
            rng.shuffle(indices)
            val_size = max(1, int(0.1 * len(indices)))
            val_indices = set(indices[:val_size].tolist())
            if split == "val":
                self.samples = [train_files[i] for i in sorted(val_indices)]
            else:
                self.samples = [train_files[i] for i in range(len(train_files)) if i not in val_indices]

        if not self.samples:
            raise RuntimeError(f"No samples found for split={split} in {self.root_dir}")

    def _collect_split_files(self, split_dir: Path) -> List[Tuple[Path, int]]:
        pairs: List[Tuple[Path, int]] = []
        for class_name in self.class_names:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue
            label = self.class_to_idx[class_name]
            for ply_file in sorted(class_dir.glob("*.ply")):
                pairs.append((ply_file, label))
        return pairs

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        file_path, label = self.samples[index]
        points = self._load_points(file_path)
        points = self._sample_points(points)
        points = self._normalize_points(points)

        if self.augment:
            points = self._augment(points)

        points_tensor = torch.from_numpy(points.astype(np.float32)).transpose(0, 1)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return points_tensor, label_tensor

    def _load_points(self, file_path: Path) -> np.ndarray:
        ply_data = PlyData.read(str(file_path))
        vertex_data = ply_data["vertex"].data
        names = vertex_data.dtype.names

        xyz = np.stack([vertex_data["x"], vertex_data["y"], vertex_data["z"]], axis=1)
        if self.use_normals and all(n in names for n in ("nx", "ny", "nz")):
            normals = np.stack([vertex_data["nx"], vertex_data["ny"], vertex_data["nz"]], axis=1)
            points = np.concatenate([xyz, normals], axis=1)
        else:
            points = xyz
        return points.astype(np.float32)

    def _sample_points(self, points: np.ndarray) -> np.ndarray:
        total_points = points.shape[0]
        if total_points == self.num_points:
            return points

        if total_points > self.num_points:
            indices = np.random.choice(total_points, self.num_points, replace=False)
        else:
            indices = np.random.choice(total_points, self.num_points, replace=True)
        return points[indices]

    def _normalize_points(self, points: np.ndarray) -> np.ndarray:
        xyz = points[:, :3]
        xyz = xyz - np.mean(xyz, axis=0, keepdims=True)
        scale = np.max(np.linalg.norm(xyz, axis=1))
        if scale > 0:
            xyz = xyz / scale
        points[:, :3] = xyz
        return points

    def _augment(self, points: np.ndarray) -> np.ndarray:
        xyz = points[:, :3]

        jitter = np.random.normal(loc=0.0, scale=0.01, size=xyz.shape).astype(np.float32)
        xyz = xyz + np.clip(jitter, -0.02, 0.02)

        scale = np.random.uniform(0.8, 1.2)
        xyz = xyz * scale

        theta = np.random.uniform(-0.2, 0.2)
        rotation = np.array(
            [
                [np.cos(theta), 0.0, np.sin(theta)],
                [0.0, 1.0, 0.0],
                [-np.sin(theta), 0.0, np.cos(theta)],
            ],
            dtype=np.float32,
        )
        xyz = xyz @ rotation.T

        points[:, :3] = xyz
        return points


def build_datasets(root_dir: str, num_points: int, use_normals: bool):
    train_ds = ModelNet10Dataset(root_dir, split="train", num_points=num_points, use_normals=use_normals, augment=True)
    val_ds = ModelNet10Dataset(root_dir, split="val", num_points=num_points, use_normals=use_normals, augment=False)
    test_ds = ModelNet10Dataset(root_dir, split="test", num_points=num_points, use_normals=use_normals, augment=False)
    return train_ds, val_ds, test_ds
