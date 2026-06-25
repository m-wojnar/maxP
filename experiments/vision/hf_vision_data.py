"""HF non-streaming data pipeline for vision experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True, slots=True)
class DatasetPreset:
    dataset_id: str
    train_split: str
    val_split: str | None
    num_classes: int
    train_samples: int
    val_samples: int | None
    image_key: str
    label_key: str


DATASET_CONFIGS: dict[str, DatasetPreset] = {
    "imagenet12k": DatasetPreset(
        dataset_id="timm/imagenet-12k-wds",
        train_split="train",
        val_split="validation",
        num_classes=11821,
        train_samples=12_129_687,
        val_samples=472_840,
        image_key="jpg",
        label_key="cls",
    ),
    "beans": DatasetPreset(
        dataset_id="AI-Lab-Makerere/beans",
        train_split="train",
        val_split="validation",
        num_classes=3,
        train_samples=1034,
        val_samples=133,
        image_key="image",
        label_key="labels",
    ),
}


class HFVisionDataset(Dataset):
    """Map-style wrapper around a downloaded HF Dataset."""

    def __init__(self, hf_dataset, *, preset: DatasetPreset, transform: Callable) -> None:
        super().__init__()
        self._hf = hf_dataset
        self._preset = preset
        self._transform = transform

    def __len__(self) -> int:
        return len(self._hf)

    def __getitem__(self, idx: int):
        sample = self._hf[idx]
        image = sample[self._preset.image_key].convert("RGB")
        label = sample[self._preset.label_key]
        x = self._transform(image)
        y = torch.tensor(label, dtype=torch.long)
        return x, y


def make_loader(
    ds: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    is_train: bool,
    prefetch_factor: int,
    generator: torch.Generator | None = None,
) -> DataLoader:
    kwargs: dict = {
        "dataset": ds,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "shuffle": is_train,
        "drop_last": is_train,
        "persistent_workers": (num_workers > 0),
    }
    if is_train and generator is not None:
        kwargs["generator"] = generator
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**kwargs)


def build_dataloaders(
    *,
    dataset_name: str,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int = 2,
    train_transform=None,
    eval_transform=None,
    generator: torch.Generator | None = None,
) -> tuple[DataLoader, DataLoader | None, DatasetPreset]:
    if train_transform is None or eval_transform is None:
        raise ValueError(
            "build_dataloaders requires train/eval transforms (pass timm create_transform outputs)."
        )

    preset = DATASET_CONFIGS.get(dataset_name)

    train_hf = load_dataset(preset.dataset_id, split=preset.train_split, streaming=False)
    train_ds = HFVisionDataset(train_hf, preset=preset, transform=train_transform)
    train_loader = make_loader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        is_train=True,
        prefetch_factor=prefetch_factor,
        generator=generator,
    )

    val_hf = load_dataset(preset.dataset_id, split=preset.val_split, streaming=False)
    val_ds = HFVisionDataset(val_hf, preset=preset, transform=eval_transform)
    val_loader = make_loader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        is_train=False,
        prefetch_factor=prefetch_factor,
    )

    return train_loader, val_loader, preset
