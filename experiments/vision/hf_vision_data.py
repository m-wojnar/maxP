"""HF streaming data pipeline for vision experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset


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


class HFVisionIterableDataset(IterableDataset):
    """Bridge HF streaming iterable datasets into torch IterableDataset."""

    def __init__(
        self,
        hf_stream,
        *,
        preset: DatasetPreset,
        transform: Callable,
    ) -> None:
        super().__init__()
        self._hf_stream = hf_stream
        self._preset = preset
        self._transform = transform

    def __iter__(self):
        for sample in self._hf_stream:
            image = sample.get(self._preset.image_key)
            image = image.convert("RGB")
            label = sample.get(self._preset.label_key)

            x = self._transform(image)
            y = torch.tensor(label, dtype=torch.long)

            yield x, y


def make_loader(
    ds: IterableDataset,
    *,
    batch_size: int,
    num_workers: int,
    is_train: bool,
    prefetch_factor: int,
) -> DataLoader:
    kwargs = {
        "dataset": ds,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": is_train,
        "persistent_workers": (num_workers > 0),
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(
        **kwargs,
    )


def build_dataloaders(
    *,
    dataset_name: str,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int = 2,
    seed: int = 1,
    train_shuffle_buffer: int = 10_000,
    train_transform=None,
    eval_transform=None,
) -> tuple[DataLoader, DataLoader | None, DatasetPreset]:
    if train_transform is None or eval_transform is None:
        raise ValueError(
            "build_dataloaders requires train/eval transforms (pass timm create_transform outputs)."
        )

    preset = DATASET_CONFIGS.get(dataset_name)

    train_stream = load_dataset(preset.dataset_id, split=preset.train_split, streaming=True)
    train_stream = train_stream.shuffle(buffer_size=train_shuffle_buffer, seed=seed)
    train_ds = HFVisionIterableDataset(train_stream, preset=preset, transform=train_transform)
    train_loader = make_loader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        is_train=True,
        prefetch_factor=prefetch_factor,
    )
    
    val_stream = load_dataset(preset.dataset_id, split=preset.val_split, streaming=True)
    val_ds = HFVisionIterableDataset(val_stream, preset=preset, transform=eval_transform)
    val_loader = make_loader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        is_train=False,
        prefetch_factor=prefetch_factor,
    )

    return train_loader, val_loader, preset
