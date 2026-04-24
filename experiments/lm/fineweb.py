"""Register FineWeb-Edu in torchtitan's dataset registry.

Import this module before building any Trainer.Config that uses
--dataset fineweb-edu or --dataset fineweb-edu-10bt or --dataset fineweb-edu-100bt.

FineWeb-Edu is loaded from the local HF cache (HF_HOME must contain 
the downloaded dataset). No pre-tokenization step is needed.
torchtitan tokenizes on-the-fly with the LLaMA-3 tokenizer.
"""

from __future__ import annotations

from datasets import load_dataset

from torchtitan.hf_datasets import DatasetConfig
from torchtitan.hf_datasets.text_datasets import DATASETS


def _process_fineweb_text(sample: dict) -> str:
    return sample["text"]


def _load_fineweb_edu(dataset_path: str):
    return load_dataset(dataset_path, split="train", streaming=True)


def _load_fineweb_edu_10bt(dataset_path: str):
    # 10B-token deduplicated subset for S1/S2/S3
    return load_dataset(dataset_path, name="sample-10BT", split="train", streaming=True)


def _load_fineweb_edu_100bt(dataset_path: str):
    # 100B-token deduplicated subset for S4/S5
    return load_dataset(dataset_path, name="sample-100BT", split="train", streaming=True)


DATASETS["fineweb-edu"] = DatasetConfig(
    path="HuggingFaceFW/fineweb-edu",
    loader=_load_fineweb_edu,
    sample_processor=_process_fineweb_text,
)

DATASETS["fineweb-edu-10bt"] = DatasetConfig(
    path="HuggingFaceFW/fineweb-edu",
    loader=_load_fineweb_edu_10bt,
    sample_processor=_process_fineweb_text,
)

DATASETS["fineweb-edu-100bt"] = DatasetConfig(
    path="HuggingFaceFW/fineweb-edu",
    loader=_load_fineweb_edu_100bt,
    sample_processor=_process_fineweb_text,
)
