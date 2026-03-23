# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the OneThinker-train-data dataset to parquet format for verl GRPO training.

Dataset: https://huggingface.co/datasets/OneThink/OneThinker-train-data
Paper:   OneThinker: All-in-one Reasoning Model for Image and Video (CVPR 2026)

Dataset files:
  - onethinker_rl_train.json           : RL training split (~60k samples, for GRPO)
  - onethinker_rl_train_unsampled.json : full unsampled RL data
  - onethinker_sft_image.json          : image SFT cold-start data
  - onethinker_sft_video.json          : video SFT cold-start data

Original JSON record schema:
  {
    "problem_id":            int,
    "problem":               str,   # question text, may contain "<image>" or "<video>" placeholder
    "data_type":             str,   # "image" | "video"
    "problem_type":          str,   # "math" | "multiple choice" | "open-ended" | "segmentation" | ...
    "options":               list,  # answer options for multiple-choice; empty list otherwise
    "data_source":           str,   # originating dataset name, e.g. "ARES", "NeXT-QA/30_60_s_nextqa"
    "answer":                str,   # wrapped as "<answer>...</answer>"
    "images":                list,  # list of relative image paths (only for image samples)
    "videos":                list,  # list of relative video paths (only for video samples)
    "problem_reserved_text": str    # original problem text without the modality placeholder
  }

verl parquet schema (output):
  - data_source  : str
  - prompt       : list[dict]  — OpenAI chat messages format, content contains <image>/<video> placeholder
  - images       : list        — PIL Images (image samples only; absent for video samples)
  - videos       : list        — video dicts with {"video": "file://..."} (video samples only)
  - ability      : str
  - reward_model : dict        — {"style": "rule", "ground_truth": str}
  - extra_info   : dict

Usage:
  python examples/data_preprocess/onethinker.py \\
      --local_dataset_dir /path/to/OneThinker-train-data \\
      --local_save_dir ~/data/onethinker \\
      --split rl_train \\
      --val_ratio 0.02
"""

import argparse
import json
import os
import re

from datasets import Dataset
from PIL import Image

from verl.utils.hdfs_io import copy, makedirs

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

DATA_SOURCE_ID = "OneThink/OneThinker-train-data"

INSTRUCTION_THINK = (
    "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
    "The reasoning process MUST BE enclosed within <think> </think> tags. "
    "The final answer MUST BE put in <answer> </answer> tags."
)

# Map from split name to JSON filename in the dataset
SPLIT_FILE_MAP = {
    "rl_train": "onethinker_rl_train.json",
    "rl_train_unsampled": "onethinker_rl_train_unsampled.json",
    "sft_image": "onethinker_sft_image.json",
    "sft_video": "onethinker_sft_video.json",
}


def extract_answer(answer_str: str) -> str:
    """Extract the content inside <answer>...</answer> tags.

    Returns the stripped inner text, or the original string if the tag is absent.
    """
    m = re.search(r"<answer>(.*?)</answer>", answer_str, re.DOTALL)
    if m:
        return m.group(1).strip()
    return answer_str.strip()


def build_prompt_content(problem: str, options: list[str]) -> str:
    """Build the final prompt string fed to the model.

    For multiple-choice questions the options are appended explicitly so that
    the model can reference them. The universal chain-of-thought instruction is
    always appended at the end.
    """
    text = problem.strip()

    # Append options for multiple-choice if not already embedded in the problem
    if options:
        option_labels = "ABCDEFGH"
        option_lines = "\n".join(
            f"{option_labels[i]}. {opt}" for i, opt in enumerate(options) if i < len(option_labels)
        )
        # Only append if options block not already present
        if option_lines not in text:
            text = text.rstrip(":").rstrip() + "\n" + option_lines

    text = text + "\n" + INSTRUCTION_THINK
    return text


def load_image_from_path(image_rel_path: str, dataset_root: str) -> Image.Image | None:
    """Load a PIL Image from a path relative to the dataset root directory.

    Returns None if the file does not exist (graceful degradation).
    """
    full_path = os.path.join(dataset_root, image_rel_path.lstrip("./"))
    if not os.path.isfile(full_path):
        return None
    try:
        return Image.open(full_path).convert("RGB")
    except Exception:
        return None


def make_video_dict(video_rel_path: str, dataset_root: str) -> dict:
    """Build a video dict in qwen_vl_utils format.

    verl's vision_utils.process_video() expects:
        {"video": "file:///absolute/path/to/video.mp4", ...}
    """
    full_path = os.path.abspath(os.path.join(dataset_root, video_rel_path.lstrip("./")))
    return {"video": f"file://{full_path}"}


# ──────────────────────────────────────────────
# Core processing function
# ──────────────────────────────────────────────


def process_record(example: dict, idx: int, split: str, dataset_root: str) -> dict | None:
    """Convert a single OneThinker JSON record into a verl-compatible dict.

    Returns None if the record is malformed or media files are missing.
    """
    problem_id = example.get("problem_id", idx)
    problem = example.get("problem", "").strip()
    data_type = example.get("data_type", "image")  # "image" | "video"
    problem_type = example.get("problem_type", "open-ended")
    options = example.get("options") or []
    data_source_raw = example.get("data_source", DATA_SOURCE_ID)
    answer_raw = example.get("answer", "")

    # ── ground truth ──────────────────────────
    ground_truth = extract_answer(answer_raw)

    # ── prompt content ────────────────────────
    prompt_content = build_prompt_content(problem, options)

    # ── media ─────────────────────────────────
    images = []
    videos = []

    if data_type == "image":
        image_paths = example.get("images") or []
        for rel_path in image_paths:
            img = load_image_from_path(rel_path, dataset_root)
            if img is None:
                # Skip records with missing images
                return None
            images.append(img)
    elif data_type == "video":
        video_paths = example.get("videos") or []
        for rel_path in video_paths:
            videos.append(make_video_dict(rel_path, dataset_root))

    # ── ability tag ───────────────────────────
    if problem_type in ("math",):
        ability = "math"
    elif problem_type == "multiple choice":
        ability = "vqa"
    elif problem_type in ("open-ended", "captioning"):
        ability = "vqa"
    elif problem_type in ("segmentation", "grounding", "tracking"):
        ability = "grounding"
    else:
        ability = "vqa"

    # ── assemble verl record ──────────────────
    record = {
        "data_source": DATA_SOURCE_ID,
        "prompt": [
            {
                "role": "user",
                "content": prompt_content,
            }
        ],
        "ability": ability,
        "reward_model": {
            "style": "rule",
            "ground_truth": ground_truth,
        },
        "extra_info": {
            "split": split,
            "index": idx,
            "problem_id": problem_id,
            "data_type": data_type,
            "problem_type": problem_type,
            "data_source_raw": data_source_raw,
            "answer_raw": answer_raw,
            "problem": problem,
        },
    }

    # Always include both fields to ensure a consistent Arrow schema when mixing
    # image and video records in the same Dataset.from_list() call.
    # Image records carry an empty videos list, and vice versa.
    record["images"] = images  # [] for video records
    record["videos"] = videos  # [] for image records

    return record


# ──────────────────────────────────────────────
# Dataset builder
# ──────────────────────────────────────────────


def load_and_process(
    json_path: str,
    dataset_root: str,
    split_name: str,
    max_samples: int = -1,
) -> Dataset:
    """Load a OneThinker JSON file and return a HuggingFace Dataset in verl format."""
    print(f"Loading {json_path} …", flush=True)
    with open(json_path, encoding="utf-8") as f:
        raw_data = json.load(f)

    print(f"  Raw records: {len(raw_data)}", flush=True)

    if max_samples > 0:
        raw_data = raw_data[:max_samples]
        print(f"  Truncated to {max_samples} samples for quick testing.", flush=True)

    processed = []
    skipped = 0
    for idx, example in enumerate(raw_data):
        record = process_record(example, idx, split_name, dataset_root)
        if record is None:
            skipped += 1
            continue
        processed.append(record)

    print(f"  Processed: {len(processed)}  |  Skipped (missing media): {skipped}", flush=True)

    all_records = processed

    # Build HuggingFace Dataset — cast images using datasets.Image() for proper parquet storage
    hf_dataset = Dataset.from_list(all_records)

    # Both columns are always present ([] for inapplicable records), so we can
    # unconditionally cast them.  This avoids a ValueError when the inferred
    # Arrow schema omits a column because all sampled rows happened to be the
    # other modality.
    import datasets as ds_lib

    hf_dataset = hf_dataset.cast_column("images", ds_lib.Sequence(ds_lib.Image()))

    return hf_dataset


# ──────────────────────────────────────────────
# Reward score function (registered separately)
# ──────────────────────────────────────────────


def compute_score(solution_str: str, ground_truth: str) -> float:
    """Rule-based reward for OneThinker.

    Scoring strategy:
    1. Format reward (0.1): response must contain <think>…</think> and <answer>…</answer>
    2. Accuracy reward (0.9): extracted answer matches ground truth (case-insensitive)

    For segmentation / grounding tasks the ground truth is a JSON string; we do
    a lenient string-match after normalisation.
    """
    FORMAT_SCORE = 0.1
    ACC_SCORE = 1.0 - FORMAT_SCORE

    # ── format check ──────────────────────────
    has_think = bool(re.search(r"<think>.*?</think>", solution_str, re.DOTALL))
    has_answer = bool(re.search(r"<answer>.*?</answer>", solution_str, re.DOTALL))
    format_reward = FORMAT_SCORE if (has_think and has_answer) else 0.0

    # ── accuracy check ────────────────────────
    predicted = ""
    m = re.search(r"<answer>(.*?)</answer>", solution_str, re.DOTALL)
    if m:
        predicted = m.group(1).strip()

    # Normalise for comparison
    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip().lower()

    acc_reward = ACC_SCORE if _norm(predicted) == _norm(ground_truth) else 0.0

    # Partial credit: single-letter option match (A/B/C/D/E)
    if acc_reward == 0.0 and len(ground_truth) == 1 and ground_truth.upper() in "ABCDE":
        # e.g. ground truth is "B", predicted might be "B. splash the water."
        if predicted.upper().startswith(ground_truth.upper()):
            acc_reward = ACC_SCORE * 0.5

    return format_reward + acc_reward


# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess OneThinker-train-data for verl GRPO training.")
    parser.add_argument(
        "--local_dataset_dir",
        type=str,
        required=True,
        help=(
            "Root directory of the downloaded OneThinker-train-data dataset. "
            "Should contain the JSON files (onethinker_rl_train.json etc.) and "
            "the media subdirectories (QA/, Segmentation/, …)."
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        default="rl_train",
        choices=list(SPLIT_FILE_MAP.keys()),
        help="Which split/file to process (default: rl_train).",
    )
    parser.add_argument(
        "--local_save_dir",
        type=str,
        default="~/data/onethinker",
        help="Directory to save the output parquet files.",
    )
    parser.add_argument(
        "--hdfs_dir",
        type=str,
        default=None,
        help="Optional HDFS destination path.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.02,
        help="Fraction of data to use as validation set (default: 0.02).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Cap the number of input records (useful for debugging, -1 = no cap).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split.",
    )

    args = parser.parse_args()

    local_dataset_dir = os.path.expanduser(args.local_dataset_dir)
    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    json_filename = SPLIT_FILE_MAP[args.split]
    json_path = os.path.join(local_dataset_dir, json_filename)

    if not os.path.isfile(json_path):
        raise FileNotFoundError(
            f"JSON file not found: {json_path}\n"
            f"Please download the dataset from https://huggingface.co/datasets/OneThink/OneThinker-train-data "
            f"and pass the root directory via --local_dataset_dir."
        )

    # ── load & process ────────────────────────
    full_dataset = load_and_process(
        json_path=json_path,
        dataset_root=local_dataset_dir,
        split_name=args.split,
        max_samples=args.max_samples,
    )

    # ── train / val split ─────────────────────
    if args.val_ratio > 0:
        split_result = full_dataset.train_test_split(test_size=args.val_ratio, seed=args.seed, shuffle=True)
        train_dataset = split_result["train"]
        val_dataset = split_result["test"]
    else:
        train_dataset = full_dataset
        val_dataset = full_dataset.select([])  # empty

    print(f"\nTrain: {len(train_dataset)}  |  Val: {len(val_dataset)}", flush=True)

    # ── save to parquet ───────────────────────
    train_path = os.path.join(local_save_dir, "train.parquet")
    val_path = os.path.join(local_save_dir, "val.parquet")

    train_dataset.to_parquet(train_path)
    print(f"Saved train → {train_path}", flush=True)

    if len(val_dataset) > 0:
        val_dataset.to_parquet(val_path)
        print(f"Saved val   → {val_path}", flush=True)

    # ── optional: copy to HDFS ────────────────
    if args.hdfs_dir:
        makedirs(args.hdfs_dir)
        copy(src=local_save_dir, dst=args.hdfs_dir)
        print(f"Copied to HDFS: {args.hdfs_dir}", flush=True)

    print("\nDone.", flush=True)
