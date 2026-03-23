#!/usr/bin/env python3
"""
过滤 onethinker_rl_train.json，保留以下子集中的多模态数据条目：

Open-ended:
  - Spatial   → data_source: OpenSpaces, Spacellava
  - General   → data_source: sharegpt4o, sharegpt4v(knowledge)

QA:
  - Holmes    → data_source: Holmes-train
  - CLEVR     → data_source: CLEVRER

Segmentation:
  - Davis17   → data_source: Ref-DAVIS17
  - ReVOS     → data_source: ReVOS

Spatial-Grounding:
  - train2014 → data_source: RefCOCO, RefCOCOp, RefCOCOg

Temporal-Grounding:
  - HiREST    → data_source: HiREST

Tracking:
  - ElysiumTrack → data_source: ElysiumTrack
"""

import json
import os
from collections import Counter

# ── 目标 data_source 配置 ────────────────────────────────────────────────────
TARGET_SOURCES = {
    # Open-ended / Spatial  (769MB)
    "OpenSpaces",
    "Spacellava",
    # Open-ended / General  (879MB)
    "sharegpt4o",
    "sharegpt4v(knowledge)",
    # QA / Holmes           (4.28GB)
    "Holmes-train",
    # QA / CLEVR            (447MB) — 仅 CLEVRER 视频数据集，不含 Math 衍生集
    "CLEVRER",
    # Segmentation / Davis17 (1.1GB)
    "Ref-DAVIS17",
    # Segmentation / ReVOS  (4.32GB)
    "ReVOS",
    # Spatial-Grounding / train2014 (12GB)
    "RefCOCO",
    "RefCOCOp",
    "RefCOCOg",
    # Temporal-Grounding / HiREST (6.46GB)
    "HiREST",
    # Tracking / ElysiumTrack (4.45GB)
    "ElysiumTrack",
}

# ── 路径配置 ─────────────────────────────────────────────────────────────────
INPUT_FILE = os.path.join(os.path.dirname(__file__), "onethinker_rl_train.json")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "onethinker_rl_train_filtered.json")


def main():
    print(f"Loading {INPUT_FILE} ...")
    with open(INPUT_FILE, encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    print(f"Total records: {total:,}")

    # ── 过滤 ──────────────────────────────────────────────────────────────────
    filtered = [item for item in data if item.get("data_source") in TARGET_SOURCES]

    kept = len(filtered)
    print(f"\nFiltered records: {kept:,} / {total:,}  ({kept / total * 100:.1f}%)")

    # ── 统计各子集数量 ────────────────────────────────────────────────────────
    counter = Counter(item["data_source"] for item in filtered)
    print("\n── Per-source breakdown ──")
    groups = {
        "Open-ended / Spatial  ": ["OpenSpaces", "Spacellava"],
        "Open-ended / General  ": ["sharegpt4o", "sharegpt4v(knowledge)"],
        "QA / Holmes            ": ["Holmes-train"],
        "QA / CLEVR             ": ["CLEVRER"],
        "Segmentation / Davis17 ": ["Ref-DAVIS17"],
        "Segmentation / ReVOS   ": ["ReVOS"],
        "Spatial-Grounding      ": ["RefCOCO", "RefCOCOp", "RefCOCOg"],
        "Temporal-Grounding     ": ["HiREST"],
        "Tracking               ": ["ElysiumTrack"],
    }
    for group_name, sources in groups.items():
        subtotal = sum(counter[s] for s in sources)
        detail = "  ".join(f"{s}={counter[s]:,}" for s in sources)
        print(f"  {group_name}  total={subtotal:,}   [{detail}]")

    # ── 统计 data_type ────────────────────────────────────────────────────────
    type_counter = Counter(item.get("data_type", "unknown") for item in filtered)
    print("\n── data_type breakdown ──")
    for dtype, cnt in sorted(type_counter.items()):
        print(f"  {dtype}: {cnt:,}")

    # ── 写出结果 ──────────────────────────────────────────────────────────────
    print(f"\nSaving to {OUTPUT_FILE} ...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    file_size_mb = os.path.getsize(OUTPUT_FILE) / 1024 / 1024
    print(f"Done! Output file size: {file_size_mb:.1f} MB")


if __name__ == "__main__":
    main()
