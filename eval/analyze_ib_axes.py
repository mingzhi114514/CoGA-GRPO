#!/usr/bin/env python3
"""
Analyze IntentBench eval JSON (`ib_*.json`).

Output layout:
  - Avg: all samples (mean reward).
  - Social (Socialiq only): Why / How / What / When / Who/Which / Other from raw.Type.
  - Emotion (EMER) and Deception (MDPE): one aggregate line each.

Accuracy = mean(reward); MC tasks use 0/1 reward = exact-match accuracy.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable


def _norm_source(src: Any) -> str:
    return str(src or "").strip()


def _axis_from_source(src: str) -> str:
    s = _norm_source(src)
    if s == "Socialiq":
        return "Social"
    if s == "EMER":
        return "Emotion"
    if s == "MDPE":
        return "Deception"
    return f"Unknown({s or 'empty'})"


def _norm_qtype(t: Any) -> str:
    return str(t or "").strip().lower()


def _social_type_bucket(qt: str) -> str:
    """Seven buckets for Socialiq only; anything else → other."""
    if qt in {"why", "how", "what", "when"}:
        return qt
    if qt in {"who/which", "who", "which"}:
        return "who/which"
    return "other"


_SOCIAL_ORDER = ["why", "how", "what", "when", "who/which", "other"]
_SOCIAL_LABEL = {
    "why": "Why",
    "how": "How",
    "what": "What",
    "when": "When",
    "who/which": "Who/Which",
    "other": "Other",
}


@dataclass
class Bucket:
    n: int = 0
    reward_sum: float = 0.0

    def add(self, reward: float) -> None:
        self.n += 1
        self.reward_sum += float(reward)

    @property
    def acc(self) -> float:
        if self.n <= 0:
            return float("nan")
        return self.reward_sum / self.n


def _fmt(x: float) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "nan"
    return f"{x:.4f}"


def analyze(rows: Iterable[dict[str, Any]]) -> None:
    overall = Bucket()
    by_axis: dict[str, Bucket] = defaultdict(Bucket)
    social_by_type: dict[str, Bucket] = defaultdict(Bucket)

    for r in rows:
        raw = r.get("raw") or {}
        src = _norm_source(raw.get("source"))
        axis = _axis_from_source(src)
        reward = float(r.get("reward", 0.0))

        overall.add(reward)
        by_axis[axis].add(reward)
        if src == "Socialiq":
            st = _social_type_bucket(_norm_qtype(raw.get("Type")))
            social_by_type[st].add(reward)

    print(f"n={overall.n}")

    print("\n## Avg (all)")
    print(f"  acc={_fmt(overall.acc)}")

    print("\n## Social (Socialiq — Why / How / What / When / Who/Which / Other)")
    social_total = by_axis.get("Social", Bucket())
    print(f"  (Social total n={social_total.n}  acc={_fmt(social_total.acc)})")
    for key in _SOCIAL_ORDER:
        b = social_by_type.get(key, Bucket())
        label = _SOCIAL_LABEL[key]
        print(f"  {label:12s}  n={b.n:5d}  acc={_fmt(b.acc)}")

    print("\n## Emotion (EMER)")
    be = by_axis.get("Emotion", Bucket())
    print(f"  n={be.n:5d}  acc={_fmt(be.acc)}")

    print("\n## Deception (MDPE)")
    bd = by_axis.get("Deception", Bucket())
    print(f"  n={bd.n:5d}  acc={_fmt(bd.acc)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "json_path",
        type=str,
        help="Path to eval output JSON, e.g. videos/eval_results/ib_stage2_eval.json",
    )
    args = ap.parse_args()

    with open(args.json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    rows = payload.get("results")
    if not isinstance(rows, list):
        raise SystemExit("Invalid JSON: expected top-level key 'results' as a list")
    analyze(rows)


if __name__ == "__main__":
    main()
