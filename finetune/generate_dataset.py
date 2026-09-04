#!/usr/bin/env python
"""
Sinh dataset SFT ~2.500 mẫu từ 96 công thức data_book.json.

Chiến lược: ~25 biến thể câu hỏi / công thức (template, không inflate bằng LLM).
Câu trả lời = ground truth từ JSON → chất lượng cao, không hallucination.

Usage:
    python generate_dataset.py
    python generate_dataset.py --output data/nckh_sft_train.json --stats
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from formatters import (
    format_rag_user_message,
    format_recipe_markdown,
    format_reject_response,
)
from templates import (
    RAG_QUESTION_TEMPLATES,
    RECIPE_QUESTION_TEMPLATES,
    REJECT_DISHES,
    REJECT_QUESTION_TEMPLATES,
    SUGGEST_QUESTION_TEMPLATES,
    SYSTEM_PROMPT,
    TIP_QUESTION_TEMPLATES,
)

DATA_BOOK = (
    Path(__file__).resolve().parents[1]
    / "be_nckh"
    / "app"
    / "schemas"
    / "ingredient_task"
    / "data_book.json"
)
DEFAULT_OUT = Path(__file__).parent / "data" / "nckh_sft_train.json"

RNG = random.Random(42)


def short_name(full_name: str) -> str:
    """Rút gọn tên món cho template tự nhiên."""
    name = full_name.strip()
    name = re.sub(r"^\([^)]+\)\s*", "", name)
    name = re.sub(r"^Cơm gà\s*", "cơm gà ", name, flags=re.I)
    if len(name) > 40:
        parts = name.split()
        if len(parts) > 4:
            return " ".join(parts[-4:])
    return name


def join_ings(names: list[str]) -> str:
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} và {names[1]}"
    return ", ".join(names[:-1]) + f" và {names[-1]}"


def sample_to_messages(user: str, assistant: str, *, system: str = SYSTEM_PROMPT) -> dict:
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def sample_to_qwen(user: str, assistant: str) -> dict:
    """Format Qwen legacy (finetune/ LLaMA-Factory sharegpt)."""
    return {
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "user", "value": user},
            {"from": "assistant", "value": assistant},
        ]
    }


def generate_for_recipe(recipe: dict, recipe_idx: int) -> list[dict]:
    samples: list[dict] = []
    name = recipe["name"]
    short = short_name(name)
    ingredients = recipe.get("ingredients", [])
    ing_names = [i["name"] for i in ingredients]

    # 1) Hỏi trực tiếp công thức (~14)
    for t_idx, tmpl in enumerate(RECIPE_QUESTION_TEMPLATES):
        user = tmpl.format(name=name, short=short)
        assistant = format_recipe_markdown(recipe)
        samples.append(
            {
                "id": f"r{recipe_idx:03d}_direct_{t_idx:02d}",
                "intent": "recipe_direct",
                **sample_to_messages(user, assistant),
            }
        )

    # 2) Gợi ý từ nguyên liệu (~8) — subset 2–5 nguyên liệu thật của món
    for t_idx, tmpl in enumerate(SUGGEST_QUESTION_TEMPLATES):
        k = RNG.randint(2, min(5, len(ing_names)))
        subset = RNG.sample(ing_names, k=k) if len(ing_names) >= k else ing_names[:]
        user = tmpl.format(ings=join_ings(subset))
        assistant = format_recipe_markdown(recipe, owned_ingredients=subset)
        samples.append(
            {
                "id": f"r{recipe_idx:03d}_suggest_{t_idx:02d}",
                "intent": "ingredient_suggest",
                **sample_to_messages(user, assistant),
            }
        )

    # 3) Mẹo (~2)
    for t_idx, tmpl in enumerate(TIP_QUESTION_TEMPLATES):
        user = tmpl.format(name=name)
        assistant = format_recipe_markdown(recipe)
        samples.append(
            {
                "id": f"r{recipe_idx:03d}_tip_{t_idx:02d}",
                "intent": "cooking_tip",
                **sample_to_messages(user, assistant),
            }
        )

    # 4) RAG context (~2)
    for t_idx, tmpl in enumerate(RAG_QUESTION_TEMPLATES):
        user = format_rag_user_message(recipe, tmpl)
        assistant = format_recipe_markdown(recipe)
        samples.append(
            {
                "id": f"r{recipe_idx:03d}_rag_{t_idx:02d}",
                "intent": "rag_context",
                **sample_to_messages(user, assistant),
            }
        )

    return samples


def generate_reject_samples(start_id: int = 9000) -> list[dict]:
    """~100 mẫu từ chối — món không có trong 96 công thức."""
    samples = []
    idx = 0
    for dish in REJECT_DISHES:
        for tmpl in REJECT_QUESTION_TEMPLATES:
            user = tmpl.format(dish=dish)
            assistant = format_reject_response(dish)
            samples.append(
                {
                    "id": f"reject_{start_id + idx:04d}",
                    "intent": "reject_unknown",
                    **sample_to_messages(user, assistant),
                }
            )
            idx += 1
    return samples


def load_recipes(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DATA_BOOK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--also-qwen", action="store_true", help="Export thêm file Qwen conversations")
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()

    recipes = load_recipes(args.input)
    all_samples: list[dict] = []

    for i, recipe in enumerate(recipes, 1):
        all_samples.extend(generate_for_recipe(recipe, i))

    all_samples.extend(generate_reject_samples())
    RNG.shuffle(all_samples)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)

    print(f"✓ Wrote {len(all_samples)} samples → {args.output}")

    if args.also_qwen:
        qwen_path = args.output.with_name(args.output.stem + "_qwen.json")
        qwen_data = [
            {"id": s["id"], **sample_to_qwen(s["messages"][1]["content"], s["messages"][2]["content"])}
            for s in all_samples
        ]
        with open(qwen_path, "w", encoding="utf-8") as f:
            json.dump(qwen_data, f, ensure_ascii=False, indent=2)
        print(f"✓ Wrote Qwen format → {qwen_path}")

    if args.stats:
        from collections import Counter

        intents = Counter(s["intent"] for s in all_samples)
        print("\nIntent breakdown:")
        for k, v in sorted(intents.items()):
            print(f"  {k}: {v}")
        per_recipe = len(RECIPE_QUESTION_TEMPLATES) + len(SUGGEST_QUESTION_TEMPLATES) + len(TIP_QUESTION_TEMPLATES) + len(RAG_QUESTION_TEMPLATES)
        print(f"\nPer recipe: {per_recipe} samples × {len(recipes)} recipes = {per_recipe * len(recipes)}")
        print(f"Reject samples: {len(all_samples) - per_recipe * len(recipes)}")


if __name__ == "__main__":
    main()
