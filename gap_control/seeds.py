"""Rich, balanced seed grid for synthetic classifier data.

Goal (user requirement): within a single class (e.g. emotion=anger) the data must span many
*genres* and *registers* and *topics* and *lengths* — not be dominated by one shape. We
enumerate a balanced grid of seed cells per (dim, label) so every genre appears roughly
equally, and topic/length/register are spread evenly. Each cell also gets a rotated
generator model, so the classifier sees multiple model "voices" and won't overfit one
model's quirks.
"""
from __future__ import annotations

import random
from typing import List

# 体裁 — the form/structure of the text
GENRES = [
    "a product review", "a restaurant review", "a movie/TV review",
    "a personal diary entry", "a short story excerpt", "a social media post",
    "a forum comment", "a customer-support message", "a news report sentence",
    "a personal email", "a text message to a friend", "a blog paragraph",
    "an advertisement", "a travel note", "a complaint letter", "a recommendation",
    "an opinion editorial line", "a dialogue line between two people",
]

# 主题 — the topic/domain the text is about
TOPICS = [
    "food and dining", "travel and places", "technology and gadgets",
    "movies and shows", "books and reading", "work and career",
    "relationships and family", "health and fitness", "sports and games",
    "weather and seasons", "shopping and products", "education and learning",
    "music and concerts", "cars and transport", "home and living",
    "money and finance", "art and design", "nature and animals",
]

# register/tone variation (NOT applied to the `style` dim, whose label *is* the register)
REGISTERS = [
    "in a plain everyday tone", "in a casual conversational tone",
    "in a polished professional tone", "in an enthusiastic tone",
    "in an understated, dry tone", "in a vivid descriptive tone",
]

# length targets the cell asks for
LENGTHS = [
    ("very short (1 sentence, <15 words)", 1),
    ("short (1-2 sentences)", 2),
    ("medium (2-3 sentences)", 3),
    ("a longer (3-4 sentences)", 4),
]


def seed_cells(dim: str, label: str, n_cells: int, seed: int = 0) -> List[dict]:
    """Return n_cells balanced seed specs. Genres are assigned round-robin (so each appears
    ~n_cells/len(GENRES) times); topic/length/register are shuffled-cycled for even spread."""
    rng = random.Random(f"{dim}|{label}|{seed}")
    genres = (GENRES * (n_cells // len(GENRES) + 1))[:n_cells]
    topics = TOPICS[:]
    lengths = LENGTHS[:]
    registers = REGISTERS[:]
    rng.shuffle(genres)
    cells = []
    for i in range(n_cells):
        cell = {
            "genre": genres[i],
            "topic": topics[(i + rng.randint(0, len(topics) - 1)) % len(topics)],
            "length_desc": lengths[i % len(lengths)][0],
            "length_n": lengths[i % len(lengths)][1],
        }
        # the style dimension's label already fixes the register -> don't override it
        if dim != "style":
            cell["register"] = registers[i % len(registers)]
        cells.append(cell)
    rng.shuffle(cells)
    return cells
