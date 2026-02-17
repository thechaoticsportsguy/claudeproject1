"""Data models for the Etsy image generator."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class Pose(str, Enum):
    STANDING = "standing"
    SITTING = "sitting"


class BackgroundStyle(str, Enum):
    LUXURY_BEDROOM = "luxury bedroom"
    CLEAN_STUDIO = "clean studio"
    OUTDOOR_GARDEN = "outdoor garden"
    MINIMALIST_WHITE = "minimalist white"


class ProviderName(str, Enum):
    REPLICATE = "replicate"
    OPENAI = "openai"


@dataclass
class Listing:
    sku: str
    title: str
    description: str
    color: str
    piping_color: str
    pose: str = "standing"
    background_style: str = "clean studio"
    base_image_path: str | None = None

    @classmethod
    def from_csv(cls, path: str | Path) -> list[Listing]:
        """Load listings from a CSV file."""
        listings: list[Listing] = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                listings.append(cls(
                    sku=row["sku"],
                    title=row["title"],
                    description=row.get("description", ""),
                    color=row.get("color", ""),
                    piping_color=row.get("piping_color", ""),
                    pose=row.get("pose", "standing"),
                    background_style=row.get("background_style", "clean studio"),
                    base_image_path=row.get("base_image_path"),
                ))
        return listings

    @classmethod
    def from_json(cls, path: str | Path) -> list[Listing]:
        """Load listings from a JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        items = data if isinstance(data, list) else data.get("listings", [])
        return [cls(**item) for item in items]


@dataclass
class Variation:
    pose: str
    background_style: str


@dataclass
class GenerationResult:
    sku: str
    variation_index: int
    prompt: str
    output_path: str
    width: int
    height: int


@dataclass
class GenerationMetadata:
    results: list[GenerationResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary grouped by SKU."""
        grouped: dict[str, dict] = {}
        for r in self.results:
            if r.sku not in grouped:
                grouped[r.sku] = {"sku": r.sku, "variations": []}
            grouped[r.sku]["variations"].append({
                "variation_index": r.variation_index,
                "prompt": r.prompt,
                "output_path": r.output_path,
                "width": r.width,
                "height": r.height,
            })
        return {"listings": list(grouped.values())}

    def save(self, path: str | Path) -> None:
        """Save metadata to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
