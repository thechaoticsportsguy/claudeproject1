"""Data models for the Etsy image generator."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class Pose(str, Enum):
    STANDING = "standing"
    SITTING = "sitting"
    WALKING = "walking"
    RECLINING = "reclining"
    CLOSE_UP = "close-up"


class BackgroundStyle(str, Enum):
    LUXURY_BEDROOM = "luxury bedroom"
    CLEAN_STUDIO = "clean studio"
    OUTDOOR_GARDEN = "outdoor garden"
    MINIMALIST_WHITE = "minimalist white"
    RUSTIC_FARMHOUSE = "rustic farmhouse"
    MODERN_LOFT = "modern loft"
    BEACH_SUNSET = "beach sunset"
    MARBLE_BATHROOM = "marble bathroom"


class ProviderName(str, Enum):
    REPLICATE = "replicate"
    OPENAI = "openai"
    GEMINI = "gemini"


class GenerationStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class QualityPreset(str, Enum):
    DRAFT = "draft"
    STANDARD = "standard"
    HIGH = "high"
    ULTRA = "ultra"


QUALITY_PRESET_CONFIG: dict[str, dict[str, Any]] = {
    "draft": {"steps": 4, "cfg_scale": 5.0, "detail_level": "low"},
    "standard": {"steps": 20, "cfg_scale": 7.0, "detail_level": "medium"},
    "high": {"steps": 35, "cfg_scale": 7.5, "detail_level": "high"},
    "ultra": {"steps": 50, "cfg_scale": 8.0, "detail_level": "ultra"},
}

PROVIDER_COST_ESTIMATES: dict[str, float] = {
    "replicate": 0.003,
    "openai": 0.040,
    "gemini": 0.020,
}


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
    tags: list[str] = field(default_factory=list)
    category: str = "sleepwear"

    @classmethod
    def from_csv(cls, path: str | Path) -> list[Listing]:
        """Load listings from a CSV file."""
        listings: list[Listing] = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tags_raw = row.get("tags", "")
                tags = [t.strip() for t in tags_raw.split(",") if t.strip()] if tags_raw else []
                listings.append(cls(
                    sku=row["sku"],
                    title=row["title"],
                    description=row.get("description", ""),
                    color=row.get("color", ""),
                    piping_color=row.get("piping_color", ""),
                    pose=row.get("pose", "standing"),
                    background_style=row.get("background_style", "clean studio"),
                    base_image_path=row.get("base_image_path"),
                    tags=tags,
                    category=row.get("category", "sleepwear"),
                ))
        return listings

    @classmethod
    def from_json(cls, path: str | Path) -> list[Listing]:
        """Load listings from a JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        items = data if isinstance(data, list) else data.get("listings", [])
        result = []
        for item in items:
            if "tags" not in item:
                item["tags"] = []
            if "category" not in item:
                item["category"] = "sleepwear"
            result.append(cls(**item))
        return result

    @property
    def fingerprint(self) -> str:
        """A short hash uniquely identifying this listing's content."""
        raw = f"{self.sku}:{self.title}:{self.color}:{self.piping_color}"
        return hashlib.md5(raw.encode()).hexdigest()[:8]


@dataclass
class Variation:
    pose: str
    background_style: str
    style_weight: float = 1.0
    lighting_override: str | None = None


@dataclass
class GenerationResult:
    sku: str
    variation_index: int
    prompt: str
    output_path: str
    width: int
    height: int
    provider: str = ""
    generation_time_s: float = 0.0
    enhanced_prompt: str = ""
    quality_preset: str = "standard"
    cost_estimate: float = 0.0
    timestamp: float = field(default_factory=time.time)

    @property
    def result_id(self) -> str:
        return f"{self.sku}_v{self.variation_index}_{int(self.timestamp)}"


@dataclass
class GenerationMetadata:
    results: list[GenerationResult] = field(default_factory=list)
    total_cost_estimate: float = 0.0
    total_generation_time_s: float = 0.0

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary grouped by SKU."""
        grouped: dict[str, dict] = {}
        for r in self.results:
            if r.sku not in grouped:
                grouped[r.sku] = {"sku": r.sku, "variations": []}
            grouped[r.sku]["variations"].append({
                "variation_index": r.variation_index,
                "prompt": r.prompt,
                "enhanced_prompt": r.enhanced_prompt,
                "output_path": r.output_path,
                "width": r.width,
                "height": r.height,
                "provider": r.provider,
                "generation_time_s": r.generation_time_s,
                "quality_preset": r.quality_preset,
                "cost_estimate": r.cost_estimate,
                "timestamp": r.timestamp,
            })
        return {
            "listings": list(grouped.values()),
            "summary": {
                "total_images": len(self.results),
                "total_cost_estimate": self.total_cost_estimate,
                "total_generation_time_s": self.total_generation_time_s,
                "providers_used": list({r.provider for r in self.results}),
            },
        }

    def save(self, path: str | Path) -> None:
        """Save metadata to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def compute_totals(self) -> None:
        self.total_cost_estimate = sum(r.cost_estimate for r in self.results)
        self.total_generation_time_s = sum(r.generation_time_s for r in self.results)


@dataclass
class PromptHistoryEntry:
    original_prompt: str
    enhanced_prompt: str
    provider: str
    sku: str
    timestamp: float = field(default_factory=time.time)
    is_favorite: bool = False


@dataclass
class SessionState:
    """Tracks all state for the current Streamlit session."""
    generation_history: list[GenerationResult] = field(default_factory=list)
    prompt_history: list[PromptHistoryEntry] = field(default_factory=list)
    favorite_prompts: list[str] = field(default_factory=list)
    total_images_generated: int = 0
    total_cost: float = 0.0
    provider_usage: dict[str, int] = field(default_factory=dict)
