"""Analytics and tracking engine for generation sessions."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.models import GenerationResult, PROVIDER_COST_ESTIMATES


@dataclass
class SessionAnalytics:
    """Tracks analytics for the current generation session."""

    generation_log: list[dict[str, Any]] = field(default_factory=list)
    provider_counts: dict[str, int] = field(default_factory=dict)
    provider_times: dict[str, list[float]] = field(default_factory=dict)
    provider_costs: dict[str, float] = field(default_factory=dict)
    sku_counts: dict[str, int] = field(default_factory=dict)
    template_counts: dict[str, int] = field(default_factory=dict)
    total_images: int = 0
    session_start: float = field(default_factory=time.time)
    errors: list[dict[str, Any]] = field(default_factory=list)

    def record_generation(self, result: GenerationResult, template_name: str = "") -> None:
        """Record a successful generation."""
        self.total_images += 1

        # Provider tracking
        provider = result.provider or "unknown"
        self.provider_counts[provider] = self.provider_counts.get(provider, 0) + 1

        if provider not in self.provider_times:
            self.provider_times[provider] = []
        self.provider_times[provider].append(result.generation_time_s)

        cost = PROVIDER_COST_ESTIMATES.get(provider, 0.0)
        self.provider_costs[provider] = self.provider_costs.get(provider, 0.0) + cost

        # SKU tracking
        self.sku_counts[result.sku] = self.sku_counts.get(result.sku, 0) + 1

        # Template tracking
        if template_name:
            self.template_counts[template_name] = self.template_counts.get(template_name, 0) + 1

        # Detailed log
        self.generation_log.append({
            "sku": result.sku,
            "provider": provider,
            "time_s": result.generation_time_s,
            "cost": cost,
            "template": template_name,
            "timestamp": result.timestamp,
            "size": f"{result.width}x{result.height}",
        })

    def record_error(self, sku: str, provider: str, error: str) -> None:
        self.errors.append({
            "sku": sku,
            "provider": provider,
            "error": error,
            "timestamp": time.time(),
        })

    @property
    def total_cost(self) -> float:
        return sum(self.provider_costs.values())

    @property
    def total_time(self) -> float:
        all_times = []
        for times in self.provider_times.values():
            all_times.extend(times)
        return sum(all_times)

    @property
    def avg_time_per_image(self) -> float:
        if self.total_images == 0:
            return 0.0
        return self.total_time / self.total_images

    @property
    def session_duration(self) -> float:
        return time.time() - self.session_start

    def provider_avg_time(self, provider: str) -> float:
        times = self.provider_times.get(provider, [])
        if not times:
            return 0.0
        return sum(times) / len(times)

    def get_summary(self) -> dict[str, Any]:
        return {
            "total_images": self.total_images,
            "total_cost": round(self.total_cost, 4),
            "total_time_s": round(self.total_time, 2),
            "avg_time_per_image_s": round(self.avg_time_per_image, 2),
            "session_duration_s": round(self.session_duration, 2),
            "providers": {
                p: {
                    "count": self.provider_counts.get(p, 0),
                    "avg_time_s": round(self.provider_avg_time(p), 2),
                    "total_cost": round(self.provider_costs.get(p, 0.0), 4),
                }
                for p in self.provider_counts
            },
            "skus_processed": dict(self.sku_counts),
            "templates_used": dict(self.template_counts),
            "errors": len(self.errors),
        }

    def to_dataframe_records(self) -> list[dict[str, Any]]:
        """Return generation log as records suitable for a pandas DataFrame."""
        return self.generation_log

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.get_summary(), f, indent=2)
