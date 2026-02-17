"""Batch generation pipeline orchestrating prompt building, image generation, and post-processing."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from PIL import Image

from core.analytics import SessionAnalytics
from core.gemini_enhancer import GeminiEnhancer
from core.models import (
    GenerationMetadata,
    GenerationResult,
    Listing,
    PROVIDER_COST_ESTIMATES,
    Variation,
)
from core.postprocess import process_image
from core.prompt_builder import build_img2img_prompt, build_prompt, get_default_variations
from core.providers import ImageProvider

logger = logging.getLogger(__name__)


def generate_for_listing(
    listing: Listing,
    provider: ImageProvider,
    variations: list[Variation] | None = None,
    output_dir: str | Path = "outputs",
    etsy_size: str = "square",
    watermark: bool = False,
    template_name: str = "satin_pajama_set",
    enhance_prompts: bool = False,
    enhancer: GeminiEnhancer | None = None,
    style_hints: str = "",
    lighting: str | None = None,
    style_modifier: str | None = None,
    color_grade: str = "none",
    sharpen: float = 1.0,
    brightness: float = 1.0,
    contrast: float = 1.0,
    analytics: SessionAnalytics | None = None,
    progress_callback=None,
) -> list[GenerationResult]:
    """Generate all image variations for a single listing.

    Supports AI prompt enhancement via Gemini, configurable post-processing,
    and real-time analytics tracking.
    """
    if variations is None:
        variations = get_default_variations()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[GenerationResult] = []

    # Check for base image (img2img mode)
    base_image: Image.Image | None = None
    if listing.base_image_path:
        base_path = Path(listing.base_image_path)
        if base_path.exists():
            base_image = Image.open(base_path).convert("RGB")
            logger.info("Using base image for img2img: %s", base_path)

    for idx, variation in enumerate(variations, start=1):
        logger.info(
            "Generating variation %d/%d for SKU=%s (pose=%s, bg=%s)",
            idx, len(variations), listing.sku,
            variation.pose, variation.background_style,
        )

        # Build prompt
        if base_image is not None:
            prompt = build_img2img_prompt(listing, variation)
        else:
            prompt = build_prompt(
                listing, variation,
                template_name=template_name,
                lighting=lighting,
                style_modifier=style_modifier,
            )

        # AI enhancement
        enhanced_prompt = ""
        if enhance_prompts and enhancer and enhancer.available:
            enhanced_prompt = enhancer.enhance_prompt(prompt, style_hints=style_hints)
            generation_prompt = enhanced_prompt
        else:
            generation_prompt = prompt

        # Generate
        start_time = time.time()
        try:
            if base_image is not None:
                raw_image = provider.img2img(generation_prompt, base_image)
            else:
                raw_image = provider.generate(generation_prompt)
            gen_time = time.time() - start_time
        except Exception:
            gen_time = time.time() - start_time
            if analytics:
                analytics.record_error(listing.sku, provider.provider_name, "Generation failed")
            raise

        # Post-process
        filename = f"{listing.sku}_v{idx}.png"
        output_path = output_dir / filename

        saved_path, width, height = process_image(
            raw_image,
            output_path,
            etsy_size=etsy_size,
            watermark=watermark,
            color_grade=color_grade,
            sharpen=sharpen,
            brightness=brightness,
            contrast=contrast,
        )

        cost = PROVIDER_COST_ESTIMATES.get(provider.provider_name, 0.0)

        result = GenerationResult(
            sku=listing.sku,
            variation_index=idx,
            prompt=prompt,
            output_path=saved_path,
            width=width,
            height=height,
            provider=provider.provider_name,
            generation_time_s=round(gen_time, 2),
            enhanced_prompt=enhanced_prompt,
            cost_estimate=cost,
        )
        results.append(result)

        if analytics:
            analytics.record_generation(result, template_name=template_name)

        if progress_callback:
            progress_callback(idx, len(variations), result)

    return results


def run_batch(
    listings: list[Listing],
    provider: ImageProvider,
    num_variations: int = 4,
    output_dir: str | Path = "outputs",
    etsy_size: str = "square",
    watermark: bool = False,
    template_name: str = "satin_pajama_set",
    custom_variations: list[Variation] | None = None,
    enhance_prompts: bool = False,
    enhancer: GeminiEnhancer | None = None,
    analytics: SessionAnalytics | None = None,
) -> GenerationMetadata:
    """Run batch generation across all listings."""
    metadata = GenerationMetadata()
    output_dir = Path(output_dir)

    if custom_variations is not None:
        variations = custom_variations
    else:
        all_variations = get_default_variations()
        variations = all_variations[:num_variations]

    logger.info(
        "Starting batch: %d listings x %d variations = %d total images",
        len(listings), len(variations), len(listings) * len(variations),
    )

    for listing_idx, listing in enumerate(listings, start=1):
        logger.info(
            "Processing listing %d/%d: SKU=%s",
            listing_idx, len(listings), listing.sku,
        )

        results = generate_for_listing(
            listing=listing,
            provider=provider,
            variations=variations,
            output_dir=output_dir,
            etsy_size=etsy_size,
            watermark=watermark,
            template_name=template_name,
            enhance_prompts=enhance_prompts,
            enhancer=enhancer,
            analytics=analytics,
        )
        metadata.results.extend(results)

    metadata.compute_totals()
    metadata_path = output_dir / "metadata.json"
    metadata.save(metadata_path)
    logger.info("Batch complete. Metadata saved to %s", metadata_path)

    return metadata
