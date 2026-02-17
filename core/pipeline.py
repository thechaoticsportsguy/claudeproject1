"""Batch generation pipeline orchestrating prompt building, image generation, and post-processing."""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image

from core.models import GenerationMetadata, GenerationResult, Listing, Variation
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
) -> list[GenerationResult]:
    """Generate all image variations for a single listing.

    Args:
        listing: The product listing.
        provider: The image generation provider.
        variations: List of pose/background variations. Defaults to 4 standard ones.
        output_dir: Directory to save output images.
        etsy_size: Etsy size preset to use.
        watermark: Whether to add a watermark.
        template_name: Which prompt template to use.

    Returns:
        List of GenerationResult for this listing.
    """
    if variations is None:
        variations = get_default_variations()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[GenerationResult] = []

    # Check if we have a base image for img2img mode
    base_image: Image.Image | None = None
    if listing.base_image_path:
        base_path = Path(listing.base_image_path)
        if base_path.exists():
            base_image = Image.open(base_path).convert("RGB")
            logger.info("Using base image for img2img: %s", base_path)

    for idx, variation in enumerate(variations, start=1):
        logger.info(
            "Generating variation %d/%d for SKU=%s (pose=%s, bg=%s)",
            idx,
            len(variations),
            listing.sku,
            variation.pose,
            variation.background_style,
        )

        # Build the prompt
        if base_image is not None:
            prompt = build_img2img_prompt(listing, variation)
            raw_image = provider.img2img(prompt, base_image)
        else:
            prompt = build_prompt(listing, variation, template_name=template_name)
            raw_image = provider.generate(prompt)

        # Post-process and save
        filename = f"{listing.sku}_v{idx}.png"
        output_path = output_dir / filename

        saved_path, width, height = process_image(
            raw_image,
            output_path,
            etsy_size=etsy_size,
            watermark=watermark,
        )

        results.append(GenerationResult(
            sku=listing.sku,
            variation_index=idx,
            prompt=prompt,
            output_path=saved_path,
            width=width,
            height=height,
        ))

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
) -> GenerationMetadata:
    """Run batch generation across all listings.

    Args:
        listings: All product listings to process.
        provider: The image generation provider.
        num_variations: Number of variations per listing (used if custom_variations is None).
        output_dir: Base output directory.
        etsy_size: Etsy size preset.
        watermark: Whether to add watermarks.
        template_name: Which prompt template to use.
        custom_variations: Optional custom list of variations.

    Returns:
        GenerationMetadata with all results.
    """
    metadata = GenerationMetadata()
    output_dir = Path(output_dir)

    # Build variations list
    if custom_variations is not None:
        variations = custom_variations
    else:
        all_variations = get_default_variations()
        variations = all_variations[:num_variations]

    logger.info(
        "Starting batch: %d listings x %d variations = %d total images",
        len(listings),
        len(variations),
        len(listings) * len(variations),
    )

    for listing_idx, listing in enumerate(listings, start=1):
        logger.info(
            "Processing listing %d/%d: SKU=%s",
            listing_idx,
            len(listings),
            listing.sku,
        )

        results = generate_for_listing(
            listing=listing,
            provider=provider,
            variations=variations,
            output_dir=output_dir,
            etsy_size=etsy_size,
            watermark=watermark,
            template_name=template_name,
        )
        metadata.results.extend(results)

    # Save metadata
    metadata_path = output_dir / "metadata.json"
    metadata.save(metadata_path)
    logger.info("Batch complete. Metadata saved to %s", metadata_path)

    return metadata
