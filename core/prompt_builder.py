"""Prompt builder that converts listing data into generation prompts."""

from __future__ import annotations

import logging

from core.models import Listing, Variation
from prompts.templates import (
    BACKGROUNDS,
    LIGHTING_PRESETS,
    POSES,
    STYLE_MODIFIERS,
    TEMPLATES,
)

logger = logging.getLogger(__name__)


def build_prompt(
    listing: Listing,
    variation: Variation,
    template_name: str = "satin_pajama_set",
    lighting: str | None = None,
    style_modifier: str | None = None,
) -> str:
    """Build a generation prompt from a listing and variation."""
    template = TEMPLATES.get(template_name, TEMPLATES["generic"])

    background_desc = BACKGROUNDS.get(variation.background_style, variation.background_style)
    pose_desc = POSES.get(variation.pose, variation.pose)

    # Use lighting override from variation if set, otherwise use parameter
    effective_lighting = variation.lighting_override or lighting

    prompt = template.safe_substitute(
        title=listing.title,
        color=listing.color,
        piping_color=listing.piping_color,
        pose=pose_desc,
        background=background_desc,
        description=listing.description,
    )

    # Append lighting preset if specified
    if effective_lighting and effective_lighting in LIGHTING_PRESETS:
        prompt += f", {LIGHTING_PRESETS[effective_lighting]}"

    # Append style modifier if specified
    if style_modifier and style_modifier in STYLE_MODIFIERS:
        prompt += f", {STYLE_MODIFIERS[style_modifier]}"

    logger.debug("Built prompt for SKU=%s: %s", listing.sku, prompt)
    return prompt


def build_img2img_prompt(
    listing: Listing,
    variation: Variation,
) -> str:
    """Build an image-to-image prompt for variation generation."""
    template = TEMPLATES["img2img"]

    background_desc = BACKGROUNDS.get(variation.background_style, variation.background_style)
    pose_desc = POSES.get(variation.pose, variation.pose)

    return template.safe_substitute(
        color=listing.color,
        piping_color=listing.piping_color,
        pose=pose_desc,
        background=background_desc,
    )


def build_mixed_style_prompt(
    listing: Listing,
    variation: Variation,
    template_names: list[str],
    weights: list[float] | None = None,
) -> str:
    """Build a prompt by mixing elements from multiple templates.

    Extracts key phrases from each template and combines them with
    optional weighting to create unique hybrid styles.
    """
    if not template_names:
        return build_prompt(listing, variation)

    if weights is None:
        weights = [1.0 / len(template_names)] * len(template_names)

    # Generate a prompt from each template
    prompts = []
    for tname in template_names:
        p = build_prompt(listing, variation, template_name=tname)
        prompts.append(p)

    # Take first prompt as base, then append unique phrases from others
    base_parts = set(prompts[0].split(", "))
    combined_parts = list(prompts[0].split(", "))

    for extra_prompt in prompts[1:]:
        for part in extra_prompt.split(", "):
            if part not in base_parts:
                combined_parts.append(part)
                base_parts.add(part)

    return ", ".join(combined_parts)


def get_default_variations() -> list[Variation]:
    """Return the default set of variations."""
    return [
        Variation(pose="standing", background_style="luxury bedroom"),
        Variation(pose="standing", background_style="clean studio"),
        Variation(pose="sitting", background_style="luxury bedroom"),
        Variation(pose="sitting", background_style="clean studio"),
    ]
