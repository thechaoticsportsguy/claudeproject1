"""Prompt builder that converts listing data into generation prompts."""

from __future__ import annotations

import logging

from core.models import Listing, Variation
from prompts.templates import BACKGROUNDS, POSES, TEMPLATES

logger = logging.getLogger(__name__)


def build_prompt(
    listing: Listing,
    variation: Variation,
    template_name: str = "satin_pajama_set",
) -> str:
    """Build a generation prompt from a listing and variation.

    Args:
        listing: The product listing data.
        variation: The pose/background variation to apply.
        template_name: Which prompt template to use.

    Returns:
        A fully expanded prompt string.
    """
    template = TEMPLATES.get(template_name, TEMPLATES["generic"])

    # Resolve rich background and pose descriptions
    background_desc = BACKGROUNDS.get(variation.background_style, variation.background_style)
    pose_desc = POSES.get(variation.pose, variation.pose)

    prompt = template.safe_substitute(
        title=listing.title,
        color=listing.color,
        piping_color=listing.piping_color,
        pose=pose_desc,
        background=background_desc,
        description=listing.description,
    )

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


def get_default_variations() -> list[Variation]:
    """Return the default set of 4 variations (2 backgrounds x 2 poses)."""
    return [
        Variation(pose="standing", background_style="luxury bedroom"),
        Variation(pose="standing", background_style="clean studio"),
        Variation(pose="sitting", background_style="luxury bedroom"),
        Variation(pose="sitting", background_style="clean studio"),
    ]
