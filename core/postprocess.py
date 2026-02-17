"""Post-processing pipeline: resize, crop, and watermark for Etsy-ready images."""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# Standard Etsy listing image sizes
ETSY_SIZES: dict[str, tuple[int, int]] = {
    "square": (2000, 2000),
    "landscape": (3000, 2400),
    "wide": (2700, 2025),  # 4:3 ratio
}


def resize_and_crop(image: Image.Image, width: int, height: int) -> Image.Image:
    """Resize an image to exactly width x height, center-cropping if needed.

    Scales the image so the shorter dimension matches the target, then
    center-crops the longer dimension.
    """
    src_w, src_h = image.size
    target_ratio = width / height
    src_ratio = src_w / src_h

    if src_ratio > target_ratio:
        # Source is wider — scale by height, crop width
        new_h = height
        new_w = int(src_w * (height / src_h))
    else:
        # Source is taller — scale by width, crop height
        new_w = width
        new_h = int(src_h * (width / src_w))

    resized = image.resize((new_w, new_h), Image.LANCZOS)

    # Center crop
    left = (new_w - width) // 2
    top = (new_h - height) // 2
    cropped = resized.crop((left, top, left + width, top + height))

    return cropped


def add_watermark(
    image: Image.Image,
    text: str = "PREVIEW",
    opacity: int = 40,
) -> Image.Image:
    """Add a subtle diagonal watermark to the image."""
    watermarked = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", watermarked.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Use a large font size relative to image
    font_size = max(watermarked.size) // 12
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()

    # Get text dimensions
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Center the watermark
    x = (watermarked.width - text_w) // 2
    y = (watermarked.height - text_h) // 2

    draw.text(
        (x, y),
        text,
        fill=(255, 255, 255, opacity),
        font=font,
    )

    watermarked = Image.alpha_composite(watermarked, overlay)
    return watermarked.convert("RGB")


def process_image(
    image: Image.Image,
    output_path: str | Path,
    etsy_size: str = "square",
    watermark: bool = False,
    watermark_text: str = "PREVIEW",
) -> tuple[str, int, int]:
    """Full post-processing pipeline for a single image.

    Args:
        image: The source PIL Image.
        output_path: Where to save the final image.
        etsy_size: One of the ETSY_SIZES keys.
        watermark: Whether to apply a watermark.
        watermark_text: Text for the watermark.

    Returns:
        Tuple of (output_path, width, height).
    """
    width, height = ETSY_SIZES.get(etsy_size, ETSY_SIZES["square"])

    processed = resize_and_crop(image, width, height)

    if watermark:
        processed = add_watermark(processed, text=watermark_text)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed.save(str(output_path), "PNG", quality=95)

    logger.info("Saved processed image: %s (%dx%d)", output_path, width, height)
    return str(output_path), width, height
