"""Post-processing pipeline: resize, crop, watermark, and effects for Etsy-ready images."""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

logger = logging.getLogger(__name__)

# Standard Etsy listing image sizes
ETSY_SIZES: dict[str, tuple[int, int]] = {
    "square": (2000, 2000),
    "landscape": (3000, 2400),
    "wide": (2700, 2025),  # 4:3 ratio
    "portrait": (2025, 2700),  # 3:4 ratio
    "etsy_thumb": (570, 456),
    "etsy_listing": (2700, 2025),
    "instagram_square": (1080, 1080),
    "instagram_story": (1080, 1920),
    "pinterest": (1000, 1500),
}


def resize_and_crop(image: Image.Image, width: int, height: int) -> Image.Image:
    """Resize an image to exactly width x height, center-cropping if needed."""
    src_w, src_h = image.size
    target_ratio = width / height
    src_ratio = src_w / src_h

    if src_ratio > target_ratio:
        new_h = height
        new_w = int(src_w * (height / src_h))
    else:
        new_w = width
        new_h = int(src_h * (width / src_w))

    resized = image.resize((new_w, new_h), Image.LANCZOS)

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

    font_size = max(watermarked.size) // 12
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

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


def apply_color_grade(
    image: Image.Image,
    preset: str = "none",
) -> Image.Image:
    """Apply a color grading preset to the image."""
    if preset == "none":
        return image

    img = image.copy()

    if preset == "warm":
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.15)
        r, g, b = img.split()
        r = r.point(lambda x: min(255, int(x * 1.08)))
        b = b.point(lambda x: int(x * 0.92))
        img = Image.merge("RGB", (r, g, b))

    elif preset == "cool":
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.1)
        r, g, b = img.split()
        r = r.point(lambda x: int(x * 0.92))
        b = b.point(lambda x: min(255, int(x * 1.1)))
        img = Image.merge("RGB", (r, g, b))

    elif preset == "vintage":
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(0.85)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(0.95)
        r, g, b = img.split()
        r = r.point(lambda x: min(255, int(x * 1.05)))
        g = g.point(lambda x: min(255, int(x * 1.02)))
        b = b.point(lambda x: int(x * 0.88))
        img = Image.merge("RGB", (r, g, b))

    elif preset == "high_contrast":
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.3)
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.2)

    elif preset == "soft_glow":
        blurred = img.filter(ImageFilter.GaussianBlur(radius=8))
        img = Image.blend(img, blurred, 0.25)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.05)

    elif preset == "matte":
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(0.88)
        r, g, b = img.split()
        r = r.point(lambda x: max(20, x))
        g = g.point(lambda x: max(20, x))
        b = b.point(lambda x: max(20, x))
        img = Image.merge("RGB", (r, g, b))

    return img


def add_border(
    image: Image.Image,
    border_width: int = 20,
    border_color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Add a solid border around the image."""
    new_w = image.width + 2 * border_width
    new_h = image.height + 2 * border_width
    bordered = Image.new("RGB", (new_w, new_h), border_color)
    bordered.paste(image, (border_width, border_width))
    return bordered


def sharpen_image(image: Image.Image, amount: float = 1.5) -> Image.Image:
    """Apply sharpening to the image."""
    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(amount)


def adjust_brightness_contrast(
    image: Image.Image,
    brightness: float = 1.0,
    contrast: float = 1.0,
) -> Image.Image:
    """Adjust brightness and contrast."""
    img = image
    if brightness != 1.0:
        img = ImageEnhance.Brightness(img).enhance(brightness)
    if contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast)
    return img


def create_comparison_image(
    images: list[Image.Image],
    labels: list[str] | None = None,
    padding: int = 10,
) -> Image.Image:
    """Create a side-by-side comparison image from multiple images."""
    if not images:
        raise ValueError("No images provided for comparison.")

    # Normalize all images to same height
    target_h = min(img.height for img in images)
    resized = []
    for img in images:
        ratio = target_h / img.height
        new_w = int(img.width * ratio)
        resized.append(img.resize((new_w, target_h), Image.LANCZOS))

    total_w = sum(img.width for img in resized) + padding * (len(resized) - 1)
    label_space = 40 if labels else 0
    canvas = Image.new("RGB", (total_w, target_h + label_space), (255, 255, 255))

    x_offset = 0
    draw = ImageDraw.Draw(canvas) if labels else None

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except (OSError, IOError):
        font = ImageFont.load_default()

    for i, img in enumerate(resized):
        canvas.paste(img, (x_offset, 0))
        if labels and draw and i < len(labels):
            text_bbox = draw.textbbox((0, 0), labels[i], font=font)
            text_w = text_bbox[2] - text_bbox[0]
            tx = x_offset + (img.width - text_w) // 2
            draw.text((tx, target_h + 8), labels[i], fill=(0, 0, 0), font=font)
        x_offset += img.width + padding

    return canvas


def process_image(
    image: Image.Image,
    output_path: str | Path,
    etsy_size: str = "square",
    watermark: bool = False,
    watermark_text: str = "PREVIEW",
    color_grade: str = "none",
    border: bool = False,
    border_width: int = 20,
    border_color: tuple[int, int, int] = (255, 255, 255),
    sharpen: float = 1.0,
    brightness: float = 1.0,
    contrast: float = 1.0,
) -> tuple[str, int, int]:
    """Full post-processing pipeline for a single image."""
    width, height = ETSY_SIZES.get(etsy_size, ETSY_SIZES["square"])

    processed = resize_and_crop(image, width, height)

    # Color grading
    processed = apply_color_grade(processed, preset=color_grade)

    # Brightness/contrast
    if brightness != 1.0 or contrast != 1.0:
        processed = adjust_brightness_contrast(processed, brightness, contrast)

    # Sharpening
    if sharpen != 1.0:
        processed = sharpen_image(processed, amount=sharpen)

    # Border
    if border:
        processed = add_border(processed, border_width=border_width, border_color=border_color)
        width = processed.width
        height = processed.height

    # Watermark
    if watermark:
        processed = add_watermark(processed, text=watermark_text)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed.save(str(output_path), "PNG", quality=95)

    logger.info("Saved processed image: %s (%dx%d)", output_path, width, height)
    return str(output_path), width, height
