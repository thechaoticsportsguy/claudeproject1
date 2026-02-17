"""Prompt templates for Etsy product image generation."""

from __future__ import annotations

from string import Template

# --- Base product prompt templates ---

SATIN_PAJAMA_SET = Template(
    "$color satin bridal party pajama set, $piping_color piping, "
    "$pose pose, $background background, "
    "photorealistic, high detail, professional product photography, "
    "soft lighting, elegant, luxury sleepwear"
)

BRIDAL_ROBE = Template(
    "$color satin bridal robe, $piping_color trim details, "
    "$pose pose, $background background, "
    "photorealistic, high detail, professional product photography, "
    "soft natural lighting, elegant bridal wear"
)

GENERIC_PRODUCT = Template(
    "$title, $color color, $piping_color accents, "
    "$pose pose, $background background, "
    "photorealistic, high detail, professional product photography, "
    "studio lighting, sharp focus"
)

# --- Image-to-image refinement templates ---

IMG2IMG_VARIATION = Template(
    "Same $color satin pajama set with $piping_color piping as in the reference image, "
    "$pose pose, $background background, "
    "photorealistic, high detail, maintain product consistency, "
    "professional product photography"
)

# --- Background descriptions ---

BACKGROUNDS: dict[str, str] = {
    "luxury bedroom": (
        "luxurious bedroom setting with soft natural light, "
        "neutral tones, silk bedding, elegant interior"
    ),
    "clean studio": (
        "clean white studio backdrop, professional lighting, "
        "minimalist, no distractions"
    ),
    "outdoor garden": (
        "lush garden setting, soft golden hour light, "
        "greenery, romantic outdoor atmosphere"
    ),
    "minimalist white": (
        "pure white background, soft shadow, "
        "product-focused, e-commerce style"
    ),
}

# --- Pose descriptions ---

POSES: dict[str, str] = {
    "standing": "model standing naturally, relaxed confident pose, full body visible",
    "sitting": "model sitting elegantly, relaxed seated pose, showing garment drape",
}

# --- Negative prompt (for models that support it) ---

NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, deformed, ugly, "
    "watermark, text, logo, oversaturated, "
    "bad anatomy, extra limbs, cropped"
)


# Map of named templates
TEMPLATES: dict[str, Template] = {
    "satin_pajama_set": SATIN_PAJAMA_SET,
    "bridal_robe": BRIDAL_ROBE,
    "generic": GENERIC_PRODUCT,
    "img2img": IMG2IMG_VARIATION,
}
