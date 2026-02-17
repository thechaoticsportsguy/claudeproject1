"""Prompt templates for Etsy product image generation."""

from __future__ import annotations

from string import Template

# ---------------------------------------------------------------------------
# Base product prompt templates
# ---------------------------------------------------------------------------

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

EDITORIAL_FASHION = Template(
    "Editorial fashion photograph, $title in $color with $piping_color details, "
    "model $pose, $background setting, "
    "Vogue magazine quality, dramatic lighting, shallow depth of field, "
    "shot on Canon EOS R5, 85mm f/1.4 lens, 8k resolution"
)

LIFESTYLE_COZY = Template(
    "Lifestyle product photo, $title in beautiful $color with $piping_color accents, "
    "model $pose, cozy $background atmosphere, "
    "warm golden hour light filtering through sheer curtains, "
    "inviting and aspirational, Instagram-worthy composition, bokeh background"
)

FLAT_LAY = Template(
    "Flat lay product photography, $title in $color with $piping_color piping, "
    "arranged on $background surface, "
    "top-down view, styled with complementary accessories, fresh flowers, "
    "natural light from above, minimalist composition, editorial quality"
)

CLOSEUP_DETAIL = Template(
    "Extreme close-up detail shot, $title fabric texture in $color, "
    "$piping_color piping visible, $background backdrop, "
    "macro photography, shallow depth of field, showing satin sheen and stitch quality, "
    "professional product detail shot, 100mm macro lens"
)

SEASONAL_HOLIDAY = Template(
    "Holiday-themed product photo, $title in festive $color with $piping_color trim, "
    "$pose pose, $background with subtle holiday decor, "
    "warm ambient lighting, gift-worthy presentation, "
    "photorealistic, celebration atmosphere, premium quality"
)

# ---------------------------------------------------------------------------
# Image-to-image refinement templates
# ---------------------------------------------------------------------------

IMG2IMG_VARIATION = Template(
    "Same $color satin pajama set with $piping_color piping as in the reference image, "
    "$pose pose, $background background, "
    "photorealistic, high detail, maintain product consistency, "
    "professional product photography"
)

IMG2IMG_STYLE_TRANSFER = Template(
    "Transform the reference product image to a $background setting, "
    "keeping the $color satin garment with $piping_color piping identical, "
    "model in $pose pose, "
    "photorealistic, preserve product details exactly, change only environment"
)

# ---------------------------------------------------------------------------
# Background descriptions
# ---------------------------------------------------------------------------

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
    "rustic farmhouse": (
        "warm rustic farmhouse interior, reclaimed wood accents, "
        "natural linen textures, soft morning light through vintage windows"
    ),
    "modern loft": (
        "contemporary urban loft, exposed brick walls, concrete floors, "
        "industrial chic with warm accent lighting"
    ),
    "beach sunset": (
        "golden beach at sunset, gentle waves, warm amber light, "
        "soft sand, tropical romantic atmosphere"
    ),
    "marble bathroom": (
        "luxury marble bathroom, carrara marble surfaces, "
        "elegant brass fixtures, spa-like atmosphere, soft diffused light"
    ),
    "boutique hotel": (
        "upscale boutique hotel suite, velvet furnishings, "
        "art deco accents, warm ambient glow, curated luxury"
    ),
    "rooftop terrace": (
        "chic rooftop terrace at twilight, city skyline backdrop, "
        "string lights, lounge seating, urban sophistication"
    ),
}

# ---------------------------------------------------------------------------
# Pose descriptions
# ---------------------------------------------------------------------------

POSES: dict[str, str] = {
    "standing": "model standing naturally, relaxed confident pose, full body visible",
    "sitting": "model sitting elegantly, relaxed seated pose, showing garment drape",
    "walking": "model walking gracefully, mid-stride movement, dynamic pose showing flow of fabric",
    "reclining": "model reclining luxuriously, lounging pose on soft surface, relaxed elegance",
    "close-up": "close-up torso shot, detailed view of fabric and construction, upper body focus",
    "back view": "model turned away, showing back details of garment, elegant over-shoulder glance",
    "candid": "candid lifestyle moment, natural unposed look, authentic and relatable",
}

# ---------------------------------------------------------------------------
# Lighting presets
# ---------------------------------------------------------------------------

LIGHTING_PRESETS: dict[str, str] = {
    "natural soft": "soft natural window light, gentle shadows, warm tone",
    "studio dramatic": "dramatic studio lighting, strong key light, deep shadows, moody",
    "golden hour": "warm golden hour sunlight, long shadows, amber glow",
    "ring light": "even ring light illumination, minimal shadows, beauty lighting",
    "backlit": "backlit silhouette edge light, ethereal glow, atmospheric",
    "overcast": "soft overcast daylight, even illumination, no harsh shadows",
}

# ---------------------------------------------------------------------------
# Style modifiers that can be mixed in
# ---------------------------------------------------------------------------

STYLE_MODIFIERS: dict[str, str] = {
    "vintage film": "shot on Kodak Portra 400 film, subtle grain, warm vintage color grading",
    "high fashion": "high-end fashion editorial, Helmut Newton inspired, dramatic contrast",
    "soft dreamy": "dreamy soft focus, pastel tones, ethereal atmosphere, lens flare",
    "bold vibrant": "bold saturated colors, high contrast, pop art inspired, eye-catching",
    "moody dark": "dark moody atmosphere, chiaroscuro lighting, rich deep tones",
    "clean minimal": "ultra-clean minimalist, negative space, Scandinavian design aesthetic",
}

# ---------------------------------------------------------------------------
# Negative prompt (for models that support it)
# ---------------------------------------------------------------------------

NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, deformed, ugly, "
    "watermark, text, logo, oversaturated, "
    "bad anatomy, extra limbs, cropped"
)

NEGATIVE_PROMPT_STRICT = (
    "blurry, low quality, distorted, deformed, ugly, disfigured, "
    "watermark, text, logo, oversaturated, underexposed, "
    "bad anatomy, extra limbs, missing limbs, cropped, "
    "amateur, snapshot quality, low resolution, pixelated, "
    "plastic skin, mannequin, CGI look, uncanny valley"
)

# ---------------------------------------------------------------------------
# Map of named templates
# ---------------------------------------------------------------------------

TEMPLATES: dict[str, Template] = {
    "satin_pajama_set": SATIN_PAJAMA_SET,
    "bridal_robe": BRIDAL_ROBE,
    "generic": GENERIC_PRODUCT,
    "editorial_fashion": EDITORIAL_FASHION,
    "lifestyle_cozy": LIFESTYLE_COZY,
    "flat_lay": FLAT_LAY,
    "closeup_detail": CLOSEUP_DETAIL,
    "seasonal_holiday": SEASONAL_HOLIDAY,
    "img2img": IMG2IMG_VARIATION,
    "img2img_style_transfer": IMG2IMG_STYLE_TRANSFER,
}
