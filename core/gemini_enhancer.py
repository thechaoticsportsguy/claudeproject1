"""AI-powered prompt enhancement engine using Google Gemini."""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

ENHANCEMENT_SYSTEM_PROMPT = """\
You are an expert AI image prompt engineer specializing in e-commerce product photography.
Your job is to take a basic product image prompt and transform it into a highly detailed,
photorealistic prompt that will produce stunning product images suitable for Etsy listings.

Rules:
- Keep the core product description intact (color, material, style)
- Add specific photography terms (lens type, lighting setup, depth of field)
- Add texture and material detail words
- Add mood and atmosphere descriptors
- Keep the prompt under 300 words
- Do NOT add any explanation — output ONLY the enhanced prompt text
- Focus on making the product look premium and desirable
"""

STYLE_ANALYSIS_PROMPT = """\
Analyze the following product listing and suggest the 3 best visual styles for
e-commerce product photography. For each style, provide:
1. Style name (2-3 words)
2. Key visual elements (lighting, background, mood)
3. Why it works for this product

Product: {title}
Description: {description}
Color: {color}

Output as a numbered list. Be specific and actionable.
"""

PROMPT_CRITIQUE_PROMPT = """\
You are an expert prompt engineer. Critique the following image generation prompt
and suggest specific improvements. Rate it on a scale of 1-10 for:
- Specificity (how detailed and clear)
- Photography quality (lighting, composition terms)
- E-commerce readiness (would it sell on Etsy?)

Prompt: {prompt}

Provide brief, actionable feedback.
"""


class GeminiEnhancer:
    """Uses Google Gemini to enhance image generation prompts and provide style analysis."""

    def __init__(self, api_key: str | None = None, model: str = "gemini-2.0-flash") -> None:
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def enhance_prompt(self, basic_prompt: str, style_hints: str = "") -> str:
        """Take a basic prompt and return an enhanced, detailed version."""
        if not self.available:
            logger.warning("Gemini API key not set — returning original prompt")
            return basic_prompt

        try:
            client = self._get_client()

            user_msg = f"Enhance this image generation prompt:\n\n{basic_prompt}"
            if style_hints:
                user_msg += f"\n\nAdditional style direction: {style_hints}"

            response = client.models.generate_content(
                model=self.model,
                contents=user_msg,
                config={
                    "system_instruction": ENHANCEMENT_SYSTEM_PROMPT,
                    "temperature": 0.7,
                    "max_output_tokens": 500,
                },
            )

            enhanced = response.text.strip()
            logger.info("Prompt enhanced: %d chars -> %d chars", len(basic_prompt), len(enhanced))
            return enhanced

        except Exception as e:
            logger.error("Gemini enhancement failed: %s", e)
            return basic_prompt

    def analyze_styles(self, title: str, description: str, color: str) -> str:
        """Analyze a product and suggest optimal visual styles."""
        if not self.available:
            return "Gemini API key not configured — style analysis unavailable."

        try:
            client = self._get_client()

            prompt = STYLE_ANALYSIS_PROMPT.format(
                title=title, description=description, color=color
            )

            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={"temperature": 0.8, "max_output_tokens": 600},
            )

            return response.text.strip()

        except Exception as e:
            logger.error("Style analysis failed: %s", e)
            return f"Style analysis error: {e}"

    def critique_prompt(self, prompt: str) -> str:
        """Get AI feedback on a prompt's quality."""
        if not self.available:
            return "Gemini API key not configured."

        try:
            client = self._get_client()

            user_msg = PROMPT_CRITIQUE_PROMPT.format(prompt=prompt)

            response = client.models.generate_content(
                model=self.model,
                contents=user_msg,
                config={"temperature": 0.5, "max_output_tokens": 400},
            )

            return response.text.strip()

        except Exception as e:
            logger.error("Prompt critique failed: %s", e)
            return f"Critique error: {e}"

    def generate_seo_tags(self, title: str, description: str) -> list[str]:
        """Generate Etsy SEO tags for a product listing."""
        if not self.available:
            return []

        try:
            client = self._get_client()

            response = client.models.generate_content(
                model=self.model,
                contents=(
                    f"Generate exactly 13 Etsy SEO tags (comma-separated, no numbering) for:\n"
                    f"Title: {title}\nDescription: {description}\n"
                    f"Output ONLY the comma-separated tags, nothing else."
                ),
                config={"temperature": 0.6, "max_output_tokens": 200},
            )

            raw = response.text.strip()
            tags = [t.strip() for t in raw.split(",") if t.strip()]
            return tags[:13]

        except Exception as e:
            logger.error("SEO tag generation failed: %s", e)
            return []

    def build_custom_prompt(self, product_info: dict[str, Any], style: str, mood: str) -> str:
        """Build a fully custom prompt from structured product info."""
        if not self.available:
            return ""

        try:
            client = self._get_client()

            request = (
                f"Create a detailed, photorealistic image generation prompt for:\n"
                f"Product: {product_info.get('title', 'product')}\n"
                f"Color: {product_info.get('color', 'neutral')}\n"
                f"Material: {product_info.get('description', 'fabric')}\n"
                f"Style: {style}\n"
                f"Mood: {mood}\n\n"
                f"Output ONLY the prompt text, no explanation."
            )

            response = client.models.generate_content(
                model=self.model,
                contents=request,
                config={
                    "system_instruction": ENHANCEMENT_SYSTEM_PROMPT,
                    "temperature": 0.8,
                    "max_output_tokens": 400,
                },
            )

            return response.text.strip()

        except Exception as e:
            logger.error("Custom prompt build failed: %s", e)
            return ""
