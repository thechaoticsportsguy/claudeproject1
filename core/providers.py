"""Image generation provider interface and implementations."""

from __future__ import annotations

import base64
import io
import logging
import os
import time
from abc import ABC, abstractmethod

import httpx
from PIL import Image

logger = logging.getLogger(__name__)


class ImageProvider(ABC):
    """Base interface for image generation providers."""

    provider_name: str = "base"

    @abstractmethod
    def generate(self, prompt: str, width: int = 1024, height: int = 1024) -> Image.Image:
        ...

    @abstractmethod
    def img2img(
        self, prompt: str, base_image: Image.Image, width: int = 1024, height: int = 1024
    ) -> Image.Image:
        ...

    def timed_generate(self, prompt: str, width: int = 1024, height: int = 1024) -> tuple[Image.Image, float]:
        start = time.time()
        img = self.generate(prompt, width, height)
        elapsed = time.time() - start
        return img, elapsed

    def timed_img2img(
        self, prompt: str, base_image: Image.Image, width: int = 1024, height: int = 1024
    ) -> tuple[Image.Image, float]:
        start = time.time()
        img = self.img2img(prompt, base_image, width, height)
        elapsed = time.time() - start
        return img, elapsed


class ReplicateProvider(ImageProvider):
    """Replicate API provider using FLUX or SDXL models."""

    provider_name = "replicate"

    def __init__(
        self,
        api_token: str | None = None,
        model: str = "black-forest-labs/flux-schnell",
    ) -> None:
        self.api_token = api_token or os.environ.get("REPLICATE_API_TOKEN", "")
        self.model = model
        if not self.api_token:
            raise ValueError(
                "Replicate API token is required. Set REPLICATE_API_TOKEN or pass api_token."
            )

    def generate(self, prompt: str, width: int = 1024, height: int = 1024) -> Image.Image:
        import replicate

        client = replicate.Client(api_token=self.api_token)
        logger.info("Generating image via Replicate model=%s", self.model)

        output = client.run(
            self.model,
            input={
                "prompt": prompt,
                "width": width,
                "height": height,
                "num_outputs": 1,
            },
        )

        image_url = output[0] if isinstance(output, list) else output
        image_url = str(image_url)

        with httpx.Client(timeout=120) as http:
            resp = http.get(image_url)
            resp.raise_for_status()

        return Image.open(io.BytesIO(resp.content)).convert("RGB")

    def img2img(
        self, prompt: str, base_image: Image.Image, width: int = 1024, height: int = 1024
    ) -> Image.Image:
        import replicate

        client = replicate.Client(api_token=self.api_token)
        logger.info("Running img2img via Replicate")

        buf = io.BytesIO()
        base_image.save(buf, format="PNG")
        buf.seek(0)

        img2img_model = "stability-ai/sdxl:7762fd07cf82c948c"
        output = client.run(
            img2img_model,
            input={
                "prompt": prompt,
                "image": buf,
                "width": width,
                "height": height,
                "prompt_strength": 0.7,
                "num_outputs": 1,
            },
        )

        image_url = output[0] if isinstance(output, list) else output
        image_url = str(image_url)

        with httpx.Client(timeout=120) as http:
            resp = http.get(image_url)
            resp.raise_for_status()

        return Image.open(io.BytesIO(resp.content)).convert("RGB")


class OpenAIProvider(ImageProvider):
    """OpenAI DALL-E provider."""

    provider_name = "openai"

    def __init__(self, api_key: str | None = None, model: str = "dall-e-3") -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY or pass api_key.")

    def generate(self, prompt: str, width: int = 1024, height: int = 1024) -> Image.Image:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        logger.info("Generating image via OpenAI model=%s", self.model)

        size = self._map_size(width, height)

        response = client.images.generate(
            model=self.model,
            prompt=prompt,
            n=1,
            size=size,
            response_format="url",
        )

        image_url = response.data[0].url
        with httpx.Client(timeout=120) as http:
            resp = http.get(image_url)
            resp.raise_for_status()

        return Image.open(io.BytesIO(resp.content)).convert("RGB")

    def img2img(
        self, prompt: str, base_image: Image.Image, width: int = 1024, height: int = 1024
    ) -> Image.Image:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        logger.info("Running img2img via OpenAI")

        buf = io.BytesIO()
        base_image.save(buf, format="PNG")
        buf.seek(0)

        size = self._map_size(width, height)

        response = client.images.edit(
            image=buf,
            prompt=prompt,
            n=1,
            size=size,
        )

        image_url = response.data[0].url
        with httpx.Client(timeout=120) as http:
            resp = http.get(image_url)
            resp.raise_for_status()

        return Image.open(io.BytesIO(resp.content)).convert("RGB")

    @staticmethod
    def _map_size(width: int, height: int) -> str:
        if width == height:
            return "1024x1024"
        elif width > height:
            return "1792x1024"
        else:
            return "1024x1792"


class GeminiProvider(ImageProvider):
    """Google Gemini Imagen provider for image generation."""

    provider_name = "gemini"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "imagen-3.0-generate-002",
    ) -> None:
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.model = model
        if not self.api_key:
            raise ValueError(
                "Gemini API key is required. Set GEMINI_API_KEY or pass api_key."
            )

    def _get_client(self):
        from google import genai
        return genai.Client(api_key=self.api_key)

    def generate(self, prompt: str, width: int = 1024, height: int = 1024) -> Image.Image:
        from google.genai import types

        client = self._get_client()
        logger.info("Generating image via Gemini Imagen model=%s", self.model)

        aspect = self._compute_aspect_ratio(width, height)

        response = client.models.generate_images(
            model=self.model,
            prompt=prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio=aspect,
                safety_filter_level="BLOCK_ONLY_HIGH",
            ),
        )

        if not response.generated_images:
            raise RuntimeError("Gemini returned no images — prompt may have been filtered.")

        image_bytes = response.generated_images[0].image.image_bytes
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    def img2img(
        self, prompt: str, base_image: Image.Image, width: int = 1024, height: int = 1024
    ) -> Image.Image:
        from google.genai import types

        client = self._get_client()
        logger.info("Running img2img via Gemini")

        buf = io.BytesIO()
        base_image.save(buf, format="PNG")
        raw_bytes = buf.getvalue()

        reference_image = types.RawReferenceImage(
            reference_id=1,
            reference_image=types.Image(image_bytes=raw_bytes),
        )

        aspect = self._compute_aspect_ratio(width, height)

        response = client.models.generate_images(
            model="imagen-3.0-capability-001",
            prompt=prompt,
            reference_images=[reference_image],
            config=types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio=aspect,
                safety_filter_level="BLOCK_ONLY_HIGH",
            ),
        )

        if not response.generated_images:
            raise RuntimeError("Gemini img2img returned no images.")

        image_bytes = response.generated_images[0].image.image_bytes
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    @staticmethod
    def _compute_aspect_ratio(width: int, height: int) -> str:
        ratio = width / height
        if abs(ratio - 1.0) < 0.15:
            return "1:1"
        elif ratio > 1.3:
            return "16:9"
        elif ratio > 1.05:
            return "4:3"
        elif ratio < 0.77:
            return "9:16"
        else:
            return "3:4"


def get_provider(name: str, **kwargs) -> ImageProvider:
    """Factory function to get a provider by name."""
    providers: dict[str, type[ImageProvider]] = {
        "replicate": ReplicateProvider,
        "openai": OpenAIProvider,
        "gemini": GeminiProvider,
    }
    if name not in providers:
        raise ValueError(f"Unknown provider: {name}. Available: {list(providers.keys())}")
    return providers[name](**kwargs)
