"""Image generation provider interface and implementations."""

from __future__ import annotations

import io
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path

import httpx
from PIL import Image

logger = logging.getLogger(__name__)


class ImageProvider(ABC):
    """Base interface for image generation providers."""

    @abstractmethod
    def generate(self, prompt: str, width: int = 1024, height: int = 1024) -> Image.Image:
        """Generate an image from a text prompt."""
        ...

    @abstractmethod
    def img2img(
        self, prompt: str, base_image: Image.Image, width: int = 1024, height: int = 1024
    ) -> Image.Image:
        """Generate a variation from a base image and a prompt."""
        ...


class ReplicateProvider(ImageProvider):
    """Replicate API provider using FLUX or SDXL models."""

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
        """Generate an image using the Replicate API."""
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

        # Replicate returns a list of URLs or FileOutput objects
        image_url = output[0] if isinstance(output, list) else output
        image_url = str(image_url)

        with httpx.Client(timeout=120) as http:
            resp = http.get(image_url)
            resp.raise_for_status()

        return Image.open(io.BytesIO(resp.content)).convert("RGB")

    def img2img(
        self, prompt: str, base_image: Image.Image, width: int = 1024, height: int = 1024
    ) -> Image.Image:
        """Generate an image-to-image variation using Replicate."""
        import replicate

        client = replicate.Client(api_token=self.api_token)
        logger.info("Running img2img via Replicate")

        # Convert PIL image to bytes for upload
        buf = io.BytesIO()
        base_image.save(buf, format="PNG")
        buf.seek(0)

        # Use an img2img capable model
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
    """OpenAI DALL-E provider (stub - implement when needed)."""

    def __init__(self, api_key: str | None = None, model: str = "dall-e-3") -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY or pass api_key.")

    def generate(self, prompt: str, width: int = 1024, height: int = 1024) -> Image.Image:
        """Generate an image using the OpenAI Images API."""
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        logger.info("Generating image via OpenAI model=%s", self.model)

        # Map dimensions to supported DALL-E sizes
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
        """OpenAI image editing (requires mask for DALL-E 2, or use variations)."""
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
        """Map arbitrary dimensions to DALL-E supported sizes."""
        if width == height:
            return "1024x1024"
        elif width > height:
            return "1792x1024"
        else:
            return "1024x1792"


def get_provider(name: str, **kwargs) -> ImageProvider:
    """Factory function to get a provider by name."""
    providers: dict[str, type[ImageProvider]] = {
        "replicate": ReplicateProvider,
        "openai": OpenAIProvider,
    }
    if name not in providers:
        raise ValueError(f"Unknown provider: {name}. Available: {list(providers.keys())}")
    return providers[name](**kwargs)
