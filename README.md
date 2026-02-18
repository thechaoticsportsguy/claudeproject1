# Etsy Product Image Generator

Generate Etsy-ready product mockup images for satin pajama sets, bridal robes, and other shop listings. Supports text-to-image generation and image-to-image variation mode.

## Features

- **Multiple input sources**: CSV, JSON, or manual entry
- **Pluggable providers**: Replicate (FLUX/SDXL) as default, OpenAI (DALL-E) as alternative
- **Batch generation**: Generate N variations per listing with different poses and backgrounds
- **Image-to-image mode**: Upload a base product photo and generate background/pose variations while keeping the product consistent
- **Etsy-ready output**: Auto-resize/crop to standard Etsy sizes (2000x2000, 3000x2400, etc.)
- **Optional watermark**: Toggle subtle preview watermarks
- **Streamlit UI**: Upload files, configure settings, preview results, download as ZIP
- **Metadata tracking**: `outputs/metadata.json` maps SKU to prompts to file paths

## Project Structure

```
├── app/
│   └── streamlit_app.py      # Streamlit web UI
├── core/
│   ├── models.py              # Data models (Listing, GenerationResult, etc.)
│   ├── pipeline.py            # Batch generation orchestrator
│   ├── postprocess.py         # Resize, crop, watermark
│   ├── prompt_builder.py      # Converts listings to generation prompts
│   └── providers.py           # Provider interface + Replicate/OpenAI implementations
├── outputs/                   # Generated images saved here
├── prompts/
│   └── templates.py           # Reusable prompt templates
├── listings.csv               # Example CSV input
├── listings.json              # Example JSON input
├── .env.example               # API key configuration
├── Makefile                   # Common commands
└── pyproject.toml             # Project config and dependencies
```

## Setup

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Or using Make:

```bash
make install
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env and add your provider API keys (at least one)
```

Get API keys:
- Replicate: https://replicate.com/account/api-tokens
- OpenAI: https://platform.openai.com/api-keys
- Gemini: https://aistudio.google.com/apikey

### 3. Run the UI

```bash
make run
# or directly:
streamlit run app/streamlit_app.py
```

## Usage

### Via Streamlit UI


The Streamlit sidebar key fields are optional: if left empty, the app now automatically uses keys from your environment/.env file.

1. Open the app in your browser (default: http://localhost:8501)
2. **Sidebar**: Choose provider, prompt template, output size, and variation settings
3. **Upload**: Select CSV or JSON file, or enter a listing manually
4. **Generate**: Click "Generate" and watch progress
5. **Download**: Preview the grid and download all images as a ZIP

### Via Python

```python
from core.models import Listing
from core.providers import get_provider
from core.pipeline import run_batch

# Load listings
listings = Listing.from_csv("listings.csv")

# Initialize provider
provider = get_provider("replicate")

# Generate
metadata = run_batch(
    listings=listings,
    provider=provider,
    num_variations=4,
    output_dir="outputs",
    etsy_size="square",
)
```

### Image-to-Image Mode

To generate variations from an existing product photo, include a `base_image_path` column in your CSV/JSON pointing to the source image file. The pipeline will automatically switch to img2img mode, preserving the product appearance while changing the background and pose.

## Input Format

### CSV (`listings.csv`)

| Column | Required | Description |
|---|---|---|
| sku | Yes | Unique product identifier |
| title | Yes | Product title |
| description | No | Product description |
| color | No | Primary product color |
| piping_color | No | Accent/piping color |
| pose | No | Default pose (standing, sitting) |
| background_style | No | Default background |
| base_image_path | No | Path to base image for img2img mode |

### JSON (`listings.json`)

```json
{
  "listings": [
    {
      "sku": "PJ-001",
      "title": "Satin Bridal Party Pajama Set",
      "color": "dark brown",
      "piping_color": "white"
    }
  ]
}
```

## Output

Images are saved to `outputs/` as:

```
outputs/PJ-001_v1.png
outputs/PJ-001_v2.png
outputs/PJ-001_v3.png
outputs/PJ-001_v4.png
outputs/metadata.json
```

`metadata.json` contains the full mapping of SKU to prompts to output file paths.

## Adding a New Provider

Implement the `ImageProvider` interface in `core/providers.py`:

```python
class MyProvider(ImageProvider):
    def generate(self, prompt: str, width: int = 1024, height: int = 1024) -> Image.Image:
        # Your implementation here
        ...

    def img2img(self, prompt: str, base_image: Image.Image, width: int = 1024, height: int = 1024) -> Image.Image:
        # Your implementation here
        ...
```

Then register it in the `get_provider` factory function.

## Requirements

- Python 3.11+
- Replicate API token (for default provider)
