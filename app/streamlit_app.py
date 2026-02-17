"""Streamlit UI for Etsy product image generation."""

from __future__ import annotations

import io
import logging
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.models import Listing, Variation
from core.postprocess import ETSY_SIZES
from core.providers import get_provider
from prompts.templates import BACKGROUNDS, POSES, TEMPLATES

load_dotenv()
logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Etsy Image Generator", layout="wide")
st.title("Etsy Product Image Generator")
st.markdown("Generate Etsy-ready product mockup images from your listings.")

# --- Sidebar: Configuration ---
with st.sidebar:
    st.header("Configuration")

    provider_name = st.selectbox(
        "Image Provider",
        options=["replicate", "openai"],
        index=0,
        help="Select which AI image generation service to use.",
    )

    api_key = st.text_input(
        "API Key (optional — uses .env if blank)",
        type="password",
        help="Override the API key from your .env file.",
    )

    template_name = st.selectbox(
        "Prompt Template",
        options=list(TEMPLATES.keys()),
        index=0,
    )

    etsy_size = st.selectbox(
        "Output Size",
        options=list(ETSY_SIZES.keys()),
        index=0,
        format_func=lambda k: f"{k} ({ETSY_SIZES[k][0]}x{ETSY_SIZES[k][1]})",
    )

    watermark = st.checkbox("Add watermark", value=False)
    watermark_text = ""
    if watermark:
        watermark_text = st.text_input("Watermark text", value="PREVIEW")

    st.divider()
    st.header("Variations")

    num_variations = st.slider("Number of variations per listing", 1, 8, 4)

    selected_poses = st.multiselect(
        "Poses",
        options=list(POSES.keys()),
        default=["standing", "sitting"],
    )

    selected_backgrounds = st.multiselect(
        "Backgrounds",
        options=list(BACKGROUNDS.keys()),
        default=["luxury bedroom", "clean studio"],
    )

# --- Main area: Input ---
st.header("1. Upload Listings")

input_method = st.radio(
    "Input method",
    options=["Upload CSV", "Upload JSON", "Manual entry"],
    horizontal=True,
)

listings: list[Listing] = []

if input_method == "Upload CSV":
    uploaded = st.file_uploader("Upload listings.csv", type=["csv"])
    if uploaded is not None:
        # Save to temp file for parsing
        tmp = Path(tempfile.mktemp(suffix=".csv"))
        tmp.write_bytes(uploaded.read())
        listings = Listing.from_csv(tmp)
        tmp.unlink()
        st.success(f"Loaded {len(listings)} listings from CSV.")

elif input_method == "Upload JSON":
    uploaded = st.file_uploader("Upload listings.json", type=["json"])
    if uploaded is not None:
        tmp = Path(tempfile.mktemp(suffix=".json"))
        tmp.write_bytes(uploaded.read())
        listings = Listing.from_json(tmp)
        tmp.unlink()
        st.success(f"Loaded {len(listings)} listings from JSON.")

elif input_method == "Manual entry":
    with st.form("manual_listing"):
        sku = st.text_input("SKU", value="PJ-001")
        title = st.text_input("Title", value="Satin Bridal Party Pajama Set")
        description = st.text_area("Description", value="Luxurious satin pajama set")
        color = st.text_input("Color", value="dark brown")
        piping_color = st.text_input("Piping Color", value="white")
        base_image_file = st.file_uploader(
            "Base product image (optional, for img2img mode)", type=["png", "jpg", "jpeg"]
        )

        submitted = st.form_submit_button("Add Listing")
        if submitted:
            base_image_path = None
            if base_image_file is not None:
                base_tmp = Path(tempfile.mktemp(suffix=".png"))
                base_tmp.write_bytes(base_image_file.read())
                base_image_path = str(base_tmp)

            listings = [Listing(
                sku=sku,
                title=title,
                description=description,
                color=color,
                piping_color=piping_color,
                base_image_path=base_image_path,
            )]

# Show listing preview
if listings:
    st.subheader("Listing Preview")
    for listing in listings:
        with st.expander(f"{listing.sku}: {listing.title}"):
            st.write(f"**Color:** {listing.color}")
            st.write(f"**Piping:** {listing.piping_color}")
            st.write(f"**Description:** {listing.description}")
            if listing.base_image_path:
                st.write(f"**Base image:** {listing.base_image_path}")

# --- Generate ---
st.header("2. Generate Images")

if st.button("Generate", type="primary", disabled=not listings):
    # Build custom variations from selected poses and backgrounds
    custom_variations = [
        Variation(pose=pose, background_style=bg)
        for pose in selected_poses
        for bg in selected_backgrounds
    ][:num_variations]

    if not custom_variations:
        st.error("Select at least one pose and one background.")
    else:
        output_dir = Path("outputs")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)

        # Init provider
        provider_kwargs = {}
        if api_key:
            key_param = "api_token" if provider_name == "replicate" else "api_key"
            provider_kwargs[key_param] = api_key

        try:
            provider = get_provider(provider_name, **provider_kwargs)
        except ValueError as e:
            st.error(str(e))
            st.stop()

        progress = st.progress(0)
        status = st.empty()
        total = len(listings) * len(custom_variations)
        generated = 0

        all_results = []
        for listing in listings:
            for idx, var in enumerate(custom_variations, start=1):
                status.text(
                    f"Generating {listing.sku} variation {idx}/{len(custom_variations)} "
                    f"(pose={var.pose}, bg={var.background_style})..."
                )
                try:
                    from core.pipeline import generate_for_listing

                    results = generate_for_listing(
                        listing=listing,
                        provider=provider,
                        variations=[var],
                        output_dir=output_dir,
                        etsy_size=etsy_size,
                        watermark=watermark,
                        template_name=template_name,
                    )
                    # Fix variation index for correct file naming
                    for r in results:
                        r.variation_index = generated + 1
                    all_results.extend(results)
                except Exception as e:
                    st.error(f"Error generating {listing.sku} v{idx}: {e}")
                    logging.exception("Generation error")

                generated += 1
                progress.progress(generated / total)

        progress.progress(1.0)
        status.text("Generation complete!")

        # Save metadata
        from core.models import GenerationMetadata

        metadata = GenerationMetadata(results=all_results)
        metadata.save(output_dir / "metadata.json")

        # --- Preview grid ---
        st.header("3. Preview")
        cols = st.columns(min(4, len(all_results)))
        for i, result in enumerate(all_results):
            img_path = Path(result.output_path)
            if img_path.exists():
                col = cols[i % len(cols)]
                with col:
                    st.image(str(img_path), caption=f"{result.sku} v{result.variation_index}")
                    st.caption(f"{result.width}x{result.height}")

        # --- Download zip ---
        st.header("4. Download")
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for result in all_results:
                p = Path(result.output_path)
                if p.exists():
                    zf.write(p, p.name)
            meta_path = output_dir / "metadata.json"
            if meta_path.exists():
                zf.write(meta_path, "metadata.json")

        st.download_button(
            label="Download all images as ZIP",
            data=zip_buf.getvalue(),
            file_name="etsy_images.zip",
            mime="application/zip",
        )

elif not listings:
    st.info("Upload or enter at least one listing to begin.")
