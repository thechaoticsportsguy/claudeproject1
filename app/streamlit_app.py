"""Pro-grade Streamlit UI for Etsy product image generation.

Features:
- Multi-tab interface (Generate, Gallery, AI Studio, Analytics, Prompt History)
- Google Gemini AI prompt enhancement and style analysis
- Three image providers: Replicate (Flux), OpenAI (DALL-E), Google Gemini (Imagen)
- Real-time analytics dashboard with Plotly charts
- Image comparison tools
- Batch processing with live progress
- Post-processing pipeline (color grading, sharpening, borders)
- Prompt history and favorites
- SEO tag generation
- Export in multiple formats
"""

from __future__ import annotations

import io
import json
import logging
import os
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.analytics import SessionAnalytics
from core.gemini_enhancer import GeminiEnhancer
from core.models import (
    Listing,
    PROVIDER_COST_ESTIMATES,
    Variation,
)
from core.postprocess import ETSY_SIZES, create_comparison_image
from core.prompt_builder import build_prompt, build_mixed_style_prompt
from core.providers import get_provider
from prompts.templates import (
    BACKGROUNDS,
    LIGHTING_PRESETS,
    POSES,
    STYLE_MODIFIERS,
    TEMPLATES,
)

load_dotenv()
logging.basicConfig(level=logging.INFO)

# ============================================================================
# Page config and custom CSS
# ============================================================================

st.set_page_config(
    page_title="Etsy Image Generator Pro",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 24px;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
    }
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #667eea22, #764ba222);
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 12px;
    }
    .stExpander { border-radius: 10px; }
    section[data-testid="stSidebar"] > div { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Session state initialization
# ============================================================================


def init_session_state():
    defaults = {
        "listings": [],
        "all_results": [],
        "generation_history": [],
        "prompt_history": [],
        "favorite_prompts": [],
        "analytics": SessionAnalytics(),
        "enhancer": None,
        "gemini_key": "",
        "comparison_images": [],
        "comparison_labels": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()

# ============================================================================
# Sidebar: Global Configuration
# ============================================================================

with st.sidebar:
    st.markdown("### Configuration")

    provider_name = st.selectbox(
        "Image Provider",
        options=["gemini", "replicate", "openai"],
        index=0,
        help="Gemini (Google Imagen), Replicate (Flux), or OpenAI (DALL-E)",
    )

    # Provider-specific API key input (manual entry overrides .env)
    if provider_name == "gemini":
        api_key = st.text_input(
            "Gemini API Key",
            value="",
            type="password",
            help="Optional: leave blank to use GEMINI_API_KEY or GOOGLE_API_KEY from your environment/.env.",
        )
        if api_key:
            st.caption("Using Gemini key from sidebar input.")
        elif os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
            st.caption("Using Gemini key from environment (.env).")
    elif provider_name == "replicate":
        api_key = st.text_input(
            "Replicate API Token",
            value="",
            type="password",
            help="Optional: leave blank to use REPLICATE_API_TOKEN from your environment/.env.",
        )
        if api_key:
            st.caption("Using Replicate token from sidebar input.")
        elif os.environ.get("REPLICATE_API_TOKEN"):
            st.caption("Using Replicate token from environment (.env).")
    else:
        api_key = st.text_input(
            "OpenAI API Key",
            value="",
            type="password",
            help="Optional: leave blank to use OPENAI_API_KEY from your environment/.env.",
        )
        if api_key:
            st.caption("Using OpenAI key from sidebar input.")
        elif os.environ.get("OPENAI_API_KEY"):
            st.caption("Using OpenAI key from environment (.env).")

    # Gemini enhancer key (always available)
    st.divider()
    st.markdown("##### AI Prompt Enhancer")
    gemini_enhancer_key = st.text_input(
        "Gemini Enhancer Key",
        value="",
        type="password",
        help="Optional: leave blank to use GEMINI_API_KEY or GOOGLE_API_KEY from your environment/.env.",
    )
    enable_enhancement = st.toggle("Enable AI Enhancement", value=True)

    # Initialize enhancer (explicit key overrides environment)
    enhancer_key = gemini_enhancer_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
    if enhancer_key:
        st.session_state["enhancer"] = GeminiEnhancer(api_key=enhancer_key)
        st.session_state["gemini_key"] = enhancer_key
    else:
        st.session_state["enhancer"] = GeminiEnhancer(api_key="")

    st.divider()

    # Template selection
    template_name = st.selectbox(
        "Prompt Template",
        options=list(TEMPLATES.keys()),
        index=0,
        format_func=lambda x: x.replace("_", " ").title(),
    )

    # Output size
    etsy_size = st.selectbox(
        "Output Size",
        options=list(ETSY_SIZES.keys()),
        index=0,
        format_func=lambda k: f"{k.replace('_', ' ').title()} ({ETSY_SIZES[k][0]}x{ETSY_SIZES[k][1]})",
    )

    st.divider()
    st.markdown("##### Visual Settings")

    # Color grading
    color_grade = st.selectbox(
        "Color Grade",
        options=["none", "warm", "cool", "vintage", "high_contrast", "soft_glow", "matte"],
        index=0,
        format_func=lambda x: x.replace("_", " ").title(),
    )

    # Lighting preset
    lighting_preset = st.selectbox(
        "Lighting",
        options=["(auto)"] + list(LIGHTING_PRESETS.keys()),
        index=0,
    )
    effective_lighting = None if lighting_preset == "(auto)" else lighting_preset

    # Style modifier
    style_modifier = st.selectbox(
        "Style Modifier",
        options=["(none)"] + list(STYLE_MODIFIERS.keys()),
        index=0,
    )
    effective_style_modifier = None if style_modifier == "(none)" else style_modifier

    # Post-processing sliders
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        sharpen_val = st.slider("Sharpen", 0.5, 3.0, 1.0, 0.1)
    with col_s2:
        brightness_val = st.slider("Brightness", 0.5, 1.5, 1.0, 0.05)
    contrast_val = st.slider("Contrast", 0.5, 1.5, 1.0, 0.05)

    # Watermark
    watermark = st.checkbox("Add watermark", value=False)
    watermark_text = ""
    if watermark:
        watermark_text = st.text_input("Watermark text", value="PREVIEW")

    st.divider()
    st.markdown("##### Variations")

    num_variations = st.slider("Variations per listing", 1, 12, 4)

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

    # Cost estimate
    st.divider()
    estimated_images = num_variations * max(len(st.session_state["listings"]), 1)
    cost_per_image = PROVIDER_COST_ESTIMATES.get(provider_name, 0.01)
    st.metric(
        "Est. Cost",
        f"${estimated_images * cost_per_image:.3f}",
        f"{estimated_images} images",
    )


# ============================================================================
# Main area — Tabs
# ============================================================================

st.title("Etsy Image Generator Pro")

tab_generate, tab_gallery, tab_ai_studio, tab_analytics, tab_history = st.tabs([
    "Generate",
    "Gallery",
    "AI Studio",
    "Analytics",
    "Prompt History",
])

# ============================================================================
# TAB: Generate
# ============================================================================

with tab_generate:
    st.header("Upload Listings")

    input_method = st.radio(
        "Input method",
        options=["Upload CSV", "Upload JSON", "Manual entry"],
        horizontal=True,
    )

    listings: list[Listing] = []

    if input_method == "Upload CSV":
        uploaded = st.file_uploader("Upload listings.csv", type=["csv"])
        if uploaded is not None:
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
            mcol1, mcol2 = st.columns(2)
            with mcol1:
                sku = st.text_input("SKU", value="PJ-001")
                title = st.text_input("Title", value="Satin Bridal Party Pajama Set")
                description = st.text_area("Description", value="Luxurious satin pajama set")
            with mcol2:
                color = st.text_input("Color", value="dark brown")
                piping_color = st.text_input("Piping Color", value="white")
                category = st.text_input("Category", value="sleepwear")
            base_image_file = st.file_uploader(
                "Base product image (optional, for img2img mode)", type=["png", "jpg", "jpeg"]
            )

            submitted = st.form_submit_button("Add Listing", type="primary")
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
                    category=category,
                )]

    if listings:
        st.session_state["listings"] = listings

    # Show listing preview
    current_listings = st.session_state["listings"]
    if current_listings:
        st.subheader("Listing Preview")
        preview_cols = st.columns(min(3, len(current_listings)))
        for i, listing in enumerate(current_listings):
            with preview_cols[i % len(preview_cols)]:
                st.markdown(f"**{listing.sku}**: {listing.title}")
                st.caption(f"Color: {listing.color} | Piping: {listing.piping_color}")
                if listing.base_image_path:
                    st.caption(f"Base image: {listing.base_image_path}")

    # Prompt preview
    if current_listings:
        with st.expander("Preview Generated Prompts", expanded=False):
            for listing in current_listings[:3]:
                sample_var = Variation(
                    pose=selected_poses[0] if selected_poses else "standing",
                    background_style=selected_backgrounds[0] if selected_backgrounds else "clean studio",
                )
                sample_prompt = build_prompt(
                    listing, sample_var,
                    template_name=template_name,
                    lighting=effective_lighting,
                    style_modifier=effective_style_modifier,
                )
                st.markdown(f"**{listing.sku}:**")
                st.code(sample_prompt, language=None)

                # Show enhanced version if available
                enhancer = st.session_state.get("enhancer")
                if enable_enhancement and enhancer and enhancer.available:
                    if st.button(f"Enhance prompt for {listing.sku}", key=f"enhance_{listing.sku}"):
                        with st.spinner("Enhancing with Gemini AI..."):
                            enhanced = enhancer.enhance_prompt(sample_prompt)
                            st.code(enhanced, language=None)

    # ---------------------------------------------------------------------------
    # Generate button
    # ---------------------------------------------------------------------------

    st.divider()
    st.header("Generate Images")

    gen_col1, gen_col2 = st.columns([3, 1])
    with gen_col2:
        total_imgs = len(selected_poses) * len(selected_backgrounds) * max(len(current_listings), 1)
        total_imgs = min(total_imgs, num_variations * max(len(current_listings), 1))
        st.metric("Total Images", total_imgs)

    if st.button("Generate All Images", type="primary", disabled=not current_listings, use_container_width=True):
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
                if provider_name == "replicate":
                    provider_kwargs["api_token"] = api_key
                else:
                    provider_kwargs["api_key"] = api_key

            try:
                provider = get_provider(provider_name, **provider_kwargs)
            except ValueError as e:
                st.error(str(e))
                st.info(
                    "Tip: add your key in the sidebar or create a .env file with the required variable "
                    "(REPLICATE_API_TOKEN, OPENAI_API_KEY, GEMINI_API_KEY/GOOGLE_API_KEY)."
                )
                st.stop()

            # Init enhancer for pipeline
            enhancer = st.session_state.get("enhancer")
            analytics = st.session_state["analytics"]

            progress = st.progress(0, text="Starting generation...")
            status_container = st.empty()
            total = len(current_listings) * len(custom_variations)
            generated = 0

            all_results = []
            for listing in current_listings:
                for idx, var in enumerate(custom_variations, start=1):
                    status_container.info(
                        f"Generating **{listing.sku}** variation {idx}/{len(custom_variations)} "
                        f"(pose={var.pose}, bg={var.background_style}) via **{provider_name}**..."
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
                            enhance_prompts=enable_enhancement,
                            enhancer=enhancer,
                            lighting=effective_lighting,
                            style_modifier=effective_style_modifier,
                            color_grade=color_grade,
                            sharpen=sharpen_val,
                            brightness=brightness_val,
                            contrast=contrast_val,
                            analytics=analytics,
                        )
                        for r in results:
                            r.variation_index = generated + 1
                        all_results.extend(results)

                        # Track prompt history
                        for r in results:
                            st.session_state["prompt_history"].append({
                                "sku": r.sku,
                                "original": r.prompt,
                                "enhanced": r.enhanced_prompt,
                                "provider": r.provider,
                                "time": r.generation_time_s,
                                "timestamp": r.timestamp,
                            })

                    except Exception as e:
                        st.error(f"Error generating {listing.sku} v{idx}: {e}")
                        logging.exception("Generation error")

                    generated += 1
                    progress.progress(
                        generated / total,
                        text=f"Generated {generated}/{total} images...",
                    )

            progress.progress(1.0, text="Generation complete!")
            status_container.success(
                f"Successfully generated {len(all_results)} images! "
                f"Estimated cost: ${sum(r.cost_estimate for r in all_results):.3f}"
            )

            st.session_state["all_results"] = all_results
            st.session_state["generation_history"].extend(all_results)

            # Save metadata
            from core.models import GenerationMetadata
            metadata = GenerationMetadata(results=all_results)
            metadata.compute_totals()
            metadata.save(output_dir / "metadata.json")

            # ---------------------------------------------------------------------------
            # Preview grid
            # ---------------------------------------------------------------------------
            st.header("Preview")
            if all_results:
                preview_size = st.select_slider(
                    "Preview columns",
                    options=[2, 3, 4, 5, 6],
                    value=4,
                )
                cols = st.columns(preview_size)
                for i, result in enumerate(all_results):
                    img_path = Path(result.output_path)
                    if img_path.exists():
                        col = cols[i % preview_size]
                        with col:
                            st.image(str(img_path), caption=f"{result.sku} v{result.variation_index}")
                            st.caption(
                                f"{result.width}x{result.height} | "
                                f"{result.generation_time_s}s | "
                                f"${result.cost_estimate:.3f}"
                            )

            # ---------------------------------------------------------------------------
            # Download
            # ---------------------------------------------------------------------------
            st.header("Download")
            dl_col1, dl_col2 = st.columns(2)

            with dl_col1:
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
                    label="Download All (ZIP)",
                    data=zip_buf.getvalue(),
                    file_name="etsy_images.zip",
                    mime="application/zip",
                    use_container_width=True,
                )

            with dl_col2:
                meta_json = json.dumps(metadata.to_dict(), indent=2)
                st.download_button(
                    label="Download Metadata (JSON)",
                    data=meta_json,
                    file_name="generation_metadata.json",
                    mime="application/json",
                    use_container_width=True,
                )

    elif not current_listings:
        st.info("Upload or enter at least one listing to begin.")

# ============================================================================
# TAB: Gallery
# ============================================================================

with tab_gallery:
    st.header("Image Gallery")

    all_results = st.session_state.get("all_results", [])

    if not all_results:
        st.info("Generate some images first to see them here.")
    else:
        # Filters
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        with filter_col1:
            unique_skus = list({r.sku for r in all_results})
            filter_sku = st.multiselect("Filter by SKU", unique_skus, default=unique_skus)
        with filter_col2:
            unique_providers = list({r.provider for r in all_results if r.provider})
            if unique_providers:
                filter_provider = st.multiselect("Filter by Provider", unique_providers, default=unique_providers)
            else:
                filter_provider = []
        with filter_col3:
            gallery_cols = st.select_slider("Columns", options=[2, 3, 4, 5, 6], value=4)

        # Apply filters
        filtered = [
            r for r in all_results
            if r.sku in filter_sku and (not unique_providers or r.provider in filter_provider)
        ]

        # Gallery grid
        cols = st.columns(gallery_cols)
        for i, result in enumerate(filtered):
            img_path = Path(result.output_path)
            if img_path.exists():
                col = cols[i % gallery_cols]
                with col:
                    st.image(str(img_path), use_container_width=True)
                    st.caption(f"{result.sku} v{result.variation_index}")
                    with st.popover("Details"):
                        st.write(f"**Provider:** {result.provider}")
                        st.write(f"**Size:** {result.width}x{result.height}")
                        st.write(f"**Time:** {result.generation_time_s}s")
                        st.write(f"**Cost:** ${result.cost_estimate:.4f}")
                        st.write("**Prompt:**")
                        st.code(result.prompt, language=None)
                        if result.enhanced_prompt:
                            st.write("**Enhanced:**")
                            st.code(result.enhanced_prompt, language=None)

        # Comparison tool
        st.divider()
        st.subheader("Image Comparison")
        compare_options = [
            f"{r.sku} v{r.variation_index}" for r in filtered
            if Path(r.output_path).exists()
        ]
        selected_for_compare = st.multiselect(
            "Select images to compare side-by-side",
            options=compare_options,
            max_selections=4,
        )

        if len(selected_for_compare) >= 2:
            from PIL import Image as PILImage
            compare_images = []
            compare_labels = []
            for label in selected_for_compare:
                for r in filtered:
                    if f"{r.sku} v{r.variation_index}" == label:
                        p = Path(r.output_path)
                        if p.exists():
                            compare_images.append(PILImage.open(str(p)))
                            compare_labels.append(label)
                        break

            if len(compare_images) >= 2:
                comparison = create_comparison_image(compare_images, compare_labels)
                st.image(comparison, use_container_width=True, caption="Side-by-side Comparison")

                buf = io.BytesIO()
                comparison.save(buf, format="PNG")
                st.download_button(
                    "Download comparison",
                    data=buf.getvalue(),
                    file_name="comparison.png",
                    mime="image/png",
                )

# ============================================================================
# TAB: AI Studio
# ============================================================================

with tab_ai_studio:
    st.header("AI Prompt Studio")
    st.caption("Powered by Google Gemini")

    enhancer = st.session_state.get("enhancer")
    if not enhancer or not enhancer.available:
        st.warning("Enter a Gemini API key in the sidebar to use AI features.")
    else:
        ai_tab1, ai_tab2, ai_tab3, ai_tab4 = st.tabs([
            "Prompt Enhancer",
            "Style Analyzer",
            "Prompt Critique",
            "SEO Tags",
        ])

        # --- Prompt Enhancer ---
        with ai_tab1:
            st.subheader("AI Prompt Enhancement")
            st.caption("Paste any prompt and get a professionally enhanced version")

            raw_prompt = st.text_area(
                "Enter a basic prompt",
                value="dark brown satin pajama set, standing pose, bedroom background, product photo",
                height=100,
            )
            style_hints = st.text_input(
                "Style direction (optional)",
                placeholder="e.g., moody editorial, bright and airy, vintage film look",
            )

            if st.button("Enhance Prompt", type="primary", key="enhance_btn"):
                with st.spinner("Gemini is enhancing your prompt..."):
                    enhanced = enhancer.enhance_prompt(raw_prompt, style_hints=style_hints)

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Original:**")
                    st.code(raw_prompt, language=None)
                    st.metric("Length", f"{len(raw_prompt)} chars")
                with col2:
                    st.markdown("**Enhanced:**")
                    st.code(enhanced, language=None)
                    st.metric("Length", f"{len(enhanced)} chars")

                if st.button("Save to favorites", key="fav_enhanced"):
                    st.session_state["favorite_prompts"].append(enhanced)
                    st.success("Saved!")

        # --- Style Analyzer ---
        with ai_tab2:
            st.subheader("Product Style Analyzer")
            st.caption("AI analyzes your product and suggests optimal photography styles")

            sa_col1, sa_col2 = st.columns(2)
            with sa_col1:
                sa_title = st.text_input("Product title", value="Satin Bridal Party Pajama Set", key="sa_title")
                sa_desc = st.text_area("Description", value="Luxurious satin pajama set for bridal parties", key="sa_desc")
            with sa_col2:
                sa_color = st.text_input("Color", value="dark brown", key="sa_color")

            if st.button("Analyze Styles", type="primary", key="analyze_btn"):
                with st.spinner("Analyzing product..."):
                    analysis = enhancer.analyze_styles(sa_title, sa_desc, sa_color)
                st.markdown(analysis)

        # --- Prompt Critique ---
        with ai_tab3:
            st.subheader("Prompt Quality Critique")
            st.caption("Get AI feedback on your prompt's quality and suggestions for improvement")

            critique_prompt = st.text_area(
                "Prompt to critique",
                height=100,
                key="critique_input",
                placeholder="Paste any image generation prompt here...",
            )

            if st.button("Critique Prompt", type="primary", key="critique_btn"):
                if critique_prompt:
                    with st.spinner("Analyzing prompt quality..."):
                        critique = enhancer.critique_prompt(critique_prompt)
                    st.markdown(critique)
                else:
                    st.warning("Enter a prompt to critique.")

        # --- SEO Tags ---
        with ai_tab4:
            st.subheader("Etsy SEO Tag Generator")
            st.caption("Generate optimized Etsy tags for your listings")

            seo_col1, seo_col2 = st.columns(2)
            with seo_col1:
                seo_title = st.text_input("Product title", key="seo_title", value="Satin Bridal Party Pajama Set")
            with seo_col2:
                seo_desc = st.text_input("Description", key="seo_desc", value="Luxurious satin pajama set")

            if st.button("Generate SEO Tags", type="primary", key="seo_btn"):
                with st.spinner("Generating tags..."):
                    tags = enhancer.generate_seo_tags(seo_title, seo_desc)
                if tags:
                    st.success(f"Generated {len(tags)} tags:")
                    tag_display = " | ".join([f"`{t}`" for t in tags])
                    st.markdown(tag_display)

                    # Copy-friendly format
                    st.text_area(
                        "Copy-paste format (comma-separated)",
                        value=", ".join(tags),
                        height=80,
                    )
                else:
                    st.warning("No tags generated. Check your API key.")

# ============================================================================
# TAB: Analytics
# ============================================================================

with tab_analytics:
    st.header("Generation Analytics")

    analytics: SessionAnalytics = st.session_state["analytics"]
    summary = analytics.get_summary()

    if analytics.total_images == 0:
        st.info("Generate some images to see analytics.")
    else:
        # KPI metrics row
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Images", summary["total_images"])
        kpi2.metric("Total Cost", f"${summary['total_cost']:.4f}")
        kpi3.metric("Avg Time/Image", f"{summary['avg_time_per_image_s']:.1f}s")
        kpi4.metric("Errors", summary["errors"])

        st.divider()

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.subheader("Provider Usage")
            if summary["providers"]:
                import pandas as pd
                provider_data = []
                for pname, pstats in summary["providers"].items():
                    provider_data.append({
                        "Provider": pname,
                        "Images": pstats["count"],
                        "Avg Time (s)": pstats["avg_time_s"],
                        "Cost ($)": pstats["total_cost"],
                    })
                df = pd.DataFrame(provider_data)
                st.dataframe(df, use_container_width=True, hide_index=True)

                try:
                    import plotly.express as px
                    fig = px.pie(
                        df, values="Images", names="Provider",
                        title="Images by Provider",
                        hole=0.4,
                    )
                    fig.update_layout(height=300, margin=dict(t=40, b=20, l=20, r=20))
                    st.plotly_chart(fig, use_container_width=True)
                except ImportError:
                    st.bar_chart(df.set_index("Provider")["Images"])

        with chart_col2:
            st.subheader("SKU Distribution")
            if summary["skus_processed"]:
                import pandas as pd
                sku_df = pd.DataFrame(
                    list(summary["skus_processed"].items()),
                    columns=["SKU", "Images"],
                )
                try:
                    import plotly.express as px
                    fig = px.bar(
                        sku_df, x="SKU", y="Images",
                        title="Images per SKU",
                        color="Images",
                        color_continuous_scale="viridis",
                    )
                    fig.update_layout(height=300, margin=dict(t=40, b=20, l=20, r=20))
                    st.plotly_chart(fig, use_container_width=True)
                except ImportError:
                    st.bar_chart(sku_df.set_index("SKU"))

        # Generation timeline
        log_records = analytics.to_dataframe_records()
        if log_records:
            st.subheader("Generation Timeline")
            import pandas as pd
            log_df = pd.DataFrame(log_records)
            try:
                import plotly.express as px
                fig = px.scatter(
                    log_df, x=log_df.index, y="time_s",
                    color="provider", size="cost",
                    hover_data=["sku", "template"],
                    title="Generation Time per Image",
                    labels={"index": "Image #", "time_s": "Time (s)"},
                )
                fig.update_layout(height=350, margin=dict(t=40, b=20, l=20, r=20))
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.line_chart(log_df["time_s"])

        # Template usage
        if summary["templates_used"]:
            st.subheader("Template Usage")
            import pandas as pd
            tpl_df = pd.DataFrame(
                list(summary["templates_used"].items()),
                columns=["Template", "Count"],
            )
            st.dataframe(tpl_df, use_container_width=True, hide_index=True)

        # Export analytics
        st.divider()
        st.download_button(
            "Export Analytics (JSON)",
            data=json.dumps(summary, indent=2),
            file_name="analytics.json",
            mime="application/json",
        )

# ============================================================================
# TAB: Prompt History
# ============================================================================

with tab_history:
    st.header("Prompt History")

    prompt_history = st.session_state.get("prompt_history", [])
    favorite_prompts = st.session_state.get("favorite_prompts", [])

    hist_tab1, hist_tab2 = st.tabs(["All Prompts", "Favorites"])

    with hist_tab1:
        if not prompt_history:
            st.info("Generate images to build prompt history.")
        else:
            for i, entry in enumerate(reversed(prompt_history)):
                with st.expander(
                    f"{entry['sku']} | {entry['provider']} | {entry['time']:.1f}s",
                    expanded=(i == 0),
                ):
                    st.markdown("**Original prompt:**")
                    st.code(entry["original"], language=None)
                    if entry.get("enhanced"):
                        st.markdown("**Enhanced prompt:**")
                        st.code(entry["enhanced"], language=None)

                    ph_col1, ph_col2 = st.columns(2)
                    with ph_col1:
                        if st.button("Add to favorites", key=f"fav_hist_{i}"):
                            fav_text = entry.get("enhanced") or entry["original"]
                            st.session_state["favorite_prompts"].append(fav_text)
                            st.success("Saved to favorites!")
                    with ph_col2:
                        st.caption(f"Provider: {entry['provider']}")

    with hist_tab2:
        if not favorite_prompts:
            st.info("No favorite prompts yet. Star prompts from history or AI Studio.")
        else:
            for i, fav in enumerate(favorite_prompts):
                with st.container():
                    st.code(fav, language=None)
                    if st.button("Remove", key=f"rm_fav_{i}"):
                        st.session_state["favorite_prompts"].pop(i)
                        st.rerun()
                    st.divider()
