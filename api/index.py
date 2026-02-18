"""Minimal Vercel serverless entrypoint for this repository.

This project is a Streamlit application, which is not a native Vercel runtime target.
When deployed to Vercel, route all traffic here so users get a clear response instead
of a generic NOT_FOUND page.
"""

from __future__ import annotations

import json


def handler(request):
    """Vercel Python serverless function handler."""
    body = {
        "ok": True,
        "project": "etsy-image-generator",
        "message": (
            "This repository contains a Streamlit app (app/streamlit_app.py). "
            "Vercel does not run Streamlit's long-lived app server directly. "
            "Deploy the Streamlit app on Streamlit Community Cloud/Render/Railway, "
            "or convert the app to API routes for Vercel."
        ),
        "streamlit_entrypoint": "app/streamlit_app.py",
    }

    return {
        "statusCode": 200,
        "headers": {"content-type": "application/json"},
        "body": json.dumps(body),
    }
