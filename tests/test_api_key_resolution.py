from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.gemini_enhancer import GeminiEnhancer
from core.providers import resolve_api_key


def test_resolve_api_key_prefers_explicit(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-value")
    assert resolve_api_key(" explicit ", "OPENAI_API_KEY") == "explicit"


def test_resolve_api_key_falls_back_to_first_non_empty_env(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "google-value")
    assert resolve_api_key(None, "GEMINI_API_KEY", "GOOGLE_API_KEY") == "google-value"


def test_gemini_enhancer_supports_google_api_key_env(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "google-value")
    enhancer = GeminiEnhancer(api_key=None)
    assert enhancer.available is True
