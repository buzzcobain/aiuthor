"""Configuration and genre/audience benchmarks for Aiuthor.

Supports:
- Environment variable overrides with sensible defaults.
- Lightweight .env file auto-discovery.
- Industry-standard publishing benchmarks by genre.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List, Optional


def load_env_file():
    """Lightweight zero-dependency .env loader."""
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent.parent / ".env",
        Path.home() / ".aiuthor" / ".env"
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            try:
                for line in p.read_text().splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k and k not in os.environ:
                        os.environ[k] = v
                break
            except Exception:
                pass


# Execute discovery at import time
load_env_file()

DEFAULT_OLLAMA_HOST = os.getenv("AIUTHOR_OLLAMA_HOST", os.getenv("OLLAMA_HOST", "http://localhost:11434"))
DEFAULT_MODEL_MANAGER_URL = os.getenv("AIUTHOR_MODEL_MANAGER_URL", "http://127.0.0.1:8990")

# Industry standard book length guidelines by genre & audience
GENRE_SPECS: Dict[str, Dict[str, Any]] = {
    "ya_fantasy": {
        "display": "Young Adult Fantasy / Sci-Fi",
        "min_words": 65000,
        "max_words": 90000,
        "target_words": 75000,
        "chapters": 25,
        "chapter_words": 3000,
        "subplots_required": 2,
        "description": "Fast-paced, high stakes, emotional core, coming-of-age tension."
    },
    "epic_fantasy": {
        "display": "Epic / High Fantasy or Space Opera",
        "min_words": 90000,
        "max_words": 130000,
        "target_words": 105000,
        "chapters": 35,
        "chapter_words": 3000,
        "subplots_required": 3,
        "description": "Expansive world-building, multiple POV/factions, political or cosmic conflict."
    },
    "thriller": {
        "display": "Commercial Thriller / Mystery / Crime",
        "min_words": 70000,
        "max_words": 85000,
        "target_words": 75000,
        "chapters": 35,
        "chapter_words": 2100,
        "subplots_required": 2,
        "description": "Tight chapters, cliffhangers, urgent pacing, ticking clock, shifting suspicions."
    },
    "literary_fiction": {
        "display": "Literary / Contemporary Fiction",
        "min_words": 70000,
        "max_words": 95000,
        "target_words": 80000,
        "chapters": 24,
        "chapter_words": 3300,
        "subplots_required": 2,
        "description": "Character-driven, deep psychological interiority, thematic depth, unique voice."
    },
    "romance": {
        "display": "Contemporary / Paranormal Romance",
        "min_words": 55000,
        "max_words": 80000,
        "target_words": 65000,
        "chapters": 22,
        "chapter_words": 2900,
        "subplots_required": 1,
        "description": "Dual emotional arcs, external complications, relationship pacing, satisfying resolution."
    },
    "horror": {
        "display": "Horror / Psychological Suspense",
        "min_words": 65000,
        "max_words": 85000,
        "target_words": 72000,
        "chapters": 26,
        "chapter_words": 2700,
        "subplots_required": 2,
        "description": "Escalating dread, sensory atmosphere, psychological unraveling, visceral stakes."
    },
    "nonfiction_business": {
        "display": "Non-Fiction / Business / Self-Help",
        "min_words": 40000,
        "max_words": 60000,
        "target_words": 48000,
        "chapters": 12,
        "chapter_words": 4000,
        "subplots_required": 0,
        "description": "Thesis-driven, case studies, actionable frameworks, clear takeaways."
    },
    "middle_grade": {
        "display": "Middle Grade (Ages 8-12)",
        "min_words": 30000,
        "max_words": 50000,
        "target_words": 38000,
        "chapters": 20,
        "chapter_words": 1900,
        "subplots_required": 1,
        "description": "Direct voice, relatable protagonist struggles, swift plot momentum."
    },
    "novella": {
        "display": "Novella",
        "min_words": 20000,
        "max_words": 40000,
        "target_words": 30000,
        "chapters": 12,
        "chapter_words": 2500,
        "subplots_required": 1,
        "description": "Focused scope, singular main conflict, minimal extraneous cast."
    }
}


@dataclass
class ModelConfig:
    reasoning_model: str
    drafting_model: str
    critic_model: str
    rewriter_model: str
    temperature_reasoning: float = 0.60
    temperature_drafting: float = 0.75
    temperature_critic: float = 0.30
    temperature_rewriter: float = 0.70


def get_model_config() -> ModelConfig:
    """Instantiates ModelConfig populated from environment variables with defaults."""
    return ModelConfig(
        reasoning_model=os.getenv(
            "AIUTHOR_REASONING_MODEL",
            os.getenv("BOOK_FOUNDRY_REASONING_MODEL", "deepseek-r1:32b")
        ),
        drafting_model=os.getenv(
            "AIUTHOR_DRAFTING_MODEL",
            os.getenv("BOOK_FOUNDRY_DRAFTING_MODEL", "qwen2.5-coder:32b")
        ),
        critic_model=os.getenv(
            "AIUTHOR_CRITIC_MODEL",
            os.getenv("BOOK_FOUNDRY_CRITIC_MODEL", "deepseek-r1:32b")
        ),
        rewriter_model=os.getenv(
            "AIUTHOR_REWRITER_MODEL",
            os.getenv("BOOK_FOUNDRY_REWRITER_MODEL", "qwen2.5-coder:32b")
        ),
        temperature_reasoning=float(os.getenv("AIUTHOR_TEMPERATURE_REASONING", "0.60")),
        temperature_drafting=float(os.getenv("AIUTHOR_TEMPERATURE_DRAFTING", "0.75")),
        temperature_critic=float(os.getenv("AIUTHOR_TEMPERATURE_CRITIC", "0.30")),
        temperature_rewriter=float(os.getenv("AIUTHOR_TEMPERATURE_REWRITER", "0.70"))
    )
