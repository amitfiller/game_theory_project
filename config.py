"""
config.py
─────────
Centralised configuration for the game theory negotiation project.

API Key Security:
  NEVER hardcode your GEMINI_API_KEY in this file or anywhere in the codebase.
  Store it exclusively in a .env file at the project root (already in .gitignore).
  The .env file format:
      GEMINI_API_KEY=your_key_here

  Install dependency once:  pip install python-dotenv
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the project root (works regardless of the working directory)
_PROJECT_ROOT = Path(__file__).parent
load_dotenv(_PROJECT_ROOT / ".env")


# ── LLM settings ─────────────────────────────────────────────────────────────

# Default model: Gemini 1.5 Flash (free tier on Google AI Studio)
# Swap to "gemini-1.5-pro" for higher reasoning quality at higher cost
# Default to gemini-2.5-flash. (gemini-2.0-flash was de-listed by Google for new
# API users — it now returns 404 NOT_FOUND, which silently routed every turn to
# the heuristic fallback.) 2.5-flash is a "thinking" model that can occasionally
# truncate JSON on long workflow prompts, but the client's smart-retry now
# handles that: a content/EOF error triggers a temperature-bumped retry
# (0.0 -> LLM_RETRY_TEMPERATURE) that breaks the deterministic failure and
# recovers on the next attempt. So we get a live, available model AND stability.
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# The API key is loaded from .env — never set a fallback string here
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")


def require_api_key() -> str:
    """
    Return the Gemini API key, raising a clear error if it is missing.

    Call this in GeminiAgent.__init__ so the error surfaces immediately
    rather than at the first API call.
    """
    key = GEMINI_API_KEY
    if not key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set.\n"
            "Add it to your .env file:\n"
            "    GEMINI_API_KEY=your_key_here\n"
            "Get a free key at: https://aistudio.google.com/app/apikey"
        )
    return key


# ── Game settings ─────────────────────────────────────────────────────────────

MAX_ROUNDS: int = int(os.getenv("MAX_ROUNDS", "10"))

# Number of independent game runs per condition (with-workflow / without-workflow)
# 5-10 is enough for a course POC (paper uses more, but we're proving the concept)
RUNS_PER_CONDITION: int = int(os.getenv("RUNS_PER_CONDITION", "5"))

# Number of candidate valuation profiles in each agent's BeliefState prior
BELIEF_CANDIDATES: int = int(os.getenv("BELIEF_CANDIDATES", "30"))

# RNG seed for reproducibility (set to None for fully random games)
RANDOM_SEED: int | None = (
    int(os.getenv("RANDOM_SEED")) if os.getenv("RANDOM_SEED") else None
)


# ── LLM client settings ───────────────────────────────────────────────────────

# Max retries on transient API errors (429, 503)
LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "3"))

# Seconds between retries (exponential backoff base)
LLM_RETRY_BACKOFF: float = float(os.getenv("LLM_RETRY_BACKOFF", "2.0"))

# Generation temperature: 0.0 = deterministic, higher = more creative.
# Set to 0.0 for maximum reproducibility — game-theory agents should play
# rationally and consistently, and a fixed temperature makes live runs as
# replayable as the API allows (transcripts are still archived as ground truth).
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))

# Temperature used ONLY on a retry that follows a content/validation error.
# At temperature 0.0 the model is deterministic, so retrying a malformed or
# empty response just reproduces the same failure. Bumping the temperature on
# the retry breaks that deterministic loop and gives the model a fresh sample.
LLM_RETRY_TEMPERATURE: float = float(os.getenv("LLM_RETRY_TEMPERATURE", "0.4"))

# Minimum seconds between consecutive LIVE API calls (RPM safety throttle).
# Even with billing enabled, this protects against strict requests-per-minute
# spikes during a multi-game live run. 4.0s → max ~15 calls/min.
LLM_REQUEST_DELAY: float = float(os.getenv("LLM_REQUEST_DELAY", "4.0"))


# ── Architecture note: llm/gemini_client.py ──────────────────────────────────
#
# The upcoming GeminiClient will:
#   - Use google-generativeai SDK  (pip install google-generativeai)
#   - Call  genai.GenerativeModel(GEMINI_MODEL).generate_content(...)
#   - Enforce JSON output via Gemini's native Structured Outputs / response_schema
#     This is critical for operational stability — prevents JSON parse crashes
#     that would fail grading criterion #4.
#   - Wrap every call in retry logic using LLM_MAX_RETRIES / LLM_RETRY_BACKOFF
#   - Log every prompt + response to results/transcripts/
#
# Planned interface:
#   client = GeminiClient()
#   proposal_json = client.generate_proposal(system_prompt, history_text, schema)
#   response_json = client.generate_response(system_prompt, proposal_text, schema)
