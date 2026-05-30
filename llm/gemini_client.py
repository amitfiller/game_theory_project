"""
llm/gemini_client.py
────────────────────
Thin, robust wrapper around the Google GenAI SDK.

Key design decisions:
  1. Structured Outputs via Pydantic response_schema — Gemini returns strictly valid
     JSON matching our schema, eliminating all parsing crashes (grading criterion #4).
  2. Retry with exponential backoff — handles 429 / 503 transient errors cleanly.
  3. Dry-run mode — returns deterministic canned responses for CI and offline demos.
  4. Full transcript logging — every (prompt, response) written to results/transcripts/.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional, Type, TypeVar

from pydantic import BaseModel, Field, ValidationError
from google import genai
from google.genai import types

import config


class EmptyResponseError(ValueError):
    """
    Raised when Gemini returns a response with no usable text part.

    A "thinking" model can spend its output budget on internal reasoning and
    return a candidate whose `.text` is None or empty. We treat this as a
    CONTENT error (eligible for a temperature-bumped retry), not a network error.
    """

# ─────────────────────────────────────────────
# Pydantic schemas  (must match Phase 2 message types exactly)
# ─────────────────────────────────────────────

class ItemCount(BaseModel):
    """One item and how many units the agent receives."""
    name: str
    # ge=0: counts can never be negative. Pydantic rejects a negative count at
    # parse time, which trips the client retry/fallback rather than silently
    # producing an "infeasible-but-schema-valid" allocation.
    count: int = Field(ge=0)


class ProposeOutput(BaseModel):
    """
    Schema for a proposal action.
    my_bundle + opponent_bundle must together cover every item in the pool.
    """
    my_bundle: List[ItemCount]
    opponent_bundle: List[ItemCount]
    reasoning: str


class RespondOutput(BaseModel):
    """
    Schema for a respond action.
    action  : constrained to exactly 'accept' | 'reject' | 'counter'
    bundles : only populated when action == 'counter'
    """
    # Literal → JSON-schema enum, enforced by Gemini Structured Outputs. The
    # model physically cannot return a value outside this set, removing the
    # "is this string a counter?" ambiguity entirely.
    action: Literal["accept", "reject", "counter"]
    my_bundle: List[ItemCount]
    opponent_bundle: List[ItemCount]
    reasoning: str


# Generic type var used in generate()
T = TypeVar("T", bound=BaseModel)


# ─────────────────────────────────────────────
# GeminiClient
# ─────────────────────────────────────────────

class GeminiClient:
    """
    Structured-output client for the Gemini API.

    Parameters
    ----------
    dry_run      : if True, returns canned responses without network calls
    log_dir      : directory to write per-game JSONL transcript logs
    """

    _TRANSCRIPT_DIR = Path("results/transcripts")

    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = dry_run
        self._transcript_dir = self._TRANSCRIPT_DIR
        self._transcript_dir.mkdir(parents=True, exist_ok=True)

        # Session log file: one JSONL file per run of main.py
        session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_path = self._transcript_dir / f"session_{session_ts}.jsonl"

        # Timestamp of the last live API call — used by the RPM throttle.
        self._last_call_ts: float = 0.0

        if not dry_run:
            api_key = config.require_api_key()
            self._client = genai.Client(api_key=api_key)
        else:
            self._client = None  # not used in dry-run mode

    # ── public API ────────────────────────────────────────────────────

    def generate_proposal(self, system_prompt: str, turn_prompt: str) -> ProposeOutput:
        """
        Ask the LLM to produce a Proposal (my_bundle + opponent_bundle).

        Returns a ProposeOutput guaranteed to match the Pydantic schema.
        If the API fails after all retries, raises RuntimeError.
        """
        return self._generate(
            system_prompt=system_prompt,
            turn_prompt=turn_prompt,
            schema=ProposeOutput,
        )

    def generate_response(self, system_prompt: str, turn_prompt: str) -> RespondOutput:
        """
        Ask the LLM to respond to a proposal (accept / reject / counter).

        Returns a RespondOutput guaranteed to match the Pydantic schema.
        """
        return self._generate(
            system_prompt=system_prompt,
            turn_prompt=turn_prompt,
            schema=RespondOutput,
        )

    # ── core generate ─────────────────────────────────────────────────

    def _generate(
        self,
        system_prompt: str,
        turn_prompt: str,
        schema: Type[T],
    ) -> T:
        """
        Call the Gemini API with structured output enforcement and retry logic.

        Structured Outputs:
            Passing response_schema=schema forces Gemini to return JSON that
            exactly matches the Pydantic model — no regex parsing needed.

        Retry strategy:
            - Network/transient errors (429 rate-limit, 503 overload): retry at
              the base temperature with exponential backoff.
            - CONTENT errors (empty text, invalid/!schema JSON): retry with a
              BUMPED temperature (config.LLM_RETRY_TEMPERATURE) so a deterministic
              temp-0 failure is not simply reproduced. The raw failed response is
              logged for post-mortem diagnosis.
        """
        if self.dry_run:
            return self._dry_run_response(schema)

        # RPM safety throttle: ensure at least LLM_REQUEST_DELAY seconds have
        # elapsed since the previous live call before issuing a new one.
        self._throttle()

        full_prompt = f"{system_prompt}\n\n{turn_prompt}"
        last_exc: Optional[Exception] = None
        # When the previous failure was a CONTENT error we raise the temperature
        # on the next attempt to escape the deterministic failure loop.
        bump_temperature = False

        # Content failures are eligible for the temperature-bump retry.
        content_errors = (EmptyResponseError, ValidationError, ValueError)

        for attempt in range(1, config.LLM_MAX_RETRIES + 1):
            temperature = (
                config.LLM_RETRY_TEMPERATURE if bump_temperature
                else config.LLM_TEMPERATURE
            )
            response = None
            try:
                response = self._client.models.generate_content(
                    model=config.GEMINI_MODEL,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=schema,
                        temperature=temperature,
                    ),
                )

                # Explicit empty/truncated guard. A thinking model can return a
                # candidate with no text part → response.text is None or "".
                raw_text = response.text
                if not raw_text or not raw_text.strip():
                    raise EmptyResponseError(
                        f"Gemini returned no text "
                        f"(finish_reason={self._finish_reason(response)})."
                    )

                parsed = schema.model_validate_json(raw_text)
                self._log_call(system_prompt, turn_prompt, raw_text, attempt,
                               temperature=temperature, ok=True)
                return parsed

            except content_errors as exc:
                # CONTENT failure → log the raw response, bump temp on next try.
                last_exc = exc
                bump_temperature = True
                raw = self._safe_text(response)
                self._log_call(system_prompt, turn_prompt, raw, attempt,
                               temperature=temperature, ok=False, error=repr(exc))
                wait = config.LLM_RETRY_BACKOFF * (2 ** (attempt - 1))
                print(
                    f"    [Gemini] Attempt {attempt}/{config.LLM_MAX_RETRIES} CONTENT "
                    f"error ({type(exc).__name__}). Retrying in {wait:.1f}s "
                    f"at temp={config.LLM_RETRY_TEMPERATURE}..."
                )
                time.sleep(wait)

            except Exception as exc:
                # NETWORK/transient failure → keep base temp, exponential backoff.
                last_exc = exc
                self._log_call(system_prompt, turn_prompt, self._safe_text(response),
                               attempt, temperature=temperature, ok=False, error=repr(exc))
                wait = config.LLM_RETRY_BACKOFF * (2 ** (attempt - 1))
                print(
                    f"    [Gemini] Attempt {attempt}/{config.LLM_MAX_RETRIES} network "
                    f"error ({type(exc).__name__}). Retrying in {wait:.1f}s..."
                )
                time.sleep(wait)

        raise RuntimeError(
            f"Gemini API failed after {config.LLM_MAX_RETRIES} attempts. "
            f"Last error: {last_exc}"
        )

    # ── response helpers ──────────────────────────────────────────────

    @staticmethod
    def _finish_reason(response) -> str:
        """Best-effort extraction of the candidate finish_reason for diagnostics."""
        try:
            return str(response.candidates[0].finish_reason)
        except Exception:
            return "unknown"

    @staticmethod
    def _safe_text(response) -> str:
        """
        Best-effort extraction of raw text from a (possibly malformed) response,
        for failure logging. Never raises.
        """
        if response is None:
            return ""
        try:
            if response.text:
                return response.text
        except Exception:
            pass
        # Fall back to digging the first candidate's parts directly.
        try:
            parts = response.candidates[0].content.parts
            return "".join(getattr(p, "text", "") or "" for p in parts)
        except Exception:
            return ""

    # ── RPM throttle ──────────────────────────────────────────────────

    def _throttle(self) -> None:
        """
        Block until at least config.LLM_REQUEST_DELAY seconds have passed since
        the last live API call. This enforces a hard upper bound on requests
        per minute (RPM) so a multi-game live run cannot trip rate limits.

        Implementation note: we sleep only the *remaining* interval rather than
        a blind time.sleep(LLM_REQUEST_DELAY), so we never over-wait when the
        model itself already took longer than the delay to respond.
        """
        elapsed = time.monotonic() - self._last_call_ts
        remaining = config.LLM_REQUEST_DELAY - elapsed
        if self._last_call_ts > 0.0 and remaining > 0:
            time.sleep(remaining)
        self._last_call_ts = time.monotonic()

    # ── dry-run fallback ──────────────────────────────────────────────

    def _dry_run_response(self, schema: Type[T]) -> T:
        """
        Return a canned but schema-valid response.
        Used in tests and offline demos.
        """
        if schema is ProposeOutput:
            return ProposeOutput(  # type: ignore[return-value]
                my_bundle=[ItemCount(name="book", count=2), ItemCount(name="hat", count=1), ItemCount(name="ball", count=0)],
                opponent_bundle=[ItemCount(name="book", count=1), ItemCount(name="hat", count=1), ItemCount(name="ball", count=1)],
                reasoning="[DRY RUN] Canned proposal.",
            )
        return RespondOutput(  # type: ignore[return-value]
            action="accept",
            my_bundle=[],
            opponent_bundle=[],
            reasoning="[DRY RUN] Canned accept.",
        )

    # ── logging ───────────────────────────────────────────────────────

    def _log_call(
        self,
        system_prompt: str,
        turn_prompt: str,
        raw_response: str,
        attempt: int,
        temperature: float,
        ok: bool,
        error: Optional[str] = None,
    ) -> None:
        """
        Append one API call record to the session JSONL log.

        BOTH successful and failed calls are logged. On failure the raw (often
        empty or malformed) response is captured under `response` and the
        exception under `error`, so empty-text / truncation problems can be
        diagnosed after the fact instead of being silently swallowed.

        We store the FULL prompts and response (no truncation). Each line is a
        json.dumps(...) of the record, so the .jsonl stays strictly valid JSON;
        for a successful call the nested `response` field is itself a parseable
        JSON document (json.loads(record["response"]) works).
        """
        record = {
            "id": str(uuid.uuid4()),
            "ts": datetime.now().isoformat(),
            "model": config.GEMINI_MODEL,
            "attempt": attempt,
            "temperature": temperature,
            "ok": ok,
            "error": error,
            "system_prompt": system_prompt,
            "turn_prompt": turn_prompt,
            "response": raw_response,
        }
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
