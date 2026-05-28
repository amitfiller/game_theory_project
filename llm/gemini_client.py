"""
llm/gemini_client.py
────────────────────
Placeholder / architecture stub for the Gemini API client.

This file will be fully implemented in Phase 3, after the offline
heuristic game loop is verified.  It is committed now so the import
structure and planned interface are visible to reviewers.

Planned implementation (Phase 3)
─────────────────────────────────
  Install:  pip install google-generativeai

  Key design decisions:
    1. Structured Outputs (JSON schema enforcement)
       Gemini 1.5 Flash/Pro supports `response_mime_type="application/json"`
       with an explicit `response_schema`.  This guarantees the model returns
       valid JSON matching our Proposal / Response schema — eliminating the
       #1 source of runtime crashes in LLM-based game agents.

    2. Retry with exponential backoff
       Transient errors (429 rate-limit, 503 overload) are retried up to
       LLM_MAX_RETRIES times with LLM_RETRY_BACKOFF-second intervals.

    3. Dry-run mode
       When dry_run=True the client returns canned JSON responses without
       making any network calls.  Used in CI and offline demos.

    4. Full transcript logging
       Every (prompt, response) pair is appended to
       results/transcripts/<timestamp>.jsonl for post-game analysis.

Planned interface
─────────────────
  client = GeminiClient(dry_run=False)

  # Returns a dict matching the Proposal JSON schema
  proposal_dict = client.generate_proposal(
      system_prompt: str,
      history_text: str,
  ) -> dict

  # Returns a dict with keys: action ("accept"|"reject"|"counter"), bundles, reasoning
  response_dict = client.generate_response(
      system_prompt: str,
      proposal_text: str,
  ) -> dict
"""


class GeminiClient:
    """
    Thin wrapper around the Google GenAI SDK for structured negotiation calls.

    NOT YET IMPLEMENTED — stub only.
    """

    def __init__(self, dry_run: bool = False) -> None:
        raise NotImplementedError(
            "GeminiClient is not yet implemented. "
            "Phase 3 will build this after the heuristic game loop is stable. "
            "Use HeuristicAgent for offline testing."
        )
