"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ORDIS — layer03a.py                                                         ║
║  Layer 03A · Voice Misinterpretation → Standard Term                        ║
║                                                                              ║
║  Input  : PII-redacted text from Layer 02                                    ║
║  Output : Text with voice misinterpretations replaced by standard terms      ║
║                                                                              ║
║  Method : Inject full Column D → Column A lookup table into LLM prompt      ║
║           LLM Call 1 — fix misheard words only, no paraphrasing              ║
║                                                                              ║
║  API    : POST /api/layer03a  (txt, accent, Ordis_ID)                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

What this layer does
─────────────────────
  Voice misinterpretations (Column D) → Standard terms (Column A)

  Examples:
    "Sundowing"        → "sundowning"
    "one to one"       → "one-to-one supervision"
    "double assist"    → "two-carer assistance"
    "nill by mouth"    → "nil by mouth"
    "verbally redited" → "verbal de-escalation"
    "comfort measures" → "palliative approach"
    "in proper touch"  → "safeguarding concern"

  Pipeline
  ────────
  1. Input validation  — guard against None / empty / whitespace-only
  2. Deterministic regex pre-pass — fast, zero LLM cost, unambiguous swaps
  3. LLM Call 1        — contextual / ambiguous corrections with full lookup
  4. Output validation — token-preservation check, length-sanity check
  5. Retry on failure  — up to _LLM_MAX_RETRIES attempts with back-off
  6. Audit diff        — SequenceMatcher multi-word diff for exact records
"""

from __future__ import annotations

import logging
import re
import time
import difflib
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

from shared import (
    ConfigManager,
    MedicalTermDatabase,
    ModelRouter,
    _GR, _YL, _GY, _R,
)

# ── Module logger (inherits root handler configured by the application) ───────
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
_LAYER_TAG            = "L3A"
_LLM_MAX_RETRIES      = 2          # attempts beyond the first
_LLM_RETRY_DELAY_S    = 1.0        # seconds between retries (linear back-off: n * delay)
_MAX_EXPAND_RATIO     = 2.0        # LLM output must not be >2× the input length
_MIN_SHRINK_RATIO     = 0.5        # LLM output must not be <50% of the input length
_PROTECTION_TOKEN_RE  = re.compile(
    r"\b(?:PATIENT|NURSE|STAFF|DOCTOR|CARER|MANAGER|RESIDENT)\d+\b"
)

# ── Preamble patterns the LLM may prepend ────────────────────────────────────
#   Covers: "Sure!", "Here is the corrected note:", "Corrected note:", markdown
#   fences, multi-sentence lead-ins ending in a colon, and "Output:".
_PREAMBLE_RE = re.compile(
    r"""(?ix)                          # ignore case, verbose
    ^(?:
        ```[a-z]*\n?                   # opening markdown fence
      | sure[,!\s]+                    # "Sure, " / "Sure!"
      | (?:of\s+course[,!\s]+)         # "Of course, "
      | here(?:'s|\s+is|\s+are)        # "Here's …" / "Here is …"
        [^:\n]{0,80}?:\s*              #   … up to 80 chars then colon
      | corrected[^:\n]{0,60}?:\s*     # "Corrected note:" etc.
      | (?:clinical\s+)?note[:\s]+     # "Note:" / "Clinical note:"
      | output[:\s]+                   # "Output:"
      | result[:\s]+                   # "Result:"
      | below\s+is[^:\n]{0,60}?:\s*   # "Below is the corrected …:"
    )+
    """,
    re.MULTILINE,
)
_TRAILING_FENCE_RE = re.compile(r"\s*```\s*$")


# ── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are a clinical transcription correction assistant for an Irish elderly care facility.

Your ONLY job in this step is to fix VOICE MISINTERPRETATIONS — words or phrases
that were misheard or mis-transcribed by the speech-to-text system.

You are given:
  1. A LOOKUP TABLE mapping misheard phrases (Column D) to their correct
     standard clinical terms (Column A).
  2. A clinical note that may contain misheard terms from Column D.

STRICT RULES:
  - Replace ONLY terms that appear in the lookup table's "Misheard" column.
  - Use EXACTLY the corresponding "Standard Term" from the table — no paraphrasing.
  - Do NOT rephrase, reorder, or restructure any other part of the text.
  - Do NOT fix grammar, spelling errors, or clinical language in this step.
  - Do NOT change patient/staff tokens (PATIENT1, NURSE1, CARER1, etc.).
  - Do NOT add or remove any clinical information.
  - Return ONLY the corrected text — no preamble, no explanation, no markdown fences.

If you find no misheard terms to correct, return the text EXACTLY as received.\
"""


# ── Public data types ─────────────────────────────────────────────────────────
@dataclass(frozen=True)
class CorrectionRecord:
    """Immutable record of a single term substitution."""
    original:   str
    corrected:  str
    source:     str   # "prepass" | "llm"
    layer:      str   # "L3A"


def run(
    text:       str,
    accent:     str                          = "ml_In",
    ordis_id:   str                          = "",
    cfg:        Optional[ConfigManager]      = None,
    router:     Optional[ModelRouter]        = None,
    medical_db: Optional[MedicalTermDatabase] = None,
) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Layer 03A — Voice Misinterpretation → Standard Term.

    Parameters
    ----------
    text       : PII-redacted text from Layer 02.
    accent     : Accent profile key (used by router / for logging).
    ordis_id   : Session ID for log correlation.
    cfg        : ConfigManager instance (created internally if omitted).
    router     : ModelRouter instance (created internally if omitted).
    medical_db : MedicalTermDatabase instance (created internally if omitted).

    Returns
    -------
    (corrected_text, corrections)
      corrected_text : Text with misheard terms replaced by standard terms.
      corrections    : List of (original_phrase, corrected_phrase) 2-tuples,
                       compatible with existing callers.  Use run_detailed()
                       for the richer CorrectionRecord form.

    Raises
    ------
    Does NOT raise.  On any unrecoverable error the function returns
    (original_text, []) and logs at ERROR level so the pipeline can continue.
    """
    records = _run_internal(text, accent, ordis_id, cfg, router, medical_db)
    corrected = records[0] if records else text
    corrections = [(r.original, r.corrected) for r in records[1]]
    return corrected, corrections


def run_detailed(
    text:       str,
    accent:     str                          = "ml_In",
    ordis_id:   str                          = "",
    cfg:        Optional[ConfigManager]      = None,
    router:     Optional[ModelRouter]        = None,
    medical_db: Optional[MedicalTermDatabase] = None,
) -> Tuple[str, List[CorrectionRecord]]:
    """
    Same as run() but returns typed CorrectionRecord objects instead of tuples.
    Prefer this form in new code.
    """
    return _run_internal(text, accent, ordis_id, cfg, router, medical_db)


# ── Internal implementation ───────────────────────────────────────────────────

def _run_internal(
    text:       str,
    accent:     str,
    ordis_id:   str,
    cfg:        Optional[ConfigManager],
    router:     Optional[ModelRouter],
    medical_db: Optional[MedicalTermDatabase],
) -> Tuple[str, List[CorrectionRecord]]:
    """Core logic shared by run() and run_detailed()."""
    log = _session_logger(ordis_id)

    # ── 0. Input validation ───────────────────────────────────────────────────
    if not isinstance(text, str):
        log.error("Non-string input received (%s). Returning empty string.", type(text))
        return "", []
    stripped = text.strip()
    if not stripped:
        log.debug("Empty / whitespace-only input — nothing to process.")
        return text, []

    # ── Lazy dependency construction ──────────────────────────────────────────
    if cfg is None:
        cfg = ConfigManager()
    if router is None:
        router = ModelRouter(cfg)
    if medical_db is None:
        medical_db = MedicalTermDatabase(cfg.get_data_path("medical_terms_csv"))

    all_records: List[CorrectionRecord] = []
    t0 = time.perf_counter()

    # ── 1. Deterministic regex pre-pass ───────────────────────────────────────
    pre_passed, raw_prepass = medical_db.apply_prepass(text)
    for orig, fixed in raw_prepass:
        all_records.append(CorrectionRecord(orig, fixed, "prepass", _LAYER_TAG))
    if raw_prepass:
        log.info("Pre-pass: %d term(s) corrected without LLM", len(raw_prepass))
        for rec in all_records:
            log.debug("  pre-pass: %r → %r", rec.original, rec.corrected)

    # ── 2. Build lookup table for LLM ─────────────────────────────────────────
    lookup_table = medical_db.build_col_d_to_a_table()
    if not lookup_table:
        log.warning("No lookup table available — skipping LLM call.")
        return pre_passed, all_records

    # ── 3. LLM Call (with retry) ──────────────────────────────────────────────
    user_prompt = (
        "LOOKUP TABLE (Misheard → Standard Term):\n\n"
        f"{lookup_table}\n\n"
        "CLINICAL NOTE TO CORRECT:\n\n"
        f"{pre_passed}\n\n"
        "Corrected note:"
    )
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]

    response = _call_llm_with_retry(router, messages, log)

    if response is None:
        log.error("All LLM attempts failed — returning pre-pass result.")
        return pre_passed, all_records

    # ── 4. Post-process LLM response ──────────────────────────────────────────
    cleaned = _strip_preamble(response)

    # ── 5. Output validation ──────────────────────────────────────────────────
    validated, ok = _validate_output(
        original=pre_passed,
        candidate=cleaned,
        layer=_LAYER_TAG,
        log=log,
    )
    if not ok:
        return pre_passed, all_records

    # ── 6. Audit diff ─────────────────────────────────────────────────────────
    llm_records = _diff_corrections(pre_passed, validated, source="llm", layer=_LAYER_TAG)
    all_records.extend(llm_records)
    if llm_records:
        log.info("LLM: %d additional term(s) corrected", len(llm_records))
        for rec in llm_records:
            log.debug("  llm: %r → %r", rec.original, rec.corrected)
    else:
        log.debug("LLM: no additional changes detected.")

    elapsed = time.perf_counter() - t0
    log.info(
        "%s complete in %.2fs — %d total correction(s)",
        _LAYER_TAG, elapsed, len(all_records),
    )
    return validated, all_records


# ── Helpers ───────────────────────────────────────────────────────────────────

def _session_logger(ordis_id: str) -> logging.Logger:
    """Return a child logger tagged with the session ID."""
    if ordis_id:
        return logger.getChild(ordis_id)
    return logger


def _call_llm_with_retry(
    router:   ModelRouter,
    messages: list,
    log:      logging.Logger,
) -> Optional[str]:
    """
    Call router.chat() up to 1 + _LLM_MAX_RETRIES times.
    Returns the raw string response, or None if all attempts fail.
    """
    for attempt in range(1 + _LLM_MAX_RETRIES):
        if attempt > 0:
            delay = attempt * _LLM_RETRY_DELAY_S
            log.warning("Retry %d/%d in %.1fs…", attempt, _LLM_MAX_RETRIES, delay)
            time.sleep(delay)

        response = router.chat(messages, max_tokens=router.max_tokens)

        if router.is_error(response):
            log.warning("LLM error (attempt %d): %s", attempt + 1, response)
            continue

        if not isinstance(response, str) or not response.strip():
            log.warning("LLM returned empty/non-string response (attempt %d).", attempt + 1)
            continue

        return response

    return None


def _strip_preamble(text: str) -> str:
    """
    Remove boilerplate the LLM may prepend or append.
    Handles markdown fences, lead-in sentences, trailing whitespace.
    """
    # Strip opening preamble
    cleaned = _PREAMBLE_RE.sub("", text, count=1).strip()
    # Strip trailing markdown fence (``` or ```text)
    cleaned = _TRAILING_FENCE_RE.sub("", cleaned).strip()
    # Strip surrounding quotation marks that some models add
    if len(cleaned) >= 2 and cleaned[0] in ('"', "'") and cleaned[0] == cleaned[-1]:
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _extract_protection_tokens(text: str) -> Set[str]:
    """Return the set of all PATIENT/NURSE/… tokens present in text."""
    return set(_PROTECTION_TOKEN_RE.findall(text))


def _validate_output(
    original:  str,
    candidate: str,
    layer:     str,
    log:       logging.Logger,
) -> Tuple[str, bool]:
    """
    Validate that the LLM output is safe to use.

    Checks:
    - Not empty after stripping.
    - Length within [_MIN_SHRINK_RATIO, _MAX_EXPAND_RATIO] of original.
    - All protection tokens (PATIENT1, NURSE1 …) are preserved.

    Returns (candidate, True) if valid, (original, False) otherwise.
    """
    stripped = candidate.strip()

    # 1. Empty output guard
    if not stripped:
        log.error("%s validation failed: LLM returned empty output.", layer)
        return original, False

    # 2. Length sanity
    orig_len = max(len(original), 1)
    cand_len = len(candidate)
    ratio    = cand_len / orig_len

    if ratio > _MAX_EXPAND_RATIO:
        log.error(
            "%s validation failed: output is %.1f× longer than input "
            "(likely hallucination). Falling back.",
            layer, ratio,
        )
        return original, False

    if ratio < _MIN_SHRINK_RATIO:
        log.error(
            "%s validation failed: output is only %.0f%% of input length "
            "(likely truncation). Falling back.",
            layer, ratio * 100,
        )
        return original, False

    # 3. Protection-token preservation
    original_tokens  = _extract_protection_tokens(original)
    candidate_tokens = _extract_protection_tokens(candidate)
    missing          = original_tokens - candidate_tokens

    if missing:
        log.error(
            "%s validation failed: LLM dropped protection token(s): %s. "
            "Falling back to input.",
            layer, ", ".join(sorted(missing)),
        )
        return original, False

    extra = candidate_tokens - original_tokens
    if extra:
        log.warning(
            "%s validation: LLM introduced unexpected token(s): %s. "
            "Accepting output but flagging for review.",
            layer, ", ".join(sorted(extra)),
        )

    return stripped, True


def _diff_corrections(
    before: str,
    after:  str,
    source: str,
    layer:  str,
) -> List[CorrectionRecord]:
    """
    Multi-word diff using difflib.SequenceMatcher.

    Compared to a naive zip-based word scan, this correctly handles:
    - Multi-word substitutions  ("nill by mouth" → "nil by mouth")
    - Length-changing edits     (insertions / deletions)
    - Non-aligned token shifts  (when earlier corrections shift word positions)

    Returns a list of CorrectionRecord for every replace/delete/insert opcode.
    """
    before_words = before.split()
    after_words  = after.split()

    if before_words == after_words:
        return []

    matcher = difflib.SequenceMatcher(
        isjunk=None,
        a=before_words,
        b=after_words,
        autojunk=False,  # disable heuristic — clinical text is not "junk"
    )
    records: List[CorrectionRecord] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        original  = " ".join(before_words[i1:i2])
        corrected = " ".join(after_words[j1:j2])
        if original != corrected:
            records.append(CorrectionRecord(original, corrected, source, layer))

    return records


# ─────────────────────────────────────────────────────────────────────────────
#  CLI smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)-8s %(name)s — %(message)s",
    )

    _SAMPLE = (
        "PATIENT1 (room 22) was observed Sundowing. "
        "He requires one to one and is on pads, needs double assist. "
        "NURSE1 reported PATIENT2 is nill by mouth. "
        "Verbally redited was used due to aggression. "
        "PATIENT3 requested comfort measures."
    )

    text_in = sys.argv[1] if len(sys.argv) > 1 else _SAMPLE
    out, corrections = run(text_in)

    print("\n── OUTPUT ──────────────────────────────────────────────")
    print(out)
    print(f"\n── CORRECTIONS ({len(corrections)}) ──")
    for orig, fixed in corrections:
        print(f"  {_YL}'{orig}'{_R}  →  {_GR}'{fixed}'{_R}")