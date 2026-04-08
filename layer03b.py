"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ORDIS — layer03b.py                                                         ║
║  Layer 03B · Standard Term → Professional Clinical Language                 ║
║                                                                              ║
║  Input  : Text from Layer 03A (misinterpretations already corrected)         ║
║  Output : Fully professional clinical note                                   ║
║                                                                              ║
║  Method : Inject Column A → Column B lookup table into LLM prompt            ║
║           LLM Call 2 — upgrade terminology + fix grammar + fix drug names   ║
║                                                                              ║
║  API    : POST /api/layer03b  (txt, accent, Ordis_ID)                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

What this layer does
─────────────────────
  Standard terms (Column A) → Professional terms (Column B)
  + Grammar correction
  + Drug name spelling  (e.g. "parasitamol" → "paracetamol")

  Examples:
    "sundowning"             → "sundowning syndrome / evening confusion"
    "one-to-one supervision" → "1:1 continuous supervision"
    "two-carer assistance"   → "requires double-assist for all cares"
    "nil by mouth"           → "NBM (nil by mouth)"
    "verbal de-escalation"   → "verbal de-escalation techniques employed"
    "refused care"           → "Resident Declined Help"
    "palliative approach"    → "palliative care plan in place"
    "safeguarding concern"   → "safeguarding concern raised — reported to senior staff"

  CLINICAL SAFETY RULES (enforced at prompt level AND via post-LLM audit):
    - NEVER convert patient actions to passive voice
    - ALWAYS name the actor (patient self-removed / staff removed)
    - Preserve recurrence markers (again, keeps, every night → "recurring")
    - Never soften refusals ("refused" must stay as patient refusal)
    - Drug names corrected to BNF-approved spelling

  Pipeline
  ────────
  1. Input validation       — guard against None / empty / whitespace-only
  2. Lookup-table injection — Column A → B terms into user prompt
  3. LLM Call 2             — professionalise with few-shot examples
  4. Output validation      — token-preservation + length sanity
  5. Safety invariant audit — detect agency / refusal / recurrence drift
  6. Retry on failure       — up to _LLM_MAX_RETRIES with linear back-off
  7. Audit diff             — SequenceMatcher multi-word diff for records
"""

from __future__ import annotations

import logging
import re
import time
import difflib
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

from shared import (
    ConfigManager,
    MedicalTermDatabase,
    ModelRouter,
    _GR, _YL, _GY, _R,
)

# ── Module logger ─────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
_LAYER_TAG            = "L3B"
_LLM_MAX_RETRIES      = 2
_LLM_RETRY_DELAY_S    = 1.0
_MAX_EXPAND_RATIO     = 2.5   # professionalisation may legitimately expand text
_MIN_SHRINK_RATIO     = 0.4
_PROTECTION_TOKEN_RE  = re.compile(
    r"\b(?:PATIENT|NURSE|STAFF|DOCTOR|CARER|MANAGER|RESIDENT)\d+\b"
)

# ── Preamble strip (same pattern as L3A — keep in sync or extract to shared) ──
_PREAMBLE_RE = re.compile(
    r"""(?ix)
    ^(?:
        ```[a-z]*\n?
      | sure[,!\s]+
      | (?:of\s+course[,!\s]+)
      | here(?:'s|\s+is|\s+are)[^:\n]{0,80}?:\s*
      | (?:professional[^:\n]{0,60}?:\s*)
      | (?:clinical[^:\n]{0,60}?:\s*)
      | (?:the\s+)?(?:upgraded|refined|formatted)\s+note[:\s]+
      | output[:\s]+
      | result[:\s]+
      | below\s+is[^:\n]{0,60}?:\s*
    )+
    """,
    re.MULTILINE,
)
_TRAILING_FENCE_RE = re.compile(r"\s*```\s*$")


# ── Safety invariant patterns ─────────────────────────────────────────────────
# These patterns detect clinical safety signals that must be preserved.
# If found in the input but absent in the LLM output, we emit a WARNING.
# We do NOT auto-revert on safety warnings alone (clinical review is required)
# but we do flag for the audit trail.

# "refused" / "refuses" / "won't take" / "not taking" (patient refusal signals)
_REFUSAL_RE = re.compile(
    r"\b(?:refus(?:ed|es|ing)|won'?t\s+take|not\s+taking|declin(?:ed|es|ing))\b",
    re.IGNORECASE,
)

# Self-agency patterns: "self-removed", "self removed", "patient removed", etc.
_AGENCY_RE = re.compile(
    r"\b(?:self[\s-]removed|self[\s-]discon|patient\s+removed|patient\s+pulled"
    r"|patient\s+got\s+up|pulled\s+out\s+(?:own|their|his|her))\b",
    re.IGNORECASE,
)

# Recurrence markers
_RECURRENCE_RE = re.compile(
    r"\b(?:again(?:\s+again)?|keeps?\s+\w+ing|every\s+night|same\s+as\s+before"
    r"|recurring|persistent|repeated|second\s+occurrence)\b",
    re.IGNORECASE,
)

# ── Few-shot examples ─────────────────────────────────────────────────────────
_FEW_SHOT = """\
Example 1:
Input:  PATIENT1 requires one-to-one supervision and needs double assist.
Output: PATIENT1 requires 1:1 continuous supervision and double-assist for all personal cares.

Example 2:
Input:  PATIENT2 had nil by mouth and verbal de-escalation was required.
Output: PATIENT2 was NBM (nil by mouth). Verbal de-escalation techniques were employed during the episode.

Example 3:
Input:  PATIENT3 refused care and requested palliative approach.
Output: PATIENT3 declined all cares (Resident Declined Help) and has a palliative care plan in place.

Example 4:
Input:  PATIENT4 self-removed IV drip again again.
Output: PATIENT4 self-removed IV cannula — recurring incident (repeated self-removal today). \
Incident documentation required.

Example 5:
Input:  bp is 180 over 120 giv parasitamol for pain.
Output: BP 180/120 mmHg. Paracetamol administered for pain management.

Example 6:
Input:  PATIENT5 was found on the floor. she gets up every night.
Output: PATIENT5 found on the floor — recurring falls (nightly pattern). \
Fall protocol initiated; bed rail and sensor mat check required.

Example 7:
Input:  PATIENT6 won't take insuleen.
Output: PATIENT6 declined insulin administration (Resident Declined Help). \
Prescriber notification required; document in medication refusal log.\
"""

# ── Clinical safety rules (injected into the system prompt) ──────────────────
_SAFETY_RULES = """\
CRITICAL CLINICAL SAFETY RULES — enforce all of these without exception:

1. AGENCY — never convert a patient action to passive voice.
   WRONG: "IV drip removed."
   RIGHT: "Patient self-removed IV cannula."
   If staff removed after completion → "IV cannula removed by staff — infusion complete."
   If unclear → "IV cannula found disconnected — cause unclear."

2. ACTOR — always name who performed the action.
   · Patient initiated → "Patient self-removed…" / "PATIENT1 self-removed…"
   · Staff completed → "removed by staff"
   · Unclear → "found [state] — cause unclear"

3. RECURRENCE — preserve ALL recurrence signals; never discard them.
   "again again", "keeps doing", "every night", "same as before"
   → "recurring", "persistent", "repeated", "ongoing", "second occurrence today"
   These are clinical safety events and must appear in the output.

4. REFUSAL — "refuses" / "won't take" / "not taking" must remain as patient refusal.
   Use "Resident Declined Help" or "declined [X] (Resident Declined Help)".
   NEVER soften to "not administered" (passive / hides patient agency).

5. DRUG NAMES — correct to BNF-approved spelling only:
   parasitamol → paracetamol   |  insuleen → insulin
   warfarine  → warfarin       |  setraline → sertraline
   amoxacillin → amoxicillin   |  metaformin → metformin\
"""

# ── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = (
    """\
You are a senior clinical documentation specialist for an Irish elderly care facility.

Your job is to transform a clinical note into polished, professional documentation.

You are given:
  1. A TERMINOLOGY TABLE mapping standard terms (Column A) to their professional
     equivalents (Column B). Use these EXACT professional terms where applicable.
  2. A clinical note that needs professional language upgrading.

YOUR TASKS (in order):
  1. Replace standard terms with their professional equivalents from the table.
  2. Fix all grammar, sentence structure, and clinical writing style.
  3. Correct drug name spelling errors (see safety rules below).
  4. Use standard clinical abbreviations: BP, IV, ECG, O2, PRN, ADLs, TDS, BD,
     NGT, PEG, NBM, NEWS2, UTI, etc.
  5. Format vitals correctly: "BP 180/120 mmHg", "O2 sats 94%", "Temp 37.2°C".

STRICT RULES:
  - Do NOT add any clinical information not present in the input.
  - Do NOT change patient/staff tokens (PATIENT1, NURSE1, etc.).
  - Return ONLY the professional note — no preamble, no explanation, no markdown fences.

"""
    + _SAFETY_RULES
)


# ── Public data types ─────────────────────────────────────────────────────────
@dataclass(frozen=True)
class CorrectionRecord:
    """Immutable record of a single term substitution."""
    original:   str
    corrected:  str
    source:     str   # "llm"
    layer:      str   # "L3B"


@dataclass
class SafetyAuditResult:
    """Result of the post-LLM clinical safety invariant check."""
    refusal_drift:    bool = False   # refusal signal present in input but absent in output
    agency_drift:     bool = False   # self-removal signal lost
    recurrence_drift: bool = False   # recurrence marker lost

    @property
    def any_drift(self) -> bool:
        return self.refusal_drift or self.agency_drift or self.recurrence_drift


def run(
    text:       str,
    accent:     str                          = "ml_In",
    ordis_id:   str                          = "",
    cfg:        Optional[ConfigManager]      = None,
    router:     Optional[ModelRouter]        = None,
    medical_db: Optional[MedicalTermDatabase] = None,
) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Layer 03B — Standard Term → Professional Clinical Language.

    Parameters
    ----------
    text       : Text from Layer 03A (standard terms in place).
    accent     : Accent profile key (used by router / for logging).
    ordis_id   : Session ID for log correlation.
    cfg        : ConfigManager instance (created internally if omitted).
    router     : ModelRouter instance.
    medical_db : MedicalTermDatabase (for lookup table injection).

    Returns
    -------
    (professional_text, corrections)
      professional_text : Fully professional clinical note.
      corrections       : List of (original_phrase, professional_phrase) tuples.

    Raises
    ------
    Does NOT raise.  On any unrecoverable error the function returns
    (original_text, []) and logs at ERROR level so the pipeline can continue.
    """
    text_out, records = _run_internal(text, accent, ordis_id, cfg, router, medical_db)
    return text_out, [(r.original, r.corrected) for r in records]


def run_detailed(
    text:       str,
    accent:     str                          = "ml_In",
    ordis_id:   str                          = "",
    cfg:        Optional[ConfigManager]      = None,
    router:     Optional[ModelRouter]        = None,
    medical_db: Optional[MedicalTermDatabase] = None,
) -> Tuple[str, List[CorrectionRecord], SafetyAuditResult]:
    """
    Same as run() but returns typed CorrectionRecord objects and a
    SafetyAuditResult.  Prefer this form in new code.
    """
    text_out, records = _run_internal(text, accent, ordis_id, cfg, router, medical_db)
    audit = _audit_safety_invariants(text, text_out, logger)
    return text_out, records, audit


# ── Internal implementation ───────────────────────────────────────────────────

def _run_internal(
    text:       str,
    accent:     str,
    ordis_id:   str,
    cfg:        Optional[ConfigManager],
    router:     Optional[ModelRouter],
    medical_db: Optional[MedicalTermDatabase],
) -> Tuple[str, List[CorrectionRecord]]:
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

    t0 = time.perf_counter()

    # ── 1. Build Column A → Column B lookup table ─────────────────────────────
    lookup_table  = medical_db.build_col_a_to_b_table()
    table_section = (
        f"TERMINOLOGY TABLE (Standard → Professional):\n\n{lookup_table}\n\n"
        if lookup_table else ""
    )

    # ── 2. Construct prompt ───────────────────────────────────────────────────
    user_prompt = (
        f"{table_section}"
        f"CLINICAL NOTE TO PROFESSIONALISE:\n\n"
        f"{text}\n\n"
        f"--- FEW-SHOT EXAMPLES ---\n{_FEW_SHOT}\n\n"
        f"Professional clinical note:"
    )
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]

    # ── 3. LLM Call (with retry) ──────────────────────────────────────────────
    log.info("Calling LLM for professional language upgrade…")
    response = _call_llm_with_retry(router, messages, log)

    if response is None:
        log.error("All LLM attempts failed — returning input unchanged.")
        return text, []

    # ── 4. Post-process ───────────────────────────────────────────────────────
    cleaned = _strip_preamble(response)

    # ── 5. Output validation ──────────────────────────────────────────────────
    validated, ok = _validate_output(
        original=text,
        candidate=cleaned,
        layer=_LAYER_TAG,
        log=log,
    )
    if not ok:
        return text, []

    # ── 6. Clinical safety invariant audit ───────────────────────────────────
    audit = _audit_safety_invariants(text, validated, log)
    if audit.any_drift:
        # Log prominently but do not auto-revert — clinical staff must review.
        log.warning(
            "%s SAFETY AUDIT ALERT for session %s: "
            "refusal_drift=%s agency_drift=%s recurrence_drift=%s. "
            "Output accepted but flagged for mandatory clinical review.",
            _LAYER_TAG, ordis_id or "unknown",
            audit.refusal_drift, audit.agency_drift, audit.recurrence_drift,
        )

    # ── 7. Audit diff ─────────────────────────────────────────────────────────
    records = _diff_corrections(text, validated, source="llm", layer=_LAYER_TAG)
    if records:
        log.info("%s: %d professional term upgrade(s)", _LAYER_TAG, len(records))
    else:
        log.debug("%s: minimal changes (grammar/formatting only).", _LAYER_TAG)

    elapsed = time.perf_counter() - t0
    log.info(
        "%s complete in %.2fs — %d correction(s), safety_drift=%s",
        _LAYER_TAG, elapsed, len(records), audit.any_drift,
    )
    return validated, records


# ── Helpers ───────────────────────────────────────────────────────────────────

def _session_logger(ordis_id: str) -> logging.Logger:
    if ordis_id:
        return logger.getChild(ordis_id)
    return logger


def _call_llm_with_retry(
    router:   ModelRouter,
    messages: list,
    log:      logging.Logger,
) -> Optional[str]:
    """Attempt router.chat() up to 1 + _LLM_MAX_RETRIES times."""
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
            log.warning("LLM empty/non-string (attempt %d).", attempt + 1)
            continue

        return response
    return None


def _strip_preamble(text: str) -> str:
    """Remove boilerplate the LLM may prepend or append."""
    cleaned = _PREAMBLE_RE.sub("", text, count=1).strip()
    cleaned = _TRAILING_FENCE_RE.sub("", cleaned).strip()
    if len(cleaned) >= 2 and cleaned[0] in ('"', "'") and cleaned[0] == cleaned[-1]:
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _extract_protection_tokens(text: str) -> Set[str]:
    return set(_PROTECTION_TOKEN_RE.findall(text))


def _validate_output(
    original:  str,
    candidate: str,
    layer:     str,
    log:       logging.Logger,
) -> Tuple[str, bool]:
    """
    Validate LLM output length and protection-token preservation.
    Returns (candidate, True) if valid, (original, False) otherwise.
    """
    stripped = candidate.strip()
    if not stripped:
        log.error("%s validation failed: LLM returned empty output.", layer)
        return original, False

    orig_len = max(len(original), 1)
    ratio    = len(candidate) / orig_len

    if ratio > _MAX_EXPAND_RATIO:
        log.error(
            "%s validation failed: output %.1f× longer than input "
            "(likely hallucination). Falling back.", layer, ratio,
        )
        return original, False

    if ratio < _MIN_SHRINK_RATIO:
        log.error(
            "%s validation failed: output only %.0f%% of input length "
            "(likely truncation). Falling back.", layer, ratio * 100,
        )
        return original, False

    missing = _extract_protection_tokens(original) - _extract_protection_tokens(candidate)
    if missing:
        log.error(
            "%s validation failed: LLM dropped token(s): %s. Falling back.",
            layer, ", ".join(sorted(missing)),
        )
        return original, False

    extra = _extract_protection_tokens(candidate) - _extract_protection_tokens(original)
    if extra:
        log.warning(
            "%s validation: LLM introduced unexpected token(s): %s.",
            layer, ", ".join(sorted(extra)),
        )

    return stripped, True


def _audit_safety_invariants(
    before: str,
    after:  str,
    log:    logging.Logger,
) -> SafetyAuditResult:
    """
    Check that key clinical safety signals present in `before` are not
    silently dropped from `after`.

    This is a heuristic safety net — it does NOT make semantic judgements
    about whether the LLM's rephrasing is clinically acceptable.  It only
    fires when a signal is COMPLETELY absent in the output.

    All detected drifts are logged as WARNING so they surface in the
    audit trail regardless of whether the caller acts on them.
    """
    result = SafetyAuditResult()

    # 1. Refusal preservation
    if _REFUSAL_RE.search(before) and not _REFUSAL_RE.search(after):
        # Allow "Resident Declined Help" as an acceptable substitute
        if "declined" not in after.lower() and "rdh" not in after.lower():
            log.warning(
                "SAFETY DRIFT [refusal]: refusal signal found in input "
                "but absent in LLM output. Input excerpt: %r",
                before[:120],
            )
            result.refusal_drift = True

    # 2. Agency preservation (self-removal)
    if _AGENCY_RE.search(before) and not _AGENCY_RE.search(after):
        log.warning(
            "SAFETY DRIFT [agency]: self-removal/agency signal found in input "
            "but absent in LLM output. Input excerpt: %r",
            before[:120],
        )
        result.agency_drift = True

    # 3. Recurrence preservation
    if _RECURRENCE_RE.search(before) and not _RECURRENCE_RE.search(after):
        log.warning(
            "SAFETY DRIFT [recurrence]: recurrence marker found in input "
            "but absent in LLM output. Input excerpt: %r",
            before[:120],
        )
        result.recurrence_drift = True

    return result


def _diff_corrections(
    before: str,
    after:  str,
    source: str,
    layer:  str,
) -> List[CorrectionRecord]:
    """
    Multi-word diff using difflib.SequenceMatcher.

    Handles multi-word substitutions, insertions, deletions, and non-aligned
    token shifts correctly.  Shared implementation with L3A (kept local to
    avoid a shared-module circular import — extract to shared.py at will).
    """
    before_words = before.split()
    after_words  = after.split()

    if before_words == after_words:
        return []

    matcher = difflib.SequenceMatcher(
        isjunk=None,
        a=before_words,
        b=after_words,
        autojunk=False,
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
        "PATIENT1 (room 22) was observed with sundowning. "
        "He requires one-to-one supervision, is on incontinence pads, "
        "and requires two-carer assistance. "
        "NURSE1 reported PATIENT2 is nil by mouth. "
        "Verbal de-escalation was required. "
        "The neighbouring patient has a safeguarding concern. "
        "PATIENT3 is declining from usual state. "
        "PATIENT4 refused care and requested palliative approach."
    )

    text_in = sys.argv[1] if len(sys.argv) > 1 else _SAMPLE
    out, corrections = run(text_in)

    print("\n── OUTPUT ──────────────────────────────────────────────")
    print(out)
    print(f"\n── CORRECTIONS ({len(corrections)}) ──")
    for orig, fixed in corrections:
        print(f"  {_YL}'{orig}'{_R}  →  {_GR}'{fixed}'{_R}")