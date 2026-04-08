"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ORDIS — layer02.py  (Presidio Edition · v2)                                 ║
║  Layer 02 · PII Redaction                                                    ║
║                                                                              ║
║  Input  : Text from Layer 01 (names already corrected)                       ║
║  Output : Text with all real names replaced by anonymous tokens              ║
║                                                                              ║
║  Tokens : PATIENT1, PATIENT2 … / NURSE1, NURSE2 …                           ║
║  Storage: PII map saved to output/pii_map_{Ordis_ID}.json                   ║
║           (Layer 04 reads this file to reverse the redaction)                ║
║                                                                              ║
║  Detection strategy (two-stage):                                             ║
║    Stage 1 — Microsoft Presidio (spaCy NER) detects PERSON entities.         ║
║    Stage 2 — Registry confirmation: each Presidio hit is confirmed against   ║
║              patients.csv / nurses.csv via fuzzy match (threshold 72).       ║
║              Only confirmed registry hits are redacted.                      ║
║              Fallback: if registry is empty, Presidio hits are redacted      ║
║              directly (useful during dev / no-CSV mode).                     ║
║                                                                              ║
║  v2 improvements over v1:                                                    ║
║    · Possessives handled   — "Marcus's" → "PATIENT1's"                       ║
║    · Honorifics stripped   — "Mr. O'Reilly" confirmed against registry       ║
║    · Unicode normalised    — NFC before every match; no code-point drift     ║
║    · Overlapping spans     — longest-span-first; sub-spans suppressed        ║
║    · Part-name redaction   — first names registered as token aliases         ║
║    · Thread-safe engine    — lazy singleton with double-checked locking      ║
║    · Input validation      — TypeError / ValueError on bad args              ║
║    · Ordis_ID sanitisation — no path traversal                               ║
║    · Registry guard        — None / empty entries silently skipped           ║
║    · Retry on NLP errors   — 2-attempt Presidio wrapper                      ║
║    · Structured audit      — RedactionEvent dataclass, not raw prints        ║
║    · logging throughout    — no bare print(); fully configurable             ║
║    · 4-word fallback window — catches "Mary Clare O'Brien"-style names       ║
║                                                                              ║
║  Install (once):                                                             ║
║    pip install presidio-analyzer presidio-anonymizer spacy                   ║
║    python -m spacy download en_core_web_lg                                   ║
║                                                                              ║
║  API    : POST /api/layer02  (txt, accent, Ordis_ID)                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import logging
import re
import threading
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ── Presidio imports (graceful fallback if not installed) ─────────────────────
try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    _PRESIDIO_AVAILABLE = True
except ImportError:
    _PRESIDIO_AVAILABLE = False

from shared import (
    ConfigManager,
    load_name_registry,
    save_pii_map,
    match_name,
    _GR, _YL, _GY, _RE, _R,
)

logger = logging.getLogger(__name__)

__all__ = ["run", "RedactionEvent"]

# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────

_PATIENT_PREFIX      = "PATIENT"
_NURSE_PREFIX        = "NURSE"

# Registry-confirmation confidence floor (0–100)
_CONFIRM_THRESHOLD   = 72
# Raised floor for the Presidio-less fallback (more conservative)
_FALLBACK_THRESHOLD  = 78

# Minimum number of characters a surface span must have to be considered
_MIN_NAME_LENGTH     = 2
# Guard against unreasonably long / crafted Ordis_IDs
_MAX_ORDIS_ID_LENGTH = 128

# Honorific / title tokens that may prefix a name in clinical notes.
# Stripped before registry lookup so "Nurse Dharani" still matches "Dharani".
_HONORIFICS: frozenset = frozenset({
    "mr", "mrs", "ms", "miss", "dr", "prof", "professor",
    "nurse", "patient", "resident", "carer", "sr", "jr",
    "rev", "reverend", "sir", "dame",
})

# Words that spaCy's NER sometimes mis-tags as PERSON entities.
# Stored in a frozenset for O(1) look-ups.
_NER_STOPWORDS: frozenset = frozenset({
    # Pronouns / articles
    "he", "she", "they", "him", "her", "his", "the", "a", "an",
    # Clinical roles (must not become tokens)
    "resident", "nurse", "patient", "doctor", "staff", "carer",
    # Clinical / facility terms
    "room", "ward", "unit", "care", "home", "recording", "report",
    "palliative", "approach", "corner", "bottle", "bed", "bandage",
    "intake", "discomfort", "aggression", "behavior", "behaviour",
    "confusion", "technique", "application", "condition", "decline",
    # Common clinical verbs sometimes tagged as entities
    "stated", "observed", "reported", "required", "performed",
    # Temporal terms
    "daily", "weekly", "monthly", "yesterday", "today", "tomorrow",
    "morning", "afternoon", "evening", "overnight", "community",
})


# ─────────────────────────────────────────────────────────────────────────────
#  Audit data-class
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RedactionEvent:
    """
    Structured record of a single name-redaction decision.

    Consumers (e.g. audit logs, test assertions) can inspect the full
    provenance of every token that was written into the output text.

    Attributes
    ----------
    token          : The anonymous token written into the text (e.g. "PATIENT1").
    canonical_name : The registry-confirmed canonical form of the name.
    surface_span   : The exact substring found in the source text.
    role           : "patient" | "nurse" | "patient" (no-registry fallback).
    confidence     : Fuzzy-match confidence score returned by match_name()
                     (0.0–100.0), or 95.0 when Presidio is trusted directly.
    method         : How the span was detected:
                       "presidio"       – Presidio NER + registry confirmation
                       "registry_only"  – conservative word-window fallback
    """
    token:          str
    canonical_name: str
    surface_span:   str
    role:           str
    confidence:     float
    method:         str
    events:         List[str] = field(default_factory=list)  # sub-events (part-names)


# ─────────────────────────────────────────────────────────────────────────────
#  Thread-safe Presidio engine singleton
# ─────────────────────────────────────────────────────────────────────────────

_engine_lock:    threading.Lock               = threading.Lock()
_PRESIDIO_ENGINE: Optional["AnalyzerEngine"]  = None
_engine_built:   bool                         = False


def _get_presidio_engine() -> Optional["AnalyzerEngine"]:
    """
    Lazy, thread-safe singleton for the Presidio AnalyzerEngine.

    Uses double-checked locking to avoid the cost of acquiring the mutex
    on every call after the engine has already been built.
    """
    global _PRESIDIO_ENGINE, _engine_built
    if _engine_built:
        return _PRESIDIO_ENGINE
    with _engine_lock:
        if _engine_built:              # second check inside the lock
            return _PRESIDIO_ENGINE
        _PRESIDIO_ENGINE = _build_presidio_engine()
        _engine_built    = True
    return _PRESIDIO_ENGINE


def _build_presidio_engine() -> Optional["AnalyzerEngine"]:
    """
    Construct a Presidio AnalyzerEngine backed by en_core_web_lg.
    Returns None (with a warning) if Presidio or the spaCy model is absent.
    """
    if not _PRESIDIO_AVAILABLE:
        logger.warning(
            "Presidio is not installed. "
            "Run: pip install presidio-analyzer presidio-anonymizer spacy "
            "&& python -m spacy download en_core_web_lg"
        )
        return None
    try:
        configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
        }
        provider   = NlpEngineProvider(nlp_configuration=configuration)
        nlp_engine = provider.create_engine()
        engine     = AnalyzerEngine(nlp_engine=nlp_engine)
        logger.info("Presidio AnalyzerEngine built successfully (en_core_web_lg).")
        return engine
    except Exception as exc:
        logger.warning(
            "Presidio engine failed to initialise: %s  "
            "→ Run: python -m spacy download en_core_web_lg",
            exc,
        )
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  Input validation
# ─────────────────────────────────────────────────────────────────────────────

def _validate_text(text: Any) -> str:
    """
    Validate and Unicode-normalise (NFC) the input text.

    NFC normalisation ensures that combining characters (e.g. accented letters
    written as base + combining mark) are represented as precomposed code-points,
    which makes all downstream fuzzy matches deterministic regardless of how the
    text was produced (copy-pasted from Word, typed on an Irish keyboard, etc.).

    Raises
    ------
    TypeError  : if text is not a str (including None).
    """
    if not isinstance(text, str):
        raise TypeError(
            f"layer02.run(): `text` must be a str, got {type(text).__name__!r}."
        )
    return unicodedata.normalize("NFC", text)


def _validate_ordis_id(ordis_id: Any) -> str:
    """
    Sanitise ordis_id to prevent path-traversal attacks and injection.

    Allowed characters: ASCII alphanumerics, hyphens, underscores, dots.
    An empty string is allowed (means "don't persist").

    Raises
    ------
    TypeError  : if ordis_id is not a str.
    ValueError : if ordis_id contains unsafe characters or is too long.
    """
    if not isinstance(ordis_id, str):
        raise TypeError(
            f"layer02.run(): `ordis_id` must be a str, got {type(ordis_id).__name__!r}."
        )
    if not ordis_id:
        return ""
    if len(ordis_id) > _MAX_ORDIS_ID_LENGTH:
        raise ValueError(
            f"ordis_id is too long ({len(ordis_id)} chars; "
            f"maximum is {_MAX_ORDIS_ID_LENGTH})."
        )
    if not re.fullmatch(r"[A-Za-z0-9_\-\.]+", ordis_id):
        raise ValueError(
            f"ordis_id {ordis_id!r} contains unsafe characters. "
            "Only ASCII alphanumerics, hyphens, underscores, and dots are permitted."
        )
    return ordis_id


def _sanitise_registry(names: List[Any]) -> List[str]:
    """
    Filter a name registry to non-empty, NFC-normalised strings.

    Silently drops None values, empty strings, and non-string entries so that
    downstream fuzzy-match calls never receive unexpected types.
    """
    result: List[str] = []
    for n in (names or []):
        if isinstance(n, str) and n.strip():
            result.append(unicodedata.normalize("NFC", n.strip()))
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Text replacement
# ─────────────────────────────────────────────────────────────────────────────

def _replace_name_in_text(text: str, name: str, token: str) -> str:
    """
    Replace all case-insensitive whole-word occurrences of *name* with *token*.

    Handles all of the following surface forms:
    ┌─────────────────────────────┬────────────────────────────────────────────┐
    │ Surface in text             │ After replacement                          │
    ├─────────────────────────────┼────────────────────────────────────────────┤
    │ Marcus                      │ PATIENT1                                   │
    │ marcus  (lower-case)        │ PATIENT1                                   │
    │ MARCUS  (upper-case)        │ PATIENT1                                   │
    │ Marcus's                    │ PATIENT1's  (possessive 's kept)           │
    │ Marcus'  (Irish possessive) │ PATIENT1'   (bare apostrophe kept)         │
    │ O'Reilly                    │ PATIENT2                                   │
    │ O'Reilly's                  │ PATIENT2's                                 │
    │ Smith-Jones                 │ PATIENT3                                   │
    ├─────────────────────────────┼────────────────────────────────────────────┤
    │ Johnsonfield  (no match)    │ unchanged  (mid-word match suppressed)     │
    │ O'Reilly (only "Reilly")    │ "Reilly" inside "O'Reilly" not touched     │
    └─────────────────────────────┴────────────────────────────────────────────┘

    Boundary characters treated as name-internal (suppressed in look-around):
        apostrophe ('), hyphen (-), ASCII letters and digits.

    Implementation
    --------------
    Two regex passes keep the logic simple and correct:
      Pass 1 — possessives: matches "Name'[s]" and keeps the suffix.
      Pass 2 — standalone: matches bare "Name" with word boundaries.
    """
    if not name or not token:
        return text

    escaped = re.escape(name)

    # ── Pass 1: possessive forms ─────────────────────────────────────────────
    # Pattern: word-boundary before name, then optional "'s" or "'" suffix.
    # The suffix is captured in group 1 so we can re-append it after the token.
    poss_pattern = (
        r"(?<![A-Za-z0-9\-])"
        + escaped
        + r"('s?)(?![A-Za-z0-9\-])"
    )
    text = re.sub(
        poss_pattern,
        lambda m: token + m.group(1),
        text,
        flags=re.IGNORECASE,
    )

    # ── Pass 2: standalone occurrences ──────────────────────────────────────
    # Negative look-around includes "'" to prevent matching "Reilly" inside
    # "O'Reilly" (apostrophe is name-internal), and "-" to prevent matching
    # "Smith" inside "Smith-Jones".
    main_pattern = (
        r"(?<![A-Za-z0-9'\-])"
        + escaped
        + r"(?![A-Za-z0-9'\-])"
    )
    text = re.sub(main_pattern, token, text, flags=re.IGNORECASE)

    return text


# ─────────────────────────────────────────────────────────────────────────────
#  Honorific stripping
# ─────────────────────────────────────────────────────────────────────────────

def _strip_honorific(span: str) -> str:
    """
    Remove a leading honorific / clinical title from a name span.

    Examples
    --------
    "Mr. Marcus O'Reilly"  →  "Marcus O'Reilly"
    "Nurse Dharani Kumar"  →  "Dharani Kumar"
    "Jennifer Davis"       →  "Jennifer Davis"   (unchanged)
    "Dr"                   →  "Dr"               (single token, never stripped)
    """
    parts = span.split()
    if len(parts) < 2:
        return span
    candidate = parts[0].rstrip(".").lower()
    if candidate in _HONORIFICS:
        return " ".join(parts[1:])
    return span


# ─────────────────────────────────────────────────────────────────────────────
#  Span deduplication  (sub-span suppression + longest-first ordering)
# ─────────────────────────────────────────────────────────────────────────────

def _dedup_spans(spans: List[str]) -> List[str]:
    """
    Given a list of surface-string candidates, remove any span that is a
    proper sub-string of another span in the list.  Remaining spans are
    sorted longest-first so that compound names ("Jennifer Davis") are
    replaced in the text before their component words ("Jennifer", "Davis"),
    preventing partial-token artefacts.

    Examples
    --------
    ["Jennifer", "Jennifer Davis", "Davis"]  →  ["Jennifer Davis"]
    ["Dharani", "Dharani Kumar"]             →  ["Dharani Kumar"]
    ["Marcus O'Reilly", "Liam"]              →  ["Marcus O'Reilly", "Liam"]
    """
    # Exact-duplicate removal while preserving first-seen order
    seen: dict = {}
    for s in spans:
        key = s.lower()
        if key not in seen:
            seen[key] = s
    unique = list(seen.values())

    # Sub-span suppression
    result: List[str] = []
    for s in unique:
        lower_s = s.lower()
        dominated = any(
            lower_s != other.lower() and lower_s in other.lower()
            for other in unique
        )
        if not dominated:
            result.append(s)

    # Longest-first to prevent partial replacements
    result.sort(key=len, reverse=True)
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Part-name expansion  (first-name alias registration)
# ─────────────────────────────────────────────────────────────────────────────

def _first_name_candidates(canonical: str) -> List[str]:
    """
    Extract the first-name token from a canonical full name for alias
    registration.  This ensures that bare "Jennifer" is caught in the text
    after "Jennifer Davis" has been assigned PATIENT1.

    Rules (conservative — minimises false positives)
    ------------------------------------------------
    · Only the first token is considered (not the surname).
    · Minimum 4 characters (prevents "Jo", "Li", etc. from polluting text).
    · Not in _NER_STOPWORDS.
    · Not returned for single-token names (the full name is the first name).

    Returns an empty list when no safe candidate is found.
    """
    parts = canonical.split()
    if len(parts) < 2:
        return []
    first = parts[0]
    if len(first) >= 4 and first.lower() not in _NER_STOPWORDS:
        return [first]
    return []


# ─────────────────────────────────────────────────────────────────────────────
#  Presidio span extraction
# ─────────────────────────────────────────────────────────────────────────────

def _presidio_person_spans(text: str) -> List[str]:
    """
    Use Presidio to extract PERSON entity spans from *text*.

    Post-processing
    ---------------
    · Strips leading honorifics from each span.
    · Filters out stopwords and spans shorter than _MIN_NAME_LENGTH.
    · Removes sub-spans dominated by longer spans (_dedup_spans).
    · Sorts longest-first.

    Reliability
    -----------
    Wraps the Presidio call in a two-attempt retry loop so that a single
    transient NLP engine error does not abort the entire request.

    Returns a deduplicated list of surface strings, or [] if Presidio is
    unavailable or both retry attempts fail.
    """
    engine = _get_presidio_engine()
    if engine is None:
        return []

    results = None
    for attempt in range(2):
        try:
            results = engine.analyze(text=text, language="en", entities=["PERSON"])
            break
        except Exception as exc:
            if attempt == 0:
                logger.warning(
                    "Presidio analyze() failed (attempt 1/2): %s — retrying.", exc
                )
            else:
                logger.error(
                    "Presidio analyze() failed on both attempts: %s. "
                    "Falling back to registry-only detection.",
                    exc,
                )
                return []

    if results is None:
        return []

    raw: List[str] = []
    for r in results:
        surface = text[r.start : r.end].strip()
        # Strip leading honorific before filtering
        surface = _strip_honorific(surface)
        if not surface or len(surface) < _MIN_NAME_LENGTH:
            continue
        low = surface.lower()
        if low in _NER_STOPWORDS:
            continue
        if all(w in _NER_STOPWORDS for w in low.split()):
            continue
        raw.append(surface)

    return _dedup_spans(raw)


# ─────────────────────────────────────────────────────────────────────────────
#  Registry role lookup
# ─────────────────────────────────────────────────────────────────────────────

def _registry_role(
    span:          str,
    patient_names: List[str],
    nurse_names:   List[str],
    threshold:     int = _CONFIRM_THRESHOLD,
) -> Tuple[Optional[str], Optional[str], float]:
    """
    Confirm a PERSON span against the patient and nurse registries.

    Strategy
    --------
    Two candidate forms are tried in order:
      1. The span as-is.
      2. The honorific-stripped variant (e.g. "Nurse Dharani" → "Dharani").

    For each form, both registries are queried.  The match with the higher
    confidence score wins.  The patient registry is preferred in a tie
    (reflecting the clinical context where most mentions are patients).

    Returns
    -------
    (matched_name, role, confidence) where role ∈ {"patient", "nurse"},
    or (None, None, 0.0) when the span is not confirmed by any registry.
    """
    candidates = [span]
    stripped   = _strip_honorific(span)
    if stripped.lower() != span.lower():
        candidates.append(stripped)

    for candidate in candidates:
        match_p, conf_p = match_name(candidate, patient_names, threshold=threshold)
        match_n, conf_n = match_name(candidate, nurse_names,   threshold=threshold)

        if match_p and match_n:
            if conf_p >= conf_n:
                return match_p, "patient", conf_p
            return match_n, "nurse", conf_n
        if match_p:
            return match_p, "patient", conf_p
        if match_n:
            return match_n, "nurse", conf_n

    return None, None, 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  Fallback: registry-only detection  (Presidio not installed)
# ─────────────────────────────────────────────────────────────────────────────

def _registry_only_detect(
    text:          str,
    patient_names: List[str],
    nurse_names:   List[str],
    threshold:     int = _FALLBACK_THRESHOLD,
) -> List[Tuple[str, str, str, float]]:
    """
    Conservative word-window scan when Presidio is unavailable.

    Scans windows of 1–4 tokens (longest-first, greedy) and fuzzy-matches
    against both registries.  Strict filters suppress the false positives
    that affected the original sliding-window approach.

    Filters applied
    ---------------
    · Span must begin with an uppercase letter (names do; common words don't).
    · Minimum 4 characters.
    · Not a stopword or all-stopword multi-word span.
    · Leading honorifics are stripped before matching.
    · Confidence threshold raised to _FALLBACK_THRESHOLD (default 78).

    Returns
    -------
    List of (surface_span, matched_name, role, confidence).
    Greedy coverage: once an index is consumed by a match it won't be
    re-used by a shorter window.
    """
    found:   List[Tuple[str, str, str, float]] = []
    words    = text.split()
    covered: set = set()

    for window in (4, 3, 2, 1):
        for i in range(len(words) - window + 1):
            if any(j in covered for j in range(i, i + window)):
                continue

            raw_span = " ".join(words[i : i + window])
            span     = _strip_honorific(raw_span)

            if not span or not span[0].isupper():
                continue
            low = span.lower()
            if low in _NER_STOPWORDS or len(span) < 4:
                continue
            if all(w in _NER_STOPWORDS for w in low.split()):
                continue

            match_p, conf_p = match_name(span, patient_names, threshold=threshold)
            match_n, conf_n = match_name(span, nurse_names,   threshold=threshold)

            if match_p or match_n:
                if match_p and (not match_n or conf_p >= conf_n):
                    found.append((span, match_p, "patient", conf_p))
                else:
                    found.append((span, match_n, "nurse",   conf_n))
                for j in range(i, i + window):
                    covered.add(j)

    return found


# ─────────────────────────────────────────────────────────────────────────────
#  Token counter (per-run, not a module-level mutable)
# ─────────────────────────────────────────────────────────────────────────────

class _TokenCounter:
    """
    Mints sequential PATIENT / NURSE tokens for a single run() invocation.

    Keeping the counter inside run() instead of at module level means
    concurrent requests each get their own independent counter, and
    tests can be run in any order without cross-contamination.
    """
    __slots__ = ("_patient", "_nurse")

    def __init__(self) -> None:
        self._patient = 0
        self._nurse   = 0

    def next(self, role: str) -> str:
        if role == "nurse":
            self._nurse += 1
            return f"{_NURSE_PREFIX}{self._nurse}"
        # Default to PATIENT for "patient" and for no-registry "patient" fallback
        self._patient += 1
        return f"{_PATIENT_PREFIX}{self._patient}"


# ─────────────────────────────────────────────────────────────────────────────
#  Main run()
# ─────────────────────────────────────────────────────────────────────────────

def run(
    text:     str,
    accent:   str = "ml_In",
    ordis_id: str = "",
    cfg:      Optional[ConfigManager] = None,
    mongo_db: Optional[Any] = None,
) -> Tuple[str, Dict[str, str]]:
    """
    Layer 02 — PII Redaction (Presidio Edition · v2).

    Parameters
    ----------
    text      : Text from Layer 01 (corrected names).  Must be a non-None str.
                Unicode-normalised to NFC internally; the caller need not
                pre-process it.  An empty / whitespace-only string is returned
                unchanged without error.
    accent    : Accent profile (reserved; not used in L2, kept for API parity).
    ordis_id  : Session identifier used as the key for the saved PII-map file.
                Must be the **same** ID that is later passed to Layer 04.
                Only ASCII alphanumerics, hyphens, underscores, and dots
                are accepted.  Pass an empty string to skip persistence.
    cfg       : ConfigManager instance.  Auto-created if None.
    mongo_db  : Optional MongoDB connection forwarded to load_name_registry().

    Returns
    -------
    (redacted_text, pii_map)
      redacted_text : Original text with all confirmed names replaced by tokens.
      pii_map       : Mapping of token → canonical name, e.g.
                      {"PATIENT1": "Marcus O'Reilly", "NURSE1": "Dharani Kumar"}

    Side-effects
    ------------
    · Writes  output/pii_map_{ordis_id}.json  when ordis_id is provided.
    · Emits structured log messages at INFO / DEBUG level via the module
      logger ("ordis.layer02" when the package is properly configured).

    Raises
    ------
    TypeError  : text or ordis_id is not a str (including None).
    ValueError : ordis_id contains path-traversal or unsafe characters,
                 or is longer than _MAX_ORDIS_ID_LENGTH.
    """
    # ── 1. Validate inputs ───────────────────────────────────────────────────
    text     = _validate_text(text)
    ordis_id = _validate_ordis_id(ordis_id)

    if not text.strip():
        logger.info("L2: received empty text — returning unchanged.")
        if ordis_id:
            save_pii_map(ordis_id, {})
        return text, {}

    if cfg is None:
        cfg = ConfigManager()

    # ── 2. Load and sanitise registries ─────────────────────────────────────
    _patient_raw, _nurse_raw = load_name_registry(cfg, mongo_db=mongo_db)
    patient_names = _sanitise_registry(_patient_raw)
    nurse_names   = _sanitise_registry(_nurse_raw)
    registry_available = bool(patient_names or nurse_names)

    logger.info(
        "L2: registry loaded — %d patient(s), %d nurse(s).",
        len(patient_names), len(nurse_names),
    )

    # ── 3. Shared redaction state ────────────────────────────────────────────
    pii_map:       Dict[str, str] = {}   # token        → canonical name
    name_to_token: Dict[str, str] = {}   # name.lower() → token  (dedup + aliases)
    counter  = _TokenCounter()
    audit:   List[RedactionEvent] = []
    result   = text

    # ── 4. Inner helpers ─────────────────────────────────────────────────────

    def _assign_or_reuse(canonical: str, role: str) -> str:
        """Return the existing token for *canonical*, or mint a new one."""
        key = canonical.lower()
        if key not in name_to_token:
            token              = counter.next(role)
            name_to_token[key] = token
            pii_map[token]     = canonical
        return name_to_token[key]

    def _apply_redaction(
        surface:   str,
        canonical: str,
        role:      str,
        confidence: float,
        method:    str,
    ) -> None:
        """
        Perform all text replacements for a confirmed name and record
        the audit event.

        Replacement order
        -----------------
        1. Canonical name (authoritative registry form, e.g. "Marcus O'Reilly").
        2. Surface span, when it differs from canonical (e.g. "Mr. O'Reilly",
           a Presidio-detected variant).
        3. First-name alias (e.g. "Marcus"), registered so that bare first
           names occurring later in the same transcript are also caught.
        """
        nonlocal result
        token  = _assign_or_reuse(canonical, role)
        event  = RedactionEvent(
            token=token, canonical_name=canonical, surface_span=surface,
            role=role, confidence=confidence, method=method,
        )
        audit.append(event)
        logger.info(
            "L2 redact [%s] '%s' → %s  (conf %.2f, method=%s)",
            role, canonical, token, confidence, method,
        )

        # Replace canonical form
        result = _replace_name_in_text(result, canonical, token)

        # Replace surface span if it is a different string
        if surface.lower() != canonical.lower():
            result = _replace_name_in_text(result, surface, token)
            logger.debug("  L2 surface alias: '%s' → %s", surface, token)

        # Register and replace first-name alias
        for fn in _first_name_candidates(canonical):
            fn_key = fn.lower()
            if fn_key not in name_to_token:
                name_to_token[fn_key] = token   # alias — same token, no new pii_map entry
                result = _replace_name_in_text(result, fn, token)
                event.events.append(f"part-name alias '{fn}' → {token}")
                logger.debug("  L2 part-name alias: '%s' → %s", fn, token)

    # ── 5a. STAGE 1 — Presidio NER detection ────────────────────────────────
    engine = _get_presidio_engine()
    if engine is not None:
        logger.info("L2: Presidio NER active ✔")
        presidio_spans = _presidio_person_spans(text)
        logger.info(
            "L2: Presidio proposed %d PERSON span(s): %s",
            len(presidio_spans), presidio_spans,
        )

        for span in presidio_spans:
            # STAGE 2 — registry confirmation
            if registry_available:
                matched_name, role, conf = _registry_role(
                    span, patient_names, nurse_names
                )
                if matched_name is None:
                    logger.debug("L2 skip (no registry match): '%s'", span)
                    continue
            else:
                # No registry — trust Presidio directly; default role to "patient"
                # (clinical notes are predominantly about patients)
                matched_name = span
                role         = "patient"
                conf         = 95.0   # sentinel: Presidio-only confidence proxy
                logger.debug(
                    "L2 no-registry mode: treating '%s' as patient.", span
                )

            _apply_redaction(span, matched_name, role, conf, method="presidio")

    # ── 5b. Fallback — registry-only (Presidio not installed) ───────────────
    else:
        logger.warning(
            "L2: Presidio unavailable — using registry-only fallback "
            "(conservative mode). "
            "Install: pip install presidio-analyzer presidio-anonymizer spacy && "
            "python -m spacy download en_core_web_lg"
        )
        hits = _registry_only_detect(result, patient_names, nurse_names)
        logger.info("L2 fallback: %d registry hit(s).", len(hits))
        for span, matched_name, role, conf in hits:
            _apply_redaction(span, matched_name, role, conf, method="registry_only")

    # ── 6. Persist PII map ───────────────────────────────────────────────────
    if ordis_id:
        save_pii_map(ordis_id, pii_map)
        logger.info(
            "L2: PII map persisted for Ordis_ID '%s' (%d entries).",
            ordis_id, len(pii_map),
        )
    else:
        logger.warning(
            "L2: No Ordis_ID provided — PII map NOT persisted. "
            "Layer 04 reversal will be unavailable for this request."
        )

    if not pii_map:
        logger.info("L2: No names found to redact in this transcript.")

    logger.debug(
        "L2: audit trail — %d redaction event(s): %s",
        len(audit),
        [(e.token, e.canonical_name, e.method) for e in audit],
    )

    return result, pii_map


# ─────────────────────────────────────────────────────────────────────────────
#  CLI smoke-test  (python layer02.py  or  python layer02.py "custom text")
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Configure logging for the CLI run
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)-8s  %(name)s  %(message)s",
    )

    # ── Representative clinical transcript covering many edge cases ───────────
    SAMPLE = (
        # Full name + room reference
        "Resident Marcus O'Reilly (room 22) was observed with a bottle. "
        # Possessive form
        "Marcus's evening confusion was noted again. "
        # Nurse with honorific
        "Nurse Dharani Kumar reported that O'Reilly's condition is stable. "
        # Multiple patients; one with a two-word name after "patient"
        "Patient Jennifer Davis was reviewed. Liam O'Brien-Walsh requires "
        "one-to-one supervision. "
        # Common-word-like name that the old approach would skip
        "Lisa Chen reported fluid overload. "
        # Bare first name re-occurrence (should map to same token as full name)
        "Jennifer had no oral intake today. "
        # Nurse name appearing standalone after full introduction
        "Dharani escalated to the on-call doctor. "
        # Recording tag
        "Recording for William Oak: the patient refused care."
    )

    text_in  = sys.argv[1] if len(sys.argv) > 1 else SAMPLE
    session  = sys.argv[2] if len(sys.argv) > 2 else "cli_test_001"

    print("\n── INPUT ────────────────────────────────────────────────")
    print(text_in)

    out, pii = run(text_in, ordis_id=session)

    print("\n── REDACTED OUTPUT ──────────────────────────────────────")
    print(out)
    print(f"\n── PII MAP  ({len(pii)} entries) ────────────────────────")
    for token, name in sorted(pii.items()):
        print(f"  {token:<12}  →  {name}")

    # ── Quick self-tests ──────────────────────────────────────────────────────
    print("\n── UNIT ASSERTIONS ──────────────────────────────────────")
    import traceback

    def _assert(condition: bool, label: str) -> None:
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {label}")

    # Possessive regression
    test_poss  = "Marcus's room"
    replaced   = _replace_name_in_text(test_poss, "Marcus", "PATIENT1")
    _assert(replaced == "PATIENT1's room", "possessive 's → PATIENT1's")

    test_poss2 = "O'Reilly's condition"
    replaced2  = _replace_name_in_text(test_poss2, "O'Reilly", "PATIENT2")
    _assert(replaced2 == "PATIENT2's condition", "apostrophe name possessive")

    # Mid-word non-match
    test_mid   = "Johnsonfield"
    replaced3  = _replace_name_in_text(test_mid, "Johnson", "PATIENT3")
    _assert(replaced3 == "Johnsonfield", "mid-word match suppressed")

    # O'Reilly partial-match guard
    test_part  = "O'Reilly was here"
    replaced4  = _replace_name_in_text(test_part, "Reilly", "PATIENT4")
    _assert(replaced4 == "O'Reilly was here", "partial match inside O'Reilly blocked")

    # Honorific stripping
    _assert(_strip_honorific("Mr. Marcus O'Reilly") == "Marcus O'Reilly", "strip Mr.")
    _assert(_strip_honorific("Nurse Dharani Kumar") == "Dharani Kumar",   "strip Nurse")
    _assert(_strip_honorific("Jennifer Davis")      == "Jennifer Davis",  "no honorific unchanged")
    _assert(_strip_honorific("Dr")                  == "Dr",              "single-token not stripped")

    # Span deduplication
    deduped = _dedup_spans(["Jennifer", "Jennifer Davis", "Davis"])
    _assert(deduped == ["Jennifer Davis"], "sub-span dedup: Jennifer Davis dominates")

    deduped2 = _dedup_spans(["Dharani", "Dharani Kumar"])
    _assert(deduped2 == ["Dharani Kumar"], "sub-span dedup: Dharani Kumar dominates")

    deduped3 = _dedup_spans(["Marcus O'Reilly", "Liam"])
    _assert(
        set(deduped3) == {"Marcus O'Reilly", "Liam"},
        "independent spans both kept",
    )

    # Input validation
    try:
        run(None)                             # type: ignore[arg-type]
        _assert(False, "TypeError raised for None text")
    except TypeError:
        _assert(True, "TypeError raised for None text")

    try:
        run("ok", ordis_id="../etc/passwd")
        _assert(False, "ValueError raised for path-traversal ordis_id")
    except ValueError:
        _assert(True, "ValueError raised for path-traversal ordis_id")

    try:
        run("ok", ordis_id="valid_id-001.session")
        _assert(True, "valid ordis_id accepted without error")
    except Exception:
        _assert(False, "valid ordis_id accepted without error")

    # Unicode normalization: NFC vs NFD form of "é" in "José"
    nfd_name = unicodedata.normalize("NFD", "José")   # combining form
    nfc_name = "José"                                  # precomposed
    _assert(
        _validate_text(nfd_name) == nfc_name,
        "NFD input normalised to NFC",
    )

    # Registry sanitiser
    dirty = [None, "", "  ", 123, "Dharani Kumar", "  Jennifer  "]  # type: ignore[list-item]
    clean = _sanitise_registry(dirty)
    _assert(
        clean == ["Dharani Kumar", "Jennifer"],
        "registry sanitiser strips None / non-str / empty",
    )

    # first_name_candidates
    _assert(_first_name_candidates("Jennifer Davis") == ["Jennifer"], "first name extracted")
    _assert(_first_name_candidates("Liam")           == [],           "single token → no alias")
    _assert(_first_name_candidates("Li Chen")        == [],           "too short first name skipped")

    # Hyphenated name
    hyph = _replace_name_in_text("Liam O'Brien-Walsh was seen", "O'Brien-Walsh", "PATIENT5")
    _assert(hyph == "Liam PATIENT5 was seen", "hyphenated name replaced")

    # Hyphenated boundary guard — "Smith" should not match "Smith-Jones"
    hyph2 = _replace_name_in_text("Smith-Jones arrived", "Smith", "PATIENT6")
    _assert(hyph2 == "Smith-Jones arrived", "Smith not matched inside Smith-Jones")

    print("\n── Done. ────────────────────────────────────────────────\n")