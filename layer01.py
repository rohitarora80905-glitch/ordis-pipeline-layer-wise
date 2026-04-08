"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ORDIS — layer01.py                                                          ║
║  Layer 01 · Name Correction                                                  ║
║                                                                              ║
║  Input  : Raw voice-transcribed text                                         ║
║  Output : Same text with nurse/patient names corrected to canonical form     ║
║                                                                              ║
║  Method : ml-IN phonetic normalisation → fuzzy + phonetic registry lookup   ║
║           (Soundex, Metaphone, WRatio bijective token scoring)               ║
║                                                                              ║
║  API    : POST /api/layer01  (txt, accent, Ordis_ID)                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

What this layer does
─────────────────────
  "Maacuus O'Rielly"  → "Marcus O'Reilly"
  "Willeem Wooak"     → "William Oak"
  "Nurs Dharanee"     → "Nurse Dharani"

The registry is loaded from:
  data/patients.csv  (column: name)
  data/nurses.csv    (column: name)

Names NOT found in the registry are left as-is (no hallucination risk).

Design notes
─────────────
  All correction discovery is READ-ONLY (two-phase approach):
    Phase 1 — collect a {wrong_form → canonical} map by scanning the text.
    Phase 2 — apply every correction in a single regex pass.

  This eliminates the position-drift and wrong-occurrence bugs that arise
  when text is mutated mid-scan.

  Corrections are applied longest-first so that "Mary Jane Smith" is not
  partially shadowed by a shorter match for "Mary Jane".
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from shared import (
    ConfigManager,
    load_name_registry,
    match_name,
    mlin_normalise_name,
)

__all__ = ["run", "Layer01Result", "NameCorrection"]

log = logging.getLogger(__name__)


# ── Tuneable constants ────────────────────────────────────────────────────────

_MAX_WINDOW          = 3   # max tokens considered as a single name fragment
_MIN_FRAGMENT_LEN    = 2   # fragments shorter than this are not sent to matcher
_MIN_SCAN_FRAG_LEN   = 4   # stricter floor for context-free window scan
_SCAN_THRESHOLD_BUMP = 5   # extra confidence required for trigger-less matches


# ── Compiled patterns ─────────────────────────────────────────────────────────

# Words that introduce a name in clinical transcripts.
# Handles "Dr.", "Dr", "Mr.", "Mrs.", "Ms." etc.
_NAME_TRIGGERS = re.compile(
    r"\b(?:nurse|patient|resident|doctor|dr\.?|mr\.?|mrs\.?|ms\.?|staff)\b\s*",
    re.IGNORECASE,
)

# Name-token pattern: matches a single name, including compound forms.
# Handles: O'Brien  Mary-Jane  Ó'Súilleabháin  Van der Berg
# Unicode letter range covers accented / non-ASCII names (Irish, Indian, etc.).
_NAME_TOKEN_RE = re.compile(
    r"[A-Za-zÀ-ÖØ-öø-ÿ\u0100-\u024F]+"
    r"(?:['\u2019\u02BC\-][A-Za-zÀ-ÖØ-öø-ÿ\u0100-\u024F]+)*"
)

# Characters to strip from fragment boundaries before matching.
_PUNCT_CHARS = r""".,;:!?()\[\]{}"'"""
_PUNCT_STRIP = str.maketrans("", "", _PUNCT_CHARS)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class NameCorrection:
    """
    Immutable record of a single name correction.

    Attributes
    ----------
    original   : The misspelled/misheard fragment found in the transcript.
    canonical  : The authoritative registry form.
    confidence : Fuzzy-match confidence score (0–100).
    source     : How the correction was found.
                 One of: "trigger:nurse", "trigger:patient",
                          "scan:nurse",    "scan:patient".
    """
    original:   str
    canonical:  str
    confidence: float
    source:     str

    def __str__(self) -> str:
        return (
            f"'{self.original}' → '{self.canonical}' "
            f"(conf {self.confidence:.0f}, {self.source})"
        )


@dataclass
class Layer01Result:
    """
    Full structured result from Layer 01.

    Iterable as ``(corrected_text, list_of_(orig, canonical) tuples)``
    for backward compatibility with callers that do::

        out, corrections = run(text)
    """
    text:              str
    corrections:       List[NameCorrection] = field(default_factory=list)
    accent_normalised: bool                 = False

    # ── Backward-compatible 2-tuple unpacking ─────────────────────────────────
    def __iter__(self) -> Iterator:
        yield self.text
        yield [(c.original, c.canonical) for c in self.corrections]

    # ── Convenience helpers ───────────────────────────────────────────────────
    @property
    def correction_map(self) -> Dict[str, str]:
        """Return {original: canonical} for all corrections."""
        return {c.original: c.canonical for c in self.corrections}

    @property
    def was_modified(self) -> bool:
        return bool(self.corrections) or self.accent_normalised


# ── Internal helpers ──────────────────────────────────────────────────────────

def _normalise_unicode(text: str) -> str:
    """
    NFC-normalise *text* so composed and decomposed Unicode forms compare equal.
    e.g. ``e\u0301`` (e + combining acute) → ``é`` (U+00E9).
    """
    return unicodedata.normalize("NFC", text)


def _strip_punct(s: str) -> str:
    """Remove leading/trailing punctuation from a name fragment."""
    return s.translate(_PUNCT_STRIP).strip()


def _extract_name_tokens(text: str, max_tokens: int = _MAX_WINDOW) -> List[str]:
    """
    Extract up to *max_tokens* name-like tokens from the **start** of *text*.

    Compound name forms are treated as a single token:
      - Apostrophe variants: O'Brien, O\u2019Brien, O\u02BCBrien
      - Hyphenated:          Mary-Jane, Van-der-Berg
      - Accented:            Seán, Áine, Dhruv
    """
    return _NAME_TOKEN_RE.findall(text)[:max_tokens]


def _build_replacement_pattern(fragments: List[str]) -> Optional[re.Pattern]:
    """
    Compile a single regex that matches any of *fragments* as a whole-word
    occurrence (case-insensitive).

    Fragments are sorted longest-first so that "Mary Jane Smith" is never
    partially shadowed by "Mary Jane".

    Returns ``None`` when *fragments* is empty.
    """
    if not fragments:
        return None

    # Sort longest first to prevent shorter matches shadowing longer ones.
    alts = sorted(fragments, key=len, reverse=True)
    # Use Unicode-aware word boundaries via lookbehind / lookahead.
    # \W matches non-word; ^ / $ anchor to string edges.
    pattern = "|".join(
        r"(?<!\w)" + re.escape(f) + r"(?!\w)" for f in alts
    )
    return re.compile(pattern, re.IGNORECASE | re.UNICODE)


def _apply_corrections(text: str, correction_map: Dict[str, str]) -> str:
    """
    Replace **all** occurrences of every key in *correction_map* with its
    canonical value in a **single regex pass**.

    This is the critical correctness guarantee: text is never mutated
    between discovery and application, so there is no position-drift
    and no risk of a replacement being used as input for another match.
    """
    if not correction_map:
        return text

    pat = _build_replacement_pattern(list(correction_map.keys()))
    if pat is None:
        return text

    # Build a normalised lookup so the replacer is case-insensitive.
    lower_map: Dict[str, str] = {k.lower(): v for k, v in correction_map.items()}

    def _replacer(m: re.Match) -> str:
        return lower_map.get(m.group(0).lower(), m.group(0))

    return pat.sub(_replacer, text)


# ── Phase 1: READ-ONLY correction discovery ───────────────────────────────────

def _collect_trigger_corrections(
    text:      str,
    registry:  List[str],
    threshold: int,
    label:     str,
) -> List[NameCorrection]:
    """
    Scan *text* for trigger words (Nurse, Patient, Resident, …) and attempt
    to resolve the following 1–3 name tokens against *registry*.

    **This function is strictly read-only** — it never modifies *text*.

    Algorithm
    ---------
    For each trigger match, extract up to ``_MAX_WINDOW`` tokens and try
    windows from longest to shortest.  Stop at the first window that
    produces a registry match (regardless of whether a correction is
    needed), to avoid a shorter partial match overriding a full-name match.

    Parameters
    ----------
    text      : The (possibly accent-normalised) transcript text.
    registry  : Flat list of canonical names to match against.
    threshold : Minimum fuzzy-match confidence (0–100).
    label     : "nurse" or "patient" (for logging/source tagging).

    Returns
    -------
    List of :class:`NameCorrection` objects (may be empty).
    """
    corrections: List[NameCorrection] = []
    seen_lower:  Set[str]             = set()   # avoid duplicate entries

    for m in _NAME_TRIGGERS.finditer(text):
        after_trigger = text[m.end():]
        tokens = _extract_name_tokens(after_trigger, _MAX_WINDOW)

        if not tokens:
            log.debug(
                "L1 %s: trigger at pos %d — no name tokens follow", label, m.start()
            )
            continue

        # Try progressively shorter windows until a registry match is found.
        matched = False
        for window_size in range(min(_MAX_WINDOW, len(tokens)), 0, -1):
            raw_frag = " ".join(tokens[:window_size])
            fragment = _strip_punct(raw_frag)

            if len(fragment) < _MIN_FRAGMENT_LEN:
                continue

            frag_key = fragment.lower()
            if frag_key in seen_lower:
                matched = True
                break

            resolved, conf = match_name(fragment, registry, threshold=threshold)

            if resolved is None:
                # No match even at this threshold; try a smaller window.
                continue

            if resolved.lower() == frag_key:
                log.debug(
                    "L1 %s trigger: '%s' already canonical (conf %.0f)",
                    label, fragment, conf,
                )
            else:
                log.debug(
                    "L1 %s trigger: '%s' → '%s' (conf %.0f)",
                    label, fragment, resolved, conf,
                )
                corrections.append(NameCorrection(
                    original   = fragment,
                    canonical  = resolved,
                    confidence = conf,
                    source     = f"trigger:{label}",
                ))

            seen_lower.add(frag_key)
            matched = True
            break   # Do not try a shorter window once any match found.

        if not matched:
            log.debug(
                "L1 %s: trigger at pos %d — no registry match for tokens %r",
                label, m.start(), tokens,
            )

    return corrections


def _collect_scan_corrections(
    text:             str,
    patient_registry: List[str],
    nurse_registry:   List[str],
    threshold:        int,
    known_originals:  Set[str],
    known_canonicals: Set[str],
) -> List[NameCorrection]:
    """
    Context-free window scan: slide a 2–3 token window over *text* and
    correct fragments that fuzzy-match a registry entry but were **not**
    already caught by the trigger scan.

    A higher confidence floor (``threshold + _SCAN_THRESHOLD_BUMP``) is
    enforced here to limit false positives from context-free matching.

    Parameters
    ----------
    text              : Text to scan (never modified).
    patient_registry  : Canonical patient names.
    nurse_registry    : Canonical nurse names.
    threshold         : Base confidence threshold from config.
    known_originals   : Lower-cased fragments already resolved by trigger scan.
    known_canonicals  : Lower-cased canonical forms already in the correction
                        map (to avoid re-processing them as new fragments).

    Returns
    -------
    List of :class:`NameCorrection` objects (may be empty).
    """
    corrections:  List[NameCorrection] = []
    seen_lower:   Set[str]             = set()
    scan_thresh   = threshold + _SCAN_THRESHOLD_BUMP

    tokens = _NAME_TOKEN_RE.findall(text)

    # Iterate windows longest-first to match multi-word names before their parts.
    for window_size in (_MAX_WINDOW, 2):
        for i in range(len(tokens) - window_size + 1):
            raw_frag = " ".join(tokens[i : i + window_size])
            fragment = _strip_punct(raw_frag)

            if len(fragment) < _MIN_SCAN_FRAG_LEN:
                continue

            frag_key = fragment.lower()

            # Skip: already seen in this scan
            if frag_key in seen_lower:
                continue
            # Skip: already handled by trigger scan
            if frag_key in known_originals:
                seen_lower.add(frag_key)
                continue
            # Skip: this IS a canonical form we already know about
            # (prevents re-correcting a name we just resolved)
            if frag_key in known_canonicals:
                seen_lower.add(frag_key)
                continue

            # Patient registry has priority in scan (broader population).
            for registry, lbl in (
                (patient_registry, "patient"),
                (nurse_registry,   "nurse"),
            ):
                resolved, conf = match_name(
                    fragment, registry, threshold=scan_thresh
                )

                if resolved is None:
                    continue

                if resolved.lower() == frag_key:
                    log.debug(
                        "L1 scan %s: '%s' already canonical (conf %.0f)",
                        lbl, fragment, conf,
                    )
                    seen_lower.add(frag_key)
                    break

                log.debug(
                    "L1 scan %s: '%s' → '%s' (conf %.0f)",
                    lbl, fragment, resolved, conf,
                )
                corrections.append(NameCorrection(
                    original   = fragment,
                    canonical  = resolved,
                    confidence = conf,
                    source     = f"scan:{lbl}",
                ))
                seen_lower.add(frag_key)
                break   # Stop at first registry that gives a match.

    return corrections


# ── Public API ────────────────────────────────────────────────────────────────

def run(
    text:     str,
    accent:   str = "ml_In",
    ordis_id: str = "",
    cfg:      Optional[ConfigManager] = None,
    mongo_db: Optional[Any]           = None,
) -> Layer01Result:
    """
    Layer 01 — Name Correction.

    Corrects nurse and patient names in a raw clinical voice transcript by
    fuzzy-matching against authoritative name registries.

    Parameters
    ----------
    text      : Raw voice-transcribed clinical note.
    accent    : Accent profile (default ``'ml_In'`` — Malayalam-influenced
                English; governs which phonetic normalisation rules apply).
    ordis_id  : Session / request ID used for structured log correlation.
                No state is written in L1.
    cfg       : :class:`ConfigManager` instance.  Created from ``config.yaml``
                automatically if not supplied.
    mongo_db  : Optional MongoDB handle forwarded to :func:`load_name_registry`.

    Returns
    -------
    :class:`Layer01Result`
        Iterable as ``(corrected_text, [(orig, canonical), …])`` for
        backward compatibility.

    Raises
    ------
    TypeError
        If *text* is not a ``str``.

    Notes
    -----
    Edge cases handled
    ~~~~~~~~~~~~~~~~~~
    * Empty / whitespace-only input → returned unchanged, zero corrections.
    * Non-ASCII / accented names (Irish, Indian) → Unicode-aware tokenisation.
    * Compound names with apostrophes or hyphens → treated as a single token.
    * Trailing punctuation on fragments (``"Dharani."``).
    * Same misspelling under multiple trigger words → deduplicated.
    * Misspelling appearing multiple times → **all** occurrences replaced.
    * Empty registries → short-circuits with a warning, no crash.
    * Corrections applied via a **single** regex pass (no position-drift,
      no wrong-occurrence substitution, no cascading replacements).
    """
    tag = ordis_id or "?"

    # ── Input validation ──────────────────────────────────────────────────────
    if not isinstance(text, str):
        raise TypeError(
            f"layer01.run: 'text' must be str, got {type(text).__name__!r}"
        )
    if not text.strip():
        log.info("L1 [%s]: empty/whitespace input — returning unchanged", tag)
        return Layer01Result(text=text)

    text = _normalise_unicode(text)

    # ── Config & registry ─────────────────────────────────────────────────────
    if cfg is None:
        cfg = ConfigManager()

    threshold = cfg.get_name_match_threshold()
    patient_names, nurse_names = load_name_registry(cfg, mongo_db=mongo_db)

    if not patient_names and not nurse_names:
        log.warning(
            "L1 [%s]: both name registries are empty — no corrections possible", tag
        )
        return Layer01Result(text=text)

    # ── Step 1: ml-IN phonetic normalisation ──────────────────────────────────
    # Transforms common Malayalam-accent mishearings at the token level
    # (e.g. "Keeran" → "Ciaran", "Shobha" → "Siobha") before fuzzy matching.
    # Applied to the whole text so downstream token extraction sees normalised
    # forms; the original text is preserved in `text` for diffing.
    text_norm      = mlin_normalise_name(text)
    accent_changed = text_norm != text
    if accent_changed:
        log.debug("L1 [%s]: ml-IN normalisation modified text", tag)

    # ── Step 2: Trigger-based name collection (READ-ONLY) ─────────────────────
    # Both registries are scanned; nurse corrections take precedence when the
    # same fragment is resolved by both (explicit "Nurse X" phrasing is a
    # stronger signal than an incidental name match in the patient registry).
    nurse_corr   = _collect_trigger_corrections(
        text_norm, nurse_names,   threshold=threshold, label="nurse"
    )
    patient_corr = _collect_trigger_corrections(
        text_norm, patient_names, threshold=threshold, label="patient"
    )

    # Merge with deduplication: first occurrence (nurse priority) wins.
    seen_originals: Set[str]           = set()
    trigger_corrections: List[NameCorrection] = []
    for c in nurse_corr + patient_corr:
        key = c.original.lower()
        if key not in seen_originals:
            trigger_corrections.append(c)
            seen_originals.add(key)

    trigger_map = {c.original: c.canonical for c in trigger_corrections}

    # ── Step 3: Context-free window scan (READ-ONLY) ──────────────────────────
    # Catches names mentioned without a trigger word (e.g. "Recording for
    # Willeem Wooak: …").  Uses a stricter confidence threshold to compensate
    # for the lack of structural context.
    known_originals  = {k.lower() for k in trigger_map}
    known_canonicals = {v.lower() for v in trigger_map.values()}

    scan_corrections = _collect_scan_corrections(
        text_norm,
        patient_registry = patient_names,
        nurse_registry   = nurse_names,
        threshold        = threshold,
        known_originals  = known_originals,
        known_canonicals = known_canonicals,
    )

    # ── Step 4: Deduplicate and finalise correction map ───────────────────────
    final_map: Dict[str, str] = {}
    all_corrections: List[NameCorrection] = []

    for c in trigger_corrections + scan_corrections:
        key = c.original.lower()
        if key not in final_map:
            final_map[c.original] = c.canonical
            all_corrections.append(c)
            # Guard against circular corrections (e.g. A→B where B was also
            # identified as wrong somewhere — extremely rare but possible with
            # noisy registries).
            if c.canonical.lower() in final_map:
                log.warning(
                    "L1 [%s]: circular correction risk — '%s' is both a "
                    "canonical form and a pending original; skipping '%s'",
                    tag, c.canonical, c.original,
                )
                del final_map[c.original]
                all_corrections.pop()

    # ── Step 5: Apply ALL corrections in a single regex pass ─────────────────
    # Key correctness property: text_norm is never mutated between discovery
    # and application.  Every correction fires against the same baseline text,
    # so positions never drift and every occurrence is replaced — not just the
    # first one.
    result = _apply_corrections(text_norm, final_map)

    # ── Summary logging ───────────────────────────────────────────────────────
    if all_corrections:
        log.info(
            "L1 [%s]: %d correction(s) — %s",
            tag,
            len(all_corrections),
            " | ".join(str(c) for c in all_corrections),
        )
    else:
        log.info("L1 [%s]: no name corrections needed", tag)

    return Layer01Result(
        text              = result,
        corrections       = all_corrections,
        accent_normalised = accent_changed,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  CLI test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level   = logging.DEBUG,
        format  = "%(levelname)-8s %(name)s — %(message)s",
    )

    _SAMPLES = [
        # Standard trigger-based cases
        "Resident Maacuus O'Rielly (room 22) was observed wandering. "
        "Nurse Dharanee reported on patient DA Dara. "
        "Recording for Willeem Wooak: the patient refused care.",

        # Trigger word at end of string (no tokens follow)
        "The incident was reported by nurse",

        # All-caps name
        "Nurse DHARANEE administered medication at 14:00.",

        # Same misspelling twice — both should be corrected
        "Nurse Dharanee spoke with patient Dharanee's family.",

        # Compound apostrophe name
        "Resident Maacuus O'Rielly was discharged.",

        # Unicode curly apostrophe
        "Resident Maacuus O\u2019Rielly was admitted.",

        # Punctuation bleeding into fragment
        "Nurse Dharanee. The patient was calm.",

        # Empty input
        "",

        # Whitespace-only input
        "   ",

        # No names at all
        "The patient was observed sleeping. Vitals are stable.",
    ]

    samples = [sys.argv[1]] if len(sys.argv) > 1 else _SAMPLES

    for idx, text_in in enumerate(samples, 1):
        print(f"\n{'─' * 66}")
        print(f"  SAMPLE {idx}: {text_in!r}")
        result = run(text_in, ordis_id=f"test-{idx:02d}")
        out, corrections = result          # backward-compat unpacking

        print(f"\n  OUTPUT : {out!r}")
        print(f"  ACCENT NORMALISED: {result.accent_normalised}")
        if corrections:
            print(f"  CORRECTIONS ({len(corrections)}):")
            for c in result.corrections:
                print(f"    {c}")
        else:
            print("  CORRECTIONS: none")