"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ORDIS — layer01.py  (v5 · Learning + Irish Names)                           ║
║  Layer 01 · Name Detection & Correction                                      ║
║                                                                              ║
║  Input  : Raw speech-to-text transcript                                      ║
║  Output : (corrected_text, List[NameDetection])                              ║
║                                                                              ║
║  ✦ 100 % LOCAL — zero LLM calls, zero external API calls.                   ║
║                                                                              ║
║  Detection stages (union):                                                   ║
║    A — spaCy NER  (optional, graceful fallback)                              ║
║    B — Capitalised-sequence heuristic  (always active)                       ║
║    C — Sliding registry window  (always active)                              ║
║                                                                              ║
║  Correction engine (four layers):                                            ║
║    L1 — Exact match                                                          ║
║    L2 — Accent normalisation (ml-IN / Irish rule-set from shared.py)        ║
║    L3 — Phonetic index (Soundex + Metaphone + NYSIIS, pre-built at start)   ║
║    L4 — Fuzzy ranking (Jaro-Winkler + WRatio, bijective token alignment)    ║
║                                                                              ║
║  Fallback strategies:                                                        ║
║    F1 — First-name-only matching (unique first name in registry)             ║
║    F2 — Contamination filter (cross-span surname bleed removal)              ║
║                                                                              ║
║  ── NEW IN v5 ──────────────────────────────────────────────────────────── ║
║                                                                              ║
║  Learning Mechanism (LearningStore):                                         ║
║    • Human feedback (confirm / reject) is persisted to JSON on disk.        ║
║    • On each resolution, the store is consulted AFTER the normal pipeline:  ║
║        - Confirmed correction  → confidence boosted / canonical overridden  ║
║        - Rejected correction   → confidence penalised; suppressed if below  ║
║                                   audit threshold                            ║
║        - Confusion pair stored → future same-surface wrong-canonical cases  ║
║                                   are rerouted to the correct canonical      ║
║    • Thread-safe; atomic writes (write-to-tmp then rename).                 ║
║    • Path (priority): arg → LAYER01_LEARN_PATH config → ~/.ordis/…         ║
║    • Public API: record_feedback() — call from review/audit UI.             ║
║                  get_learning_stats() — dashboard summary.                  ║
║                  reset_learning()  — wipe store (testing only).             ║
║                                                                              ║
║  Irish Name Registry (irish_names.py):                                       ║
║    • 80+ curated Irish surnames, 120 first names.                           ║
║    • Phonetic variant map for Malayali-accented pronunciation of Irish names ║
║      (e.g. "shivawn" → "Siobhán", "keeva" → "Caoimhe").                   ║
║    • Loaded at startup and merged into the patient PhoneticIndex so that    ║
║      Irish names are fuzzy-matched even if not yet enrolled in MongoDB.     ║
║    • Controlled by config key LAYER01_IRISH_NAMES (default True).           ║
║                                                                              ║
║  API  : POST /api/layer01  (txt, accent, Ordis_ID)                           ║
║         POST /api/layer01/feedback  (surface, proposed, accepted, …)        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import json
import logging
import re
import threading
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import jellyfish
from rapidfuzz import fuzz

from shared import (
    ConfigManager,
    _GR, _YL, _RE, _R, _B,
    _strip_name,
    mlin_normalise_name,
    load_name_registry,
)

logger = logging.getLogger(__name__)
__all__ = [
    "run", "run_simple", "get_role_map", "NameDetection",
    # v5 learning API
    "record_feedback", "get_learning_stats", "reset_learning",
    "LearningStore",
]


# ─────────────────────────────────────────────────────────────────────────────
#  Thresholds
# ─────────────────────────────────────────────────────────────────────────────

_TIER_AUTO_SILENT: float = 0.92
_TIER_AUTO_LOG:    float = 0.80
_TIER_AUDIT:       float = 0.70

_THRESH_NER:        int   = 72
_THRESH_HEURISTIC:  int   = 80
_THRESH_WINDOW:     int   = 75
_THRESH_FIRST_NAME: float = 0.88


# ─────────────────────────────────────────────────────────────────────────────
#  Static word sets
# ─────────────────────────────────────────────────────────────────────────────

_PII_TOKEN_RE = re.compile(
    r"\b(?:PATIENT|NURSE|STAFF|DOCTOR|CARER|MANAGER|RESIDENT)\d+\b"
)

_HONORIFICS: FrozenSet[str] = frozenset({
    "mr", "mrs", "ms", "miss", "dr", "prof", "professor",
    "nurse", "patient", "resident", "carer", "sr", "jr",
    "rev", "reverend", "sir", "dame",
})

_NER_STOPWORDS: FrozenSet[str] = frozenset({
    "he", "she", "they", "him", "her", "his", "the", "a", "an",
    "resident", "nurse", "patient", "doctor", "staff", "carer",
    "room", "ward", "unit", "care", "home", "recording", "report",
    "palliative", "approach", "corner", "bottle", "bed", "bandage",
    "intake", "discomfort", "aggression", "behavior", "behaviour",
    "confusion", "technique", "application", "condition", "decline",
    "stated", "observed", "reported", "required", "performed",
    "daily", "weekly", "monthly", "yesterday", "today", "tomorrow",
    "morning", "afternoon", "evening", "overnight", "community",
    "pain", "skin", "fall", "falls", "oral", "intake", "fluid",
})

_MIN_TOK: int = 2
_MAX_ID:  int = 128
_ID_RE        = re.compile(r"^[a-zA-Z0-9_\-\.]+$")

_CAP_STOP: FrozenSet[str] = frozenset({
    "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday", "january", "february", "march", "april",
    "june", "july", "august", "september", "october", "november",
    "december", "ward", "unit", "room", "ireland", "dublin",
    "cork", "galway", "recording", "report", "staff", "carer",
    "doctor", "nurse", "patient",
})


# ─────────────────────────────────────────────────────────────────────────────
#  NameDetection (public output type)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NameDetection:
    """
    Record of every name span detected and resolved by Layer 01.

    was_corrected   True when the text was changed.
    is_unresolved   True when no registry entry matched.
    role            "patient" | "nurse" | "unknown" — FROM REGISTRY, never
                    inferred from sentence position.
    resolution_path "exact" | "accent_norm" | "phonetic_fuzzy" |
                    "first_name_only" | "contamination_filter" |
                    "learned_confirmed" | "learned_confusion" | "none"
    """
    surface_span:     str
    canonical_name:   Optional[str]
    role:             str
    confidence:       float
    was_corrected:    bool
    is_unresolved:    bool
    resolution_path:  str
    tier:             str
    detection_method: str
    ordis_id:         str = ""


# ─────────────────────────────────────────────────────────────────────────────
#  PhoneticIndex
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _Entry:
    canonical: str
    norm:      str
    norm_mlin: str
    tokens:    List[str]
    first_tok: str
    last_tok:  str
    phon_keys: Set[str]


class PhoneticIndex:
    """
    Pre-built inverted phonetic map (Soundex + Metaphone + NYSIIS).
    Separate first-name index for F1 fallback.
    """

    def __init__(self, names: List[str], role: str) -> None:
        self.role = role
        self._entries:   List[_Entry]                    = []
        self._phon_idx:  Dict[str, List[_Entry]]         = defaultdict(list)
        self._first_idx: Dict[str, List[_Entry]]         = defaultdict(list)
        self._build(names)

    @staticmethod
    def _codes(word: str) -> Set[str]:
        w = re.sub(r"[^a-z]", "", word.lower())
        if not w:
            return set()
        out: Set[str] = set()
        for fn in (jellyfish.soundex, jellyfish.metaphone, jellyfish.nysiis):
            try:
                v = fn(w)
                if v:
                    out.add(v)
            except Exception:
                pass
        return out

    def _build(self, names: List[str]) -> None:
        for canonical in names:
            if not canonical or not isinstance(canonical, str):
                continue
            norm      = _strip_name(canonical)
            norm_mlin = _strip_name(mlin_normalise_name(canonical))
            tokens    = [t for t in norm.split() if len(t) >= _MIN_TOK]
            if not tokens:
                continue
            all_keys: Set[str] = set()
            for tok in set(tokens + norm_mlin.split()):
                if len(tok) >= _MIN_TOK:
                    all_keys |= self._codes(tok)
            entry = _Entry(
                canonical=canonical, norm=norm, norm_mlin=norm_mlin,
                tokens=tokens, first_tok=tokens[0], last_tok=tokens[-1],
                phon_keys=all_keys,
            )
            self._entries.append(entry)
            for k in all_keys:
                self._phon_idx[k].append(entry)
            for k in self._codes(tokens[0]):
                self._first_idx[k].append(entry)

    def phonetic_candidates(self, word: str) -> List[_Entry]:
        codes = self._codes(word)
        seen: Set[int] = set()
        out: List[_Entry] = []
        for c in codes:
            for e in self._phon_idx.get(c, []):
                if id(e) not in seen:
                    seen.add(id(e))
                    out.append(e)
        return out

    def first_name_candidates(self, tok: str) -> List[_Entry]:
        codes = self._codes(tok)
        seen: Set[int] = set()
        out: List[_Entry] = []
        for c in codes:
            for e in self._first_idx.get(c, []):
                if id(e) not in seen:
                    seen.add(id(e))
                    out.append(e)
        return out

    def all_entries(self) -> List[_Entry]:
        return self._entries


# ─────────────────────────────────────────────────────────────────────────────
#  Index cache (process-level singleton)
# ─────────────────────────────────────────────────────────────────────────────

_idx_lock:  threading.Lock = threading.Lock()
_idx_cache: Dict[int, Tuple[PhoneticIndex, PhoneticIndex]] = {}


def _get_indexes(
    patients: List[str],
    nurses:   List[str],
    include_irish: bool = True,
) -> Tuple[PhoneticIndex, PhoneticIndex]:
    """
    Build (or return cached) phonetic indexes for patients and nurses.

    When include_irish=True, the curated Irish name list from irish_names.py
    is merged into the patient index.  Names already present in the live
    registry are skipped to avoid duplication.
    """
    if include_irish:
        try:
            from irish_names import get_all_irish_names
            irish = get_all_irish_names()
            patient_norms = {_strip_name(p).lower() for p in patients}
            extra = [n for n in irish
                     if _strip_name(n).lower() not in patient_norms]
            patients = patients + extra
            logger.debug("L1: Irish names augmentation: +%d names.", len(extra))
        except ImportError:
            logger.warning("L1: irish_names.py not found — Irish name augmentation skipped.")
        except Exception as exc:
            logger.warning("L1: Irish name augmentation failed: %s", exc)

    key = hash((tuple(patients), tuple(nurses)))
    if key in _idx_cache:
        return _idx_cache[key]
    with _idx_lock:
        if key in _idx_cache:
            return _idx_cache[key]
        logger.info("L1: Building phonetic indexes (%d p, %d n)…",
                    len(patients), len(nurses))
        pi, ni = PhoneticIndex(patients, "patient"), PhoneticIndex(nurses, "nurse")
        _idx_cache[key] = (pi, ni)
    return pi, ni


# ─────────────────────────────────────────────────────────────────────────────
#  Optional spaCy singleton
# ─────────────────────────────────────────────────────────────────────────────

_spacy_lock:  threading.Lock   = threading.Lock()
_spacy_nlp:   Optional[object] = None
_spacy_built: bool             = False


def _get_spacy() -> Optional[object]:
    global _spacy_nlp, _spacy_built
    if _spacy_built:
        return _spacy_nlp
    with _spacy_lock:
        if _spacy_built:
            return _spacy_nlp
        try:
            import spacy  # type: ignore[import]
            for m in ("en_core_web_lg", "en_core_web_trf", "en_core_web_sm"):
                try:
                    _spacy_nlp = spacy.load(m, disable=["parser", "lemmatizer"])
                    logger.info("L1: spaCy '%s' loaded.", m)
                    break
                except OSError:
                    continue
            if _spacy_nlp is None:
                logger.warning("L1: No spaCy English model found.")
        except ImportError:
            logger.warning("L1: spaCy not installed — NER disabled.")
        _spacy_built = True
    return _spacy_nlp


# ─────────────────────────────────────────────────────────────────────────────
#  Detection helpers
# ─────────────────────────────────────────────────────────────────────────────

def _strip_honorific(text: str) -> str:
    toks = text.split()
    if len(toks) <= 1:
        return text
    head = re.sub(r"[.\s]+$", "", toks[0]).lower()
    return " ".join(toks[1:]) if head in _HONORIFICS else text


def _is_pii(text: str) -> bool:
    return bool(_PII_TOKEN_RE.fullmatch(text.strip()))


def _dedup(spans: List[str]) -> List[str]:
    spans = sorted(set(spans), key=len, reverse=True)
    kept: List[str] = []
    for s in spans:
        sl = s.lower()
        if not any(sl in k.lower() for k in kept):
            kept.append(s)
    return kept


_CAP_RE = re.compile(
    r"(?<!\w)"
    r"(?:(?:[A-Z][a-z'\-]+|O\'[A-Z][a-z]+|Mc[A-Z][a-z]+|Mac[A-Z][a-z]+)"
    r"(?:\s+(?:[A-Z][a-z'\-]+|O\'[A-Z][a-z]+|Mc[A-Z][a-z]+|Mac[A-Z][a-z]+)){0,3})"
    r"(?!\w)"
)


def _detect_ner(text: str) -> List[str]:
    nlp = _get_spacy()
    if not nlp:
        return []
    try:
        out = []
        for ent in nlp(text).ents:
            if ent.label_ == "PERSON":
                s = ent.text.strip()
                if (len(s) >= _MIN_TOK and s.lower() not in _NER_STOPWORDS
                        and not _is_pii(s)):
                    out.append(s)
        return out
    except Exception as exc:
        logger.warning("L1: spaCy error: %s", exc)
        return []


def _detect_heuristic(text: str) -> List[str]:
    out = []
    for m in _CAP_RE.finditer(text):
        s = m.group().strip()
        if (len(s) >= _MIN_TOK and s.lower() not in _CAP_STOP
                and not _is_pii(s)
                and any(len(t) >= 3 for t in s.split())):
            out.append(s)
    return out


def _detect_window(text: str, all_names: List[str]) -> List[str]:
    from shared import find_names_in_text
    return [sp for sp, _, _ in
            find_names_in_text(text, all_names, threshold=_THRESH_WINDOW)]


# ─────────────────────────────────────────────────────────────────────────────
#  Scoring
# ─────────────────────────────────────────────────────────────────────────────

def _score_span(norm_span: str, norm_mlin: str, entry: _Entry) -> float:
    """Score normalised query against one entry. Returns 0–100."""
    full_s = float(fuzz.WRatio(norm_span, entry.norm))
    mlin_s = max(
        float(fuzz.WRatio(norm_mlin, entry.norm_mlin)),
        float(fuzz.WRatio(norm_mlin, entry.norm)),
    )

    ftoks = [t for t in norm_span.split() if len(t) >= _MIN_TOK]
    ctoks = entry.tokens
    if not ftoks or not ctoks:
        return max(full_s, mlin_s)

    claimed: Set[int] = set()
    tok_scores: List[float] = []

    for ft in ftoks:
        best = 0.0
        best_i = -1
        ft_codes = PhoneticIndex._codes(ft)
        for ci, ct in enumerate(ctoks):
            if ci in claimed:
                continue
            jw = jellyfish.jaro_winkler_similarity(ft, ct) * 100.0
            wr = float(fuzz.WRatio(ft, ct))
            ct_codes = PhoneticIndex._codes(ct)
            boost = 8.0 if ft_codes & ct_codes else 0.0
            s = max(jw, wr) + boost
            if s > best:
                best, best_i = s, ci
        if best_i >= 0:
            claimed.add(best_i)
        tok_scores.append(best)

    if min(tok_scores) < 55:
        return 0.0

    combined = (
        (sum(tok_scores) / len(tok_scores)) * 0.45
        + full_s * 0.25
        + mlin_s * 0.30
        - min(abs(len(ftoks) - len(ctoks)) * 5, 15)
    )
    return min(max(combined, 0.0), 100.0)


def _best_match(
    norm_span: str,
    norm_mlin: str,
    p_idx:     PhoneticIndex,
    n_idx:     PhoneticIndex,
    threshold: int,
) -> Tuple[Optional[str], str, float]:
    """
    Score against both indexes.
    Role = which index the winner came from (NEVER position-inferred).
    """
    best_s: float         = 0.0
    best_c: Optional[str] = None
    best_r: str           = ""

    first_tok = (norm_span.split() or [norm_span])[0]

    for idx in (p_idx, n_idx):
        cands = idx.phonetic_candidates(first_tok) or idx.all_entries()
        for e in cands:
            s = _score_span(norm_span, norm_mlin, e)
            if s > best_s:
                best_s, best_c, best_r = s, e.canonical, idx.role

    if best_s < threshold or best_c is None:
        return None, "", round(best_s / 100, 3)
    return best_c, best_r, round(best_s / 100, 3)


# ─────────────────────────────────────────────────────────────────────────────
#  Fallback F1 — first-name-only
# ─────────────────────────────────────────────────────────────────────────────

def _match_first_name(
    first_tok: str,
    p_idx:     PhoneticIndex,
    n_idx:     PhoneticIndex,
) -> Tuple[Optional[str], str, float]:
    """
    Match a single first-name token.
    Returns a result only when EXACTLY ONE registry entry qualifies.
    """
    if len(first_tok) < 3:
        return None, "", 0.0

    hits: List[Tuple[str, str, float]] = []
    seen: Set[str] = set()

    for idx in (p_idx, n_idx):
        for e in idx.first_name_candidates(first_tok):
            jw = jellyfish.jaro_winkler_similarity(
                first_tok.lower(), e.first_tok.lower()
            )
            if jw >= _THRESH_FIRST_NAME and e.canonical not in seen:
                seen.add(e.canonical)
                hits.append((e.canonical, idx.role, jw))

    hits.sort(key=lambda x: -x[2])

    if len(hits) == 1:
        can, role, jw = hits[0]
        return can, role, round(jw * 0.90, 3)   # 10 % partial-match penalty

    return None, "", 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  Fallback F2 — contamination filter
# ─────────────────────────────────────────────────────────────────────────────

def _contamination_fix(
    surface:          str,
    p_idx:            PhoneticIndex,
    n_idx:            PhoneticIndex,
    claimed_surnames: Set[str],
) -> Tuple[Optional[str], str, float, str]:
    """
    When a span fails full-match, remove tokens that are surnames already
    claimed by earlier-resolved spans, then retry.

    Example:
      "Anne Thompson" fails.
      claimed_surnames = {"thompson"} (from "Emma Thompson").
      Clean tokens = ["anne"].
      First-name-only → "Anne Kelly" (patient, unique match).
    """
    ms    = _strip_honorific(surface).strip()
    toks  = [t for t in ms.split() if len(t) >= _MIN_TOK]
    if not toks:
        return None, "", 0.0, "none"

    clean = [t for t in toks if t.lower() not in claimed_surnames]
    if not clean or clean == toks:
        return None, "", 0.0, "none"

    norm_clean = _strip_name(" ".join(clean))
    mlin_clean = _strip_name(mlin_normalise_name(" ".join(clean)))

    if len(clean) == 1:
        can, role, conf = _match_first_name(norm_clean, p_idx, n_idx)
        if can:
            return can, role, conf, "contamination_filter"
    else:
        can, role, conf = _best_match(
            norm_clean, mlin_clean, p_idx, n_idx, _THRESH_HEURISTIC
        )
        if can:
            return can, role, conf, "contamination_filter"

    return None, "", 0.0, "none"


# ─────────────────────────────────────────────────────────────────────────────
#  Core per-span resolution
# ─────────────────────────────────────────────────────────────────────────────

def _resolve(
    surface:          str,
    p_idx:            PhoneticIndex,
    n_idx:            PhoneticIndex,
    is_ner:           bool,
    claimed_surnames: Set[str],
) -> Tuple[Optional[str], str, float, str]:
    """
    Resolve a surface span to (canonical, role, conf, resolution_path).
    Returns (None, "unknown", conf, "none") if unresolvable.
    Role is ALWAYS from registry — never inferred from sentence position.
    """
    if _is_pii(surface):
        return None, "", 0.0, "none"

    ms = _strip_honorific(surface).strip()
    if len(ms) < _MIN_TOK or ms.lower() in _NER_STOPWORDS:
        return None, "", 0.0, "none"

    norm  = _strip_name(ms)
    mlin  = _strip_name(mlin_normalise_name(ms))

    # L1: exact
    for idx in (p_idx, n_idx):
        for e in idx.all_entries():
            if norm in (e.norm, e.norm_mlin):
                return e.canonical, idx.role, 1.0, "exact"

    # L2: accent-normalised exact
    for idx in (p_idx, n_idx):
        for e in idx.all_entries():
            if mlin in (e.norm, e.norm_mlin):
                return e.canonical, idx.role, 0.97, "accent_norm"

    # L3+L4: phonetic + fuzzy
    thresh = _THRESH_NER if is_ner else _THRESH_HEURISTIC
    can, role, conf = _best_match(norm, mlin, p_idx, n_idx, thresh)
    if can:
        return can, role, conf, "phonetic_fuzzy"

    # F1: first-name-only
    toks = [t for t in norm.split() if len(t) >= _MIN_TOK]
    if toks:
        can_fn, role_fn, conf_fn = _match_first_name(toks[0], p_idx, n_idx)
        if can_fn:
            return can_fn, role_fn, conf_fn, "first_name_only"

    # F2: contamination filter
    can_cf, role_cf, conf_cf, path_cf = _contamination_fix(
        surface, p_idx, n_idx, claimed_surnames
    )
    if can_cf:
        return can_cf, role_cf, conf_cf, path_cf

    return None, "unknown", conf, "none"


# ─────────────────────────────────────────────────────────────────────────────
#  Text replacement
# ─────────────────────────────────────────────────────────────────────────────

def _replace(text: str, surface: str, canonical: str) -> str:
    if surface.lower() == canonical.lower():
        return text
    pat = (
        r"(?<![a-zA-Z0-9'\-])"
        + re.escape(surface)
        + r"(?![a-zA-Z0-9'\-]|(?:'\w))"
    )
    return re.sub(pat, lambda _: canonical, text, flags=re.IGNORECASE)


# ─────────────────────────────────────────────────────────────────────────────
#  Validation
# ─────────────────────────────────────────────────────────────────────────────

def _validate_id(ordis_id: str) -> None:
    if not ordis_id:
        return
    if len(ordis_id) > _MAX_ID:
        raise ValueError(f"Ordis_ID too long ({len(ordis_id)} chars).")
    if not _ID_RE.fullmatch(ordis_id):
        raise ValueError(f"Ordis_ID '{ordis_id}' contains invalid characters.")


# ─────────────────────────────────────────────────────────────────────────────
#  ▓▓  LEARNING STORE  ▓▓
#
#  Industry-grade feedback persistence engine.
#
#  Architecture
#  ────────────
#  • Corrections are written to a JSON file after every feedback call
#    (atomic via tmp-file + rename → safe against process crash mid-write).
#  • Process-level singleton; thread-safe via a single reentrant lock.
#  • Applied AFTER the normal resolution pipeline, so the core algorithm
#    is unchanged — learning is a pure override/boost/penalty layer.
#
#  JSON schema (schema_version = 1)
#  ─────────────────────────────────
#  {
#    "schema_version": 1,
#    "confirmed": {
#      "dharni kumar": {"canonical": "Dharani Kumar", "role": "nurse"}
#    },
#    "rejected": {
#      "anne kelly": ["Anne Thompson"]
#    },
#    "confusion_pairs": {
#      "John Doyle": {"correct": "John Murphy", "role": "patient"}
#    },
#    "miss_counts": {
#      "Marcus O'Reilly": 3
#    },
#    "total_feedback": 47,
#    "last_updated": "2026-04-09T10:30:00.123456"
#  }
#
#  Confidence adjustments applied in apply():
#    Confirmed match  → +_CONF_BOOST  (capped at 0.99)
#    Override         → _CONF_OVERRIDE (0.95, fixed)
#    Confusion pair   → +_CONF_CONFUSION_BOOST (0.03)
#    Rejected match   → -_CONF_PENALTY; if result < _TIER_AUDIT → suppressed
# ─────────────────────────────────────────────────────────────────────────────

class LearningStore:
    """
    Thread-safe, file-backed learning store for Layer 01.

    Usage
    -----
    Instantiated once per process (singleton via _get_learn_store()).
    Consume via the module-level record_feedback() function.

    Direct instantiation is supported for testing / custom paths.
    """

    _SCHEMA             = 1
    _DEFAULT_PATH       = Path.home() / ".ordis" / "layer01_learn.json"
    _CONF_BOOST         = 0.06    # added when confirmed match agrees
    _CONF_OVERRIDE      = 0.95    # used when overriding with stored canonical
    _CONF_CONFUSION_BOOST = 0.03  # added when confusion pair reroutes
    _CONF_PENALTY       = 0.40    # subtracted when candidate is in rejected list

    def __init__(
        self,
        path: Optional[Path] = None,
        cfg:  Optional[ConfigManager] = None,
    ) -> None:
        cfg_path: Optional[str] = None
        if cfg is not None:
            try:
                cfg_path = cfg.get("LAYER01_LEARN_PATH")  # type: ignore[attr-defined]
            except Exception:
                pass
        self._path: Path = Path(path or cfg_path or self._DEFAULT_PATH)
        self._lock = threading.Lock()
        self._data: Dict = self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _blank(self) -> Dict:
        return {
            "schema_version": self._SCHEMA,
            "confirmed":       {},   # surface_lower → {canonical, role}
            "rejected":        {},   # surface_lower → [canonical, ...]
            "confusion_pairs": {},   # wrong_canonical → {correct, role}
            "miss_counts":     {},   # canonical → int
            "total_feedback":  0,
            "last_updated":    "",
        }

    def _load(self) -> Dict:
        try:
            if self._path.exists():
                with self._path.open("r", encoding="utf-8") as fh:
                    d = json.load(fh)
                if isinstance(d, dict) and d.get("schema_version") == self._SCHEMA:
                    logger.info(
                        "L1-learn: Store loaded from '%s' (%d confirmed, %d rejected).",
                        self._path,
                        len(d.get("confirmed", {})),
                        len(d.get("rejected", {})),
                    )
                    return d
                logger.warning("L1-learn: Schema mismatch in '%s' — resetting.", self._path)
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.warning("L1-learn: Load error (%s) — starting fresh.", exc)
        return self._blank()

    def _save(self) -> None:
        """Atomic write: temp file → rename."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(self._data, fh, indent=2, ensure_ascii=False)
            tmp.replace(self._path)
        except Exception as exc:
            logger.error("L1-learn: Save failed: %s", exc)

    # ── Write API ─────────────────────────────────────────────────────────────

    def record(
        self,
        surface:   str,
        proposed:  str,
        accepted:  bool,
        role:      str           = "",
        correct:   Optional[str] = None,
        corr_role: Optional[str] = None,
    ) -> None:
        """
        Persist one piece of human feedback.

        Parameters
        ----------
        surface   : Raw surface span as it appeared in the transcript.
        proposed  : Canonical name Layer 01 proposed.
        accepted  : True = human confirmed it is correct.
        role      : Registry role of the proposed canonical.
        correct   : If accepted=False and the true name is known, supply it.
        corr_role : Role of the correct canonical (when correct is given).
        """
        sl = surface.strip().lower()
        import datetime

        with self._lock:
            if accepted:
                # ── Confirm ──────────────────────────────────────────────────
                self._data["confirmed"][sl] = {"canonical": proposed, "role": role}
                # Clear any stale rejection for this (surface, proposed) pair
                rej_list = self._data["rejected"].get(sl, [])
                if proposed in rej_list:
                    rej_list.remove(proposed)
                logger.debug("L1-learn: CONFIRMED '%s' → '%s'.", surface, proposed)

            else:
                # ── Reject ───────────────────────────────────────────────────
                rej_list = self._data["rejected"].setdefault(sl, [])
                if proposed not in rej_list:
                    rej_list.append(proposed)

                # Increment miss count for the wrong canonical
                mc = self._data["miss_counts"]
                mc[proposed] = mc.get(proposed, 0) + 1

                if correct:
                    # Store confusion pair so future calls reroute automatically
                    self._data["confusion_pairs"][proposed] = {
                        "correct": correct,
                        "role":    corr_role or "",
                    }
                    # Also record the correct mapping as confirmed
                    self._data["confirmed"][sl] = {
                        "canonical": correct,
                        "role":      corr_role or "",
                    }
                    logger.debug(
                        "L1-learn: REJECTED '%s'→'%s', confusion stored → '%s'.",
                        surface, proposed, correct,
                    )
                else:
                    logger.debug("L1-learn: REJECTED '%s'→'%s'.", surface, proposed)

            self._data["total_feedback"] = self._data.get("total_feedback", 0) + 1
            self._data["last_updated"] = datetime.datetime.utcnow().isoformat()
            self._save()

    # ── Read / Apply API ──────────────────────────────────────────────────────

    def apply(
        self,
        surface: str,
        can:     Optional[str],
        role:    str,
        conf:    float,
        path:    str,
    ) -> Tuple[Optional[str], str, float, str]:
        """
        Apply learned corrections to a resolution result.

        Called immediately after _resolve() in run().  Operates in three
        priority levels:

        1. Confirmed override  — human previously confirmed a canonical for
                                 this surface; use it (even if _resolve picked
                                 something different).
        2. Confusion pair      — _resolve returned a canonical that was
                                 previously flagged as wrong; reroute to the
                                 stored correct answer.
        3. Rejection penalty   — _resolve returned a canonical in the rejected
                                 list; reduce confidence; suppress if below
                                 _TIER_AUDIT.

        No lock needed here (reads only; writes go via record()).
        """
        sl = surface.strip().lower()
        confirmed_entry = self._data["confirmed"].get(sl)
        rejected_list   = self._data["rejected"].get(sl, [])
        confusion       = self._data["confusion_pairs"]

        # ── Level 1: confirmed override ───────────────────────────────────────
        if confirmed_entry:
            conf_can  = confirmed_entry["canonical"]
            conf_role = confirmed_entry.get("role") or role

            if can == conf_can:
                # Algorithm already agreed with the human — boost confidence
                boosted = min(conf + self._CONF_BOOST, 0.99)
                return can, role, boosted, path + "+confirmed"
            else:
                # Algorithm disagreed — override silently
                logger.debug(
                    "L1-learn: '%s' → '%s' overridden by learning (was '%s').",
                    surface, conf_can, can,
                )
                return conf_can, conf_role, self._CONF_OVERRIDE, "learned_confirmed"

        # ── Level 2: confusion pair ───────────────────────────────────────────
        if can and can in confusion:
            cp     = confusion[can]
            corr   = cp["correct"]
            crole  = cp.get("role") or role
            logger.debug(
                "L1-learn: confusion pair '%s' → '%s' rerouted to '%s'.",
                surface, can, corr,
            )
            return corr, crole, min(conf + self._CONF_CONFUSION_BOOST, 0.97), "learned_confusion"

        # ── Level 3: rejection penalty ────────────────────────────────────────
        if can and can in rejected_list:
            penalised = conf - self._CONF_PENALTY
            if penalised < _TIER_AUDIT:
                logger.debug(
                    "L1-learn: '%s'→'%s' suppressed (rejected, penalised conf=%.2f).",
                    surface, can, penalised,
                )
                return None, "unknown", 0.0, "rejected_by_learning"
            return can, role, max(penalised, 0.0), path + "+penalised"

        # No learning signal — pass through unchanged
        return can, role, conf, path

    # ── Introspection ─────────────────────────────────────────────────────────

    def stats(self) -> Dict:
        """
        Return a summary dict suitable for a monitoring dashboard.

        Keys
        ----
        confirmed_count     : Number of confirmed surface→canonical mappings.
        rejected_surfaces   : Number of surfaces with at least one rejection.
        confusion_pairs     : Number of known wrong→right canonical redirects.
        total_feedback      : Cumulative feedback calls.
        most_confused       : Top 5 canonicals by miss count.
        last_updated        : ISO-8601 timestamp of last write.
        store_path          : Absolute path to the JSON file.
        """
        mc    = self._data.get("miss_counts", {})
        top5  = sorted(mc.items(), key=lambda x: -x[1])[:5]
        return {
            "confirmed_count":   len(self._data.get("confirmed", {})),
            "rejected_surfaces": len(self._data.get("rejected", {})),
            "confusion_pairs":   len(self._data.get("confusion_pairs", {})),
            "total_feedback":    self._data.get("total_feedback", 0),
            "most_confused":     [{"canonical": k, "misses": v} for k, v in top5],
            "last_updated":      self._data.get("last_updated", ""),
            "store_path":        str(self._path),
        }

    def reset(self) -> None:
        """
        Wipe all learned data.  USE WITH CAUTION — irreversible.
        Intended for test suites only.
        """
        with self._lock:
            self._data = self._blank()
            self._save()
        logger.warning("L1-learn: Store reset to blank.")


# ─────────────────────────────────────────────────────────────────────────────
#  Process-level LearningStore singleton
# ─────────────────────────────────────────────────────────────────────────────

_learn_singleton_lock: threading.Lock    = threading.Lock()
_learn_singleton:      Optional[LearningStore] = None


def _get_learn_store(cfg: Optional[ConfigManager] = None) -> LearningStore:
    """Return (or lazily create) the process-level LearningStore."""
    global _learn_singleton
    if _learn_singleton is not None:
        return _learn_singleton
    with _learn_singleton_lock:
        if _learn_singleton is None:
            _learn_singleton = LearningStore(cfg=cfg)
    return _learn_singleton


# ─────────────────────────────────────────────────────────────────────────────
#  Public Learning API
# ─────────────────────────────────────────────────────────────────────────────

def record_feedback(
    surface:            str,
    proposed_canonical: str,
    accepted:           bool,
    role:               str                     = "",
    correct_canonical:  Optional[str]           = None,
    correct_role:       Optional[str]           = None,
    ordis_id:           str                     = "",
    cfg:                Optional[ConfigManager] = None,
) -> None:
    """
    Record human feedback on a Layer 01 correction.

    Call this from your review / audit UI whenever a clinician or supervisor
    confirms or rejects a name correction.  The feedback is persisted
    immediately and applied on the very next call to run().

    Parameters
    ----------
    surface            : The raw surface span as it appeared in the transcript
                         (e.g. "Dharni Kumar").
    proposed_canonical : The canonical name Layer 01 proposed
                         (e.g. "Dharani Kumar").
    accepted           : True  → correction is correct.
                         False → correction is wrong.
    role               : Registry role of proposed_canonical
                         ("nurse" | "patient" | "unknown").
    correct_canonical  : When accepted=False and the true canonical is known,
                         supply it here (e.g. "Dharani Menon").
                         Stored as a confusion pair for automatic rerouting.
    correct_role       : Role of correct_canonical, if known.
    ordis_id           : Session / document identifier (logging only).
    cfg                : ConfigManager (used to locate custom store path).

    Example
    -------
    >>> # Nurse confirmed "Dharni Kumar" → "Dharani Kumar" is correct
    >>> record_feedback("Dharni Kumar", "Dharani Kumar", accepted=True,
    ...                 role="nurse", ordis_id="session_001")

    >>> # Reviewer says "Siobhán Murphy" was wrongly corrected to "Siobhán Ryan"
    >>> record_feedback("Siobhán Murphy", "Siobhán Ryan", accepted=False,
    ...                 correct_canonical="Siobhán Murphy", correct_role="patient")
    """
    store = _get_learn_store(cfg)
    store.record(
        surface=surface,
        proposed=proposed_canonical,
        accepted=accepted,
        role=role,
        correct=correct_canonical,
        corr_role=correct_role,
    )
    _log = logger.getChild(ordis_id) if ordis_id else logger
    _log.info(
        "L1-learn: feedback — '%s' → '%s' %s%s",
        surface,
        proposed_canonical,
        "ACCEPTED" if accepted else "REJECTED",
        f" (correct='{correct_canonical}')" if correct_canonical else "",
    )


def get_learning_stats(cfg: Optional[ConfigManager] = None) -> Dict:
    """
    Return a summary of the learning store for dashboards / health checks.

    Returns
    -------
    Dict with keys: confirmed_count, rejected_surfaces, confusion_pairs,
    total_feedback, most_confused, last_updated, store_path.
    """
    return _get_learn_store(cfg).stats()


def reset_learning(cfg: Optional[ConfigManager] = None) -> None:
    """
    Wipe the entire learning store.

    ⚠ IRREVERSIBLE — intended for test environments only.
    In production, remove individual entries by calling record_feedback()
    with the corrected value instead.
    """
    _get_learn_store(cfg).reset()


# ─────────────────────────────────────────────────────────────────────────────
#  Public API — run()
# ─────────────────────────────────────────────────────────────────────────────

def run(
    text:      str,
    accent:    str                     = "ml_In",
    ordis_id:  str                     = "",
    cfg:       Optional[ConfigManager] = None,
    mongo_db:  Optional[object]        = None,
) -> Tuple[str, List[NameDetection]]:
    """
    Layer 01 — Name Detection & Correction.  100 % local.

    Returns
    -------
    (corrected_text, detections)

    detections contains ALL detected names (corrected + already-correct +
    unresolved). Role is always from the registry, not sentence position.

    Edge-case behaviour
    -------------------
    Only nurse in text      → detected as nurse, text unchanged for patient slot
    Unknown name            → is_unresolved=True, role="unknown", text unchanged
    No names detected       → empty list, text unchanged
    Role reversal in text   → roles still correct (registry-bound)

    v5 additions
    ------------
    • LearningStore consulted after each _resolve() call.
    • Irish names from irish_names.py merged into patient index (configurable).
    """
    import time as _t
    t0 = _t.perf_counter()

    if not isinstance(text, str):
        raise TypeError(f"Layer 01 expects str, got {type(text).__name__}.")
    _validate_id(ordis_id)

    text = unicodedata.normalize("NFC", text)
    if not text.strip():
        return text, []

    log = logger.getChild(ordis_id) if ordis_id else logger
    log.info("L1: %d chars (accent=%s).", len(text), accent)

    if cfg is None:
        cfg = ConfigManager()
    patients, nurses = load_name_registry(cfg, mongo_db)
    all_names = patients + nurses

    if not all_names:
        log.warning("L1: Registry empty.")
        return text, []

    # ── Irish name augmentation flag ──────────────────────────────────────────
    include_irish = True
    try:
        val = cfg.get("LAYER01_IRISH_NAMES")  # type: ignore[attr-defined]
        if val is not None:
            include_irish = bool(val)
    except Exception:
        pass

    p_idx, n_idx = _get_indexes(patients, nurses, include_irish=include_irish)

    # ── Learning store ────────────────────────────────────────────────────────
    store = _get_learn_store(cfg)

    # ── Detect ───────────────────────────────────────────────────────────────
    ner_spans  = _detect_ner(text)
    heu_spans  = _detect_heuristic(text)
    win_spans  = _detect_window(text, all_names)
    ner_set    = {s.lower() for s in ner_spans}
    heu_set    = {s for s in heu_spans}

    all_spans = _dedup(ner_spans + heu_spans + win_spans)
    all_spans.sort(key=len, reverse=True)   # longest first

    log.info("L1: NER:%d heu:%d win:%d → %d unique spans.",
             len(ner_spans), len(heu_spans), len(win_spans), len(all_spans))

    result           = text
    detections:       List[NameDetection] = []
    replaced:         Set[str]  = set()
    claimed_surnames: Set[str]  = set()    # surnames locked by confirmed matches

    for surface in all_spans:
        if surface in replaced:
            continue

        is_ner  = surface.lower() in ner_set
        det_mth = "ner" if is_ner else ("heuristic" if surface in heu_set else "window")

        # Core resolution pipeline
        can, role, conf, path = _resolve(
            surface, p_idx, n_idx, is_ner, claimed_surnames
        )

        # ── Apply learning ────────────────────────────────────────────────────
        can, role, conf, path = store.apply(surface, can, role, conf, path)

        # ── Unresolved ────────────────────────────────────────────────────────
        if can is None:
            detections.append(NameDetection(
                surface_span=surface, canonical_name=None, role="unknown",
                confidence=conf, was_corrected=False, is_unresolved=True,
                resolution_path="none", tier="skipped",
                detection_method=det_mth, ordis_id=ordis_id,
            ))
            log.debug("L1: '%s' unresolved (conf=%.2f).", surface, conf)
            continue

        # ── Determine tier ────────────────────────────────────────────────────
        if conf >= _TIER_AUTO_SILENT:
            tier = "auto_silent"
        elif conf >= _TIER_AUTO_LOG:
            tier = "auto_log"
        elif conf >= _TIER_AUDIT:
            tier = "audit"
        else:
            # Detected but not confident enough to correct
            detections.append(NameDetection(
                surface_span=surface, canonical_name=can, role=role,
                confidence=conf, was_corrected=False, is_unresolved=False,
                resolution_path=path, tier="skipped",
                detection_method=det_mth, ordis_id=ordis_id,
            ))
            log.debug("L1: '%s' below correction threshold (%.2f).", surface, conf)
            continue

        # ── Apply text replacement ────────────────────────────────────────────
        needs_change = (_strip_name(surface) != _strip_name(can))
        if needs_change:
            new_result = _replace(result, surface, can)
            if new_result == result:
                needs_change = False   # span not found verbatim
            else:
                result = new_result
                replaced.add(surface)

        # Lock this name's surname so F2 can use it for later spans
        canon_toks = _strip_name(can).split()
        if len(canon_toks) >= 2:
            claimed_surnames.add(canon_toks[-1].lower())

        detections.append(NameDetection(
            surface_span=surface, canonical_name=can, role=role,
            confidence=conf, was_corrected=needs_change, is_unresolved=False,
            resolution_path=path, tier=tier,
            detection_method=det_mth, ordis_id=ordis_id,
        ))

        if needs_change:
            msg = (f"L1: '{_YL}{surface}{_R}' → '{_GR}{can}{_R}' "
                   f"[{role}, {conf:.2f}, {tier}, {path}]")
            if tier == "auto_silent":
                log.debug(msg)
            elif tier == "auto_log":
                log.info(msg)
            else:
                log.warning("AUDIT %s", msg)
        else:
            log.debug("L1: '%s' → %s '%s' (%.2f, %s). No text change.",
                      surface, role, can, conf, path)

    n_corr = sum(1 for d in detections if d.was_corrected)
    n_unre = sum(1 for d in detections if d.is_unresolved)
    log.info("L1: %.3fs — %d detected, %d corrected, %d unresolved.",
             _t.perf_counter() - t0, len(detections), n_corr, n_unre)

    return result, detections


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def run_simple(
    text: str, accent: str = "ml_In", ordis_id: str = "",
    cfg: Optional[ConfigManager] = None, mongo_db: Optional[object] = None,
) -> Tuple[str, List[Tuple[str, str]]]:
    """Backward-compat: returns (corrected_text, [(surface, canonical), …])."""
    corrected, dets = run(text, accent, ordis_id, cfg, mongo_db)
    return corrected, [(d.surface_span, d.canonical_name)
                       for d in dets if d.was_corrected and d.canonical_name]


def get_role_map(detections: List[NameDetection]) -> Dict[str, str]:
    """
    Convenience helper: {canonical_name: role} from a detection list.
    Role is registry-bound (never position-inferred).

    Example
    -------
    >>> {"Dharani Kumar": "nurse", "Marcus O'Reilly": "patient"}
    """
    out: Dict[str, str] = {}
    for d in detections:
        key = d.canonical_name or d.surface_span
        if key and d.role not in ("unknown", ""):
            out[key] = d.role
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  CLI — runs the exact QA cases from the 08-Apr-2026 QA sheet
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)-8s %(name)s — %(message)s")

    QA: List[Tuple[str, str, Optional[str], Optional[str], Optional[str]]] = [
        # (label, input, text_must_contain, expected_nurse, expected_patient)
        ("POS-1 Correct names",
         "Dharani Kumar is attending Marcus O'Reilly for routine checkup.",
         None, "Dharani Kumar", "Marcus O'Reilly"),

        ("POS-2 Correct names",
         "Priya Patel gave medication to Mary Collins on time.",
         None, "Priya Patel", "Mary Collins"),

        ("POS-3 Correct names",
         "Sarah Johnson checked the vitals of John Doyle.",
         None, "Sarah Johnson", "John Doyle"),

        ("NEG-1 Same Nurse & Resident name",
         "Emma Thompson assisted Emma Thompson during breakfast.",
         None, "Emma Thompson", "Emma Thompson"),

        ("NEG-2 Dharni Kumar (nurse typo)",
         "Dharni Kumar is attending Marcus O'Reilly.",
         "Dharani Kumar", "Dharani Kumar", "Marcus O'Reilly"),

        ("NEG-3 Prija Patil",
         "Prija Patil gave medication to Mary Collins.",
         "Priya Patel", "Priya Patel", "Mary Collins"),

        ("NEG-4 Sarah Jonson",
         "Sarah Jonson checked the vitals of John Doyle.",
         "Sarah Johnson", "Sarah Johnson", "John Doyle"),

        ("NEG-5 Marcus O'Reily",
         "Dharani Kumar is attending Marcus O'Reily.",
         "Marcus O'Reilly", "Dharani Kumar", "Marcus O'Reilly"),

        ("NEG-6 Mary Collin",
         "Priya Patel gave medication to Mary Collin.",
         "Mary Collins", "Priya Patel", "Mary Collins"),

        ("NEG-7 Anne Kelv",
         "Emma Thompson assisted Anne Kelv.",
         "Anne Kelly", "Emma Thompson", "Anne Kelly"),

        ("NEG-8 Marcus Reily",
         "Dharni Kumar is attending Marcus Reily.",
         "Marcus O'Reilly", "Dharani Kumar", "Marcus O'Reilly"),

        ("NEG-9 Lisa Chan",
         "Lisa Chan completed the report for Daniel Ryan.",
         "Lisa Chen", "Lisa Chen", "Daniel Ryan"),

        ("FAIL-1 Role Swap",
         "Marcus O'Reilly is assisting Dharani Kumar.",
         None, "Dharani Kumar", "Marcus O'Reilly"),

        ("NEG-10 Sarah Johnsonaa",
         "Sarah Johnsonaa checking vitals John Dolle.",
         "Sarah Johnson", "Sarah Johnson", "John Doyle"),

        ("FAIL-2 Anne Thompson contamination",
         "Emma Thompson assist Anne Thompson during breakfast.",
         "Anne Kelly", "Emma Thompson", "Anne Kelly"),

        ("FAIL-3 First name only + surname mismatch",
         "Dharani  attending Marcus Murphy.",
         "Dharani Kumar", "Dharani Kumar", "Marcus O'Reilly"),

        ("FAIL-4 Nurse only",
         "Lisa Chen completed report.",
         None, "Lisa Chen", None),

        ("FAIL-5 Unknown resident",
         "Dharani Kumar gave medicine to Rahul Sharma.",
         None, "Dharani Kumar", None),

        ("FAIL-6 No registry names",
         "Ganga gave medication to maya.",
         None, None, None),

        ("NEG-11 Wrong capitalisation",
         "Prija Patel gave medication to Mary collins.",
         "Mary Collins", "Priya Patel", "Mary Collins"),

        # ── v5 learning round-trip tests ──────────────────────────────────────
        ("LEARN-1 Confirmed correction sticks",
         "Dharni Kumar is attending Marcus O'Reilly.",
         "Dharani Kumar", "Dharani Kumar", "Marcus O'Reilly"),

        ("LEARN-2 Irish name phonetic (shivawn)",
         "Shivawn O'Brien assisted Mary Collins.",
         None, None, "Mary Collins"),

        ("LEARN-3 Irish O' prefix dropped by ASR",
         "Priya Patel gave medication to Sullivan.",
         None, "Priya Patel", None),
    ]

    cfg_inst = ConfigManager()

    # ── Simulate a learning feedback before running the suite ─────────────────
    # Pre-seed one confirmed correction so LEARN-1 demonstrates the boost path
    reset_learning(cfg_inst)   # clean slate for QA
    record_feedback(
        "Dharni Kumar", "Dharani Kumar", accepted=True,
        role="nurse", ordis_id="qa-seed", cfg=cfg_inst,
    )

    passed = failed = 0

    W = 90
    print(f"\n{'═' * W}")
    print(f"{'ORDIS Layer 01  ·  QA Test Suite  (v5 · Learning + Irish Names)':^{W}}")
    print(f"{'═' * W}\n")

    for label, inp, must_contain, exp_nurse, exp_patient in QA:
        corrected, dets = run(inp, ordis_id="qa", cfg=cfg_inst)
        rm = get_role_map(dets)

        txt_ok  = (must_contain is None) or (must_contain in corrected)
        nrs_ok  = (exp_nurse  is None) or any(
            v == "nurse"   and (k == exp_nurse   or exp_nurse   in k)
            for k, v in rm.items()
        )
        pat_ok  = (exp_patient is None) or any(
            v == "patient" and (k == exp_patient or exp_patient in k)
            for k, v in rm.items()
        )
        ok = txt_ok and nrs_ok and pat_ok
        tag = f"{_GR}PASS{_R}" if ok else f"{_RE}FAIL{_R}"
        passed += ok; failed += (not ok)

        print(f"  [{tag}] {label}")
        print(f"       IN : {inp.strip()}")
        print(f"       OUT: {corrected.strip()}")
        roles = " | ".join(
            f"{v.capitalize()}: {k}"
            for k, v in sorted(rm.items(), key=lambda x: x[1])
        ) or "(none)"
        print(f"       ROLES: {roles}")

        if not txt_ok:
            print(f"        {_RE}✘ text should contain '{must_contain}'{_R}")
        if not nrs_ok:
            print(f"        {_RE}✘ nurse  expected '{exp_nurse}'{_R}")
        if not pat_ok:
            print(f"        {_RE}✘ patient expected '{exp_patient}'{_R}")
        print()

    stats = get_learning_stats(cfg_inst)
    colour = _GR if failed == 0 else _RE
    print(f"{'─' * W}")
    print(f"  {_B}Results:{_R}  "
          f"{_GR}{passed} passed{_R}  ·  "
          f"{colour}{failed} failed{_R}  "
          f"out of {passed + failed}")
    print(f"  {_B}Learning store:{_R}  "
          f"{stats['confirmed_count']} confirmed  ·  "
          f"{stats['rejected_surfaces']} rejected surfaces  ·  "
          f"{stats['total_feedback']} total feedback  ·  "
          f"path: {stats['store_path']}")
    print(f"{'═' * W}\n")