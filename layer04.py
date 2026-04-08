"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ORDIS — layer04.py                                                          ║
║  Layer 04 · PII Reversal                                                     ║
║                                                                              ║
║  Input  : Professional text from Layer 03B (tokens still in place)           ║
║  Output : Final polished note with real names restored                       ║
║                                                                              ║
║  Method : Load PII map from  output/pii_map_{Ordis_ID}.json                 ║
║           Regex-based whole-boundary token → real_name substitution          ║
║                                                                              ║
║  API    : POST /api/layer04  (txt, accent, Ordis_ID)                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import logging
import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from shared import (
    ConfigManager,
    load_pii_map,
    _GR, _YL, _GY, _R,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Module-level logger — honours the host application's logging config
# ─────────────────────────────────────────────────────────────────────────────
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────

#: All token prefixes that Layer 02 is allowed to emit.
#: Extend this tuple if new prefix types are added upstream.
_KNOWN_PREFIXES: Tuple[str, ...] = ("PATIENT", "NURSE", "DOCTOR", "CARER", "STAFF")

#: Compiled pattern that detects *any* unreversed structured token.
#: Used in the post-substitution sanity sweep.
_RESIDUAL_TOKEN_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(p) for p in _KNOWN_PREFIXES) + r")\d+\b"
)

#: Characters that are valid *inside* a real name but NOT word characters
#: under \w (e.g. apostrophe in O'Reilly, hyphen in Anne-Marie, dot in Dr.).
_NAME_INNER_CHARS = r"[A-Za-z0-9'\-\.]"


# ─────────────────────────────────────────────────────────────────────────────
#  Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class ReversalResult:
    """
    Immutable result object returned by :func:`run`.

    Attributes
    ----------
    text :
        Final output — all reversible PII tokens replaced by real names.
    reversals :
        Ordered list of ``(token, real_name)`` pairs that were substituted,
        in the order they appeared in the text (first occurrence).
    residual_tokens :
        Any structured tokens that were present in the text but absent from
        the PII map — these could not be reversed.
    elapsed_ms :
        Wall-clock time for the reversal step alone (excludes map I/O).
    stats :
        Aggregate counts for observability / monitoring.
    """
    text:             str
    reversals:        Tuple[Tuple[str, str], ...]
    residual_tokens:  FrozenSet[str]
    elapsed_ms:       float
    stats:            "ReversalStats"

    @property
    def is_clean(self) -> bool:
        """True when no unreversed tokens remain in the output text."""
        return len(self.residual_tokens) == 0

    @property
    def reversal_count(self) -> int:
        return len(self.reversals)


@dataclass(frozen=True, slots=True)
class ReversalStats:
    tokens_found:    int   # unique tokens present in input text
    tokens_reversed: int   # unique tokens successfully reversed
    tokens_missed:   int   # unique tokens in text but NOT in map
    occurrences:     int   # total individual replacements made


# ─────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_text(text: str) -> str:
    """
    NFC-normalise unicode so that composed and decomposed forms of the same
    character are treated identically.

    E.g. ``é`` (U+00E9) and ``e`` + combining acute (U+0065 U+0301) become
    the same codepoint sequence before any regex is applied.
    """
    return unicodedata.normalize("NFC", text)


def _normalise_map(pii_map: Dict[str, str]) -> Dict[str, str]:
    """
    Return a sanitised copy of the PII map:

    * NFC-normalise both keys and values.
    * Strip surrounding whitespace from keys/values.
    * Drop entries where key or value is empty after stripping.
    * Warn about duplicate keys that differ only in case (kept as-is).

    Does NOT mutate the caller's dict.
    """
    seen_upper: Dict[str, str] = {}
    out: Dict[str, str] = {}

    for raw_token, raw_name in pii_map.items():
        token = _normalise_text(raw_token).strip()
        name  = _normalise_text(raw_name).strip()

        if not token:
            log.warning("L4: PII map contained an empty token key — skipped.")
            continue
        if not name:
            log.warning("L4: PII map entry '%s' has an empty real name — skipped.", token)
            continue

        upper = token.upper()
        if upper in seen_upper and seen_upper[upper] != token:
            log.warning(
                "L4: PII map has near-duplicate tokens '%s' and '%s' "
                "(differ only in case). Both kept — results may be unexpected.",
                seen_upper[upper], token,
            )
        seen_upper[upper] = token
        out[token] = name

    return out


def _build_token_pattern(token: str) -> re.Pattern[str]:
    """
    Compile a pattern that matches *token* as a complete, standalone unit —
    not as part of a longer token or alphanumeric sequence — while correctly
    handling punctuation on both sides.

    Design decisions
    ----------------
    * ``(?<![A-Za-z0-9_])``  — left boundary: token must not follow a word char.
    * ``(?![A-Za-z0-9_])``   — right boundary: token must not precede a word char.
    * We use ``re.escape`` on the token so that any special regex chars inside
      are treated as literals (tokens should be all-caps+digits, but be safe).
    * Compiled with no flags — tokens are case-sensitive by design.

    Examples that MUST match
    ~~~~~~~~~~~~~~~~~~~~~~~~
    ``PATIENT1``        plain occurrence
    ``PATIENT1.``       token followed by full stop
    ``(PATIENT1)``      token inside parentheses
    ``"PATIENT1"``      token inside quotation marks
    ``PATIENT1,``       token followed by comma
    ``\nPATIENT1``      token at start of line
    ``PATIENT1\n``      token at end of line
    ``PATIENT1's``      possessive (English)  → must NOT match; handled below

    Examples that MUST NOT match
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ``PATIENT10``       when token is ``PATIENT1``  (longer token)
    ``MYPATIENT1``      alpha prefix before token
    ``PATIENT1X``       alpha suffix after token
    ``PATIENT1_NOTE``   underscore suffix
    """
    escaped = re.escape(token)
    pattern = r"(?<![A-Za-z0-9_])" + escaped + r"(?![A-Za-z0-9_])"
    return re.compile(pattern)


def _replace_all_occurrences(
    text:       str,
    token:      str,
    real_name:  str,
    pattern:    re.Pattern[str],
) -> Tuple[str, int]:
    """
    Replace every non-overlapping occurrence of *token* (matched by *pattern*)
    with *real_name*.

    Returns
    -------
    (new_text, count)
        ``count`` is the number of individual substitutions made.
    """
    new_text, n = pattern.subn(real_name, text)
    return new_text, n


def _scan_tokens_in_text(text: str, known_tokens: Sequence[str]) -> FrozenSet[str]:
    """
    Return the subset of *known_tokens* that are actually present in *text*,
    using the same boundary rules as :func:`_build_token_pattern`.

    Runs in O(k) compiled-regex matches where k = len(known_tokens).
    """
    present: set[str] = set()
    for token in known_tokens:
        if re.search(r"(?<![A-Za-z0-9_])" + re.escape(token) + r"(?![A-Za-z0-9_])", text):
            present.add(token)
    return frozenset(present)


def _find_residual_tokens(text: str) -> FrozenSet[str]:
    """
    Scan *text* for any unreversed structured tokens matching the known-prefix
    pattern (e.g. ``PATIENT3``, ``NURSE12``).

    This is the post-substitution sanity check — it catches tokens that were
    in the text but absent from the PII map (i.e. genuine data gaps, not bugs
    in the replacement logic).
    """
    return frozenset(_RESIDUAL_TOKEN_RE.findall(text))


# ─────────────────────────────────────────────────────────────────────────────
#  Validation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _validate_inputs(
    text:     Optional[str],
    ordis_id: Optional[str],
    pii_map:  Optional[Dict[str, str]],
) -> Optional[str]:
    """
    Return an error message string if inputs are invalid, else ``None``.

    Checks
    ------
    * ``text`` must be a non-None string (empty string is technically valid —
      there is nothing to reverse, which is not an error).
    * Either ``ordis_id`` (non-empty, non-whitespace) or ``pii_map`` must be
      supplied so the map can be loaded/used.
    """
    if text is None:
        return "text must be a string, got None."
    if not isinstance(text, str):
        return f"text must be str, got {type(text).__name__}."
    if pii_map is None and not (ordis_id and ordis_id.strip()):
        return (
            "Either a non-empty ordis_id or a pre-loaded pii_map must be "
            "provided so that PII tokens can be resolved."
        )
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────

def run(
    text:     str,
    accent:   str                    = "ml_In",
    ordis_id: str                    = "",
    cfg:      Optional[ConfigManager] = None,
    pii_map:  Optional[Dict[str, str]] = None,
) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Layer 04 — PII Reversal  (compatibility shim).

    This function is the **public API surface** consumed by the ORDIS pipeline
    and the ``POST /api/layer04`` endpoint.  It delegates all logic to
    :func:`run_full` and unpacks the result for backward compatibility with
    callers that expect ``(str, list)`` rather than a :class:`ReversalResult`.

    Parameters
    ----------
    text :
        Professional text from Layer 03B, still containing ``PATIENT``/``NURSE``
        (and other prefix) tokens.
    accent :
        Accent profile identifier — not consumed by this layer; preserved for
        pipeline API consistency.
    ordis_id :
        Session UUID.  Must match the ID used in Layer 02.  If *pii_map* is
        supplied directly this parameter is ignored.
    cfg :
        :class:`ConfigManager` instance.  A default instance is created when
        ``None`` is passed.
    pii_map :
        Pre-loaded PII map (``{token: real_name}``).  When supplied, no disk
        I/O is performed.  Useful for in-memory pipeline runs or unit tests.

    Returns
    -------
    (final_text, reversals)
        *final_text* — the note with all reversible PII tokens replaced.
        *reversals*  — list of ``(token, real_name)`` pairs that were applied.

    Raises
    ------
    ValueError
        When *text* is ``None`` or not a ``str``.

    Notes
    -----
    * This layer is **purely deterministic** — no LLM call is made.
    * The substitution engine is safe for Unicode text (NFC-normalised).
    * Tokens are replaced longest-first to prevent partial-token collisions
      (e.g. ``PATIENT10`` is resolved before ``PATIENT1``).
    * All edge-case warnings are emitted through the ``logging`` subsystem at
      ``WARNING`` level *and* echoed to stdout with ANSI colour codes to
      preserve visibility in terminal pipelines.
    """
    result = run_full(
        text=text,
        accent=accent,
        ordis_id=ordis_id,
        cfg=cfg,
        pii_map=pii_map,
    )
    return result.text, list(result.reversals)


def run_full(
    text:     str,
    accent:   str                    = "ml_In",
    ordis_id: str                    = "",
    cfg:      Optional[ConfigManager] = None,
    pii_map:  Optional[Dict[str, str]] = None,
) -> ReversalResult:
    """
    Layer 04 — PII Reversal  (full-fidelity result).

    Identical to :func:`run` but returns a :class:`ReversalResult` dataclass
    that carries statistics, residual-token information, and timing data.

    Edge cases handled
    ──────────────────
    Input / type safety
      * ``text`` is ``None``                        → ``ValueError``
      * ``text`` is not a ``str``                   → ``ValueError``
      * ``text`` is empty string                    → clean early exit, no-op
      * Both ``ordis_id`` and ``pii_map`` absent    → ``ValueError``
      * ``ordis_id`` is whitespace-only             → treated as absent

    PII map
      * Map fails to load from disk                 → empty-map path, all tokens preserved
      * Map is ``None`` after load attempt          → empty-map path
      * Map is an empty dict                        → clean early exit, no-op
      * Map entries with empty key or value         → silently skipped
      * Near-duplicate keys (differ only in case)   → both kept, warning emitted
      * Non-string keys or values in map            → coerced to str, warning emitted
      * Map values are ``None``                     → entry skipped, warning emitted

    Token matching
      * Token is a substring of a longer token
        (``PATIENT1`` vs ``PATIENT10``)             → resolved by sorting longest-first
      * Token appears with surrounding punctuation
        (``PATIENT1.`` / ``(PATIENT1)`` / etc.)     → matched via boundary regex
      * Token at start/end of string                → matched correctly
      * Token on its own line                       → matched correctly
      * Token in multi-line / CRLF text             → normalised before processing
      * Token appears zero times in text            → skipped silently (no entry in reversals)
      * Token appears multiple times in text        → all occurrences replaced
      * Token in ALL-CAPS context                   → unchanged (tokens are already uppercase)
      * Unicode text (e.g. accented real names)     → handled via NFC normalisation

    Post-substitution
      * Tokens in text but not in PII map           → logged, reported in ``residual_tokens``
      * No tokens found at all                      → clean exit with empty reversals
      * Text already clean (no tokens)              → no-op, returns input unchanged
    """
    # ── 0. Validate inputs ────────────────────────────────────────────────────
    err = _validate_inputs(text, ordis_id, pii_map)
    if err:
        raise ValueError(f"L4 input error: {err}")

    if cfg is None:
        cfg = ConfigManager()

    # ── 1. Guard: empty text ──────────────────────────────────────────────────
    if not text:
        log.debug("L4: Received empty text — nothing to reverse.")
        return ReversalResult(
            text="",
            reversals=(),
            residual_tokens=frozenset(),
            elapsed_ms=0.0,
            stats=ReversalStats(0, 0, 0, 0),
        )

    # ── 2. Normalise text (unicode NFC + CRLF → LF) ──────────────────────────
    normalised_text = _normalise_text(text.replace("\r\n", "\n").replace("\r", "\n"))

    # ── 3. Load / sanitise PII map ────────────────────────────────────────────
    if pii_map is None:
        loaded = _load_map_safe(ordis_id)
        if loaded is None:
            _warn(f"L4: Could not load PII map for session '{ordis_id}'. "
                  "Tokens will remain in output.")
            residual = _find_residual_tokens(normalised_text)
            return ReversalResult(
                text=normalised_text,
                reversals=(),
                residual_tokens=residual,
                elapsed_ms=0.0,
                stats=ReversalStats(
                    tokens_found=len(residual),
                    tokens_reversed=0,
                    tokens_missed=len(residual),
                    occurrences=0,
                ),
            )
        pii_map = loaded

    clean_map = _normalise_map(_coerce_map_values(pii_map))

    if not clean_map:
        log.info("L4: PII map is empty after sanitisation — nothing to reverse.")
        print(f"    {_GY}L4: PII map is empty — nothing to reverse.{_R}")
        residual = _find_residual_tokens(normalised_text)
        return ReversalResult(
            text=normalised_text,
            reversals=(),
            residual_tokens=residual,
            elapsed_ms=0.0,
            stats=ReversalStats(
                tokens_found=len(residual),
                tokens_reversed=0,
                tokens_missed=len(residual),
                occurrences=0,
            ),
        )

    # ── 4. Pre-scan: which tokens from the map are actually present? ──────────
    tokens_in_text = _scan_tokens_in_text(normalised_text, list(clean_map.keys()))
    tokens_not_in_text = frozenset(clean_map.keys()) - tokens_in_text

    if tokens_not_in_text:
        log.debug(
            "L4: %d map token(s) not found in text (no-op): %s",
            len(tokens_not_in_text),
            sorted(tokens_not_in_text),
        )

    # ── 5. Substitution pass ──────────────────────────────────────────────────
    #   Sort descending by token length so that, e.g., PATIENT10 is processed
    #   before PATIENT1, preventing partial-token contamination.
    sorted_entries: List[Tuple[str, str]] = sorted(
        ((tok, clean_map[tok]) for tok in tokens_in_text),
        key=lambda pair: -len(pair[0]),
    )

    t_start      = time.perf_counter()
    result_text  = normalised_text
    reversals:   List[Tuple[str, str]] = []
    total_occurrences = 0

    for token, real_name in sorted_entries:
        pattern = _build_token_pattern(token)
        result_text, n = _replace_all_occurrences(result_text, token, real_name, pattern)

        if n > 0:
            reversals.append((token, real_name))
            total_occurrences += n
            log.debug("L4: '%s' → '%s' (%d occurrence(s))", token, real_name, n)
            print(f"    {_GR}L4: {token} → '{real_name}' ({n}×){_R}")
        else:
            # Pattern found token in pre-scan but subn found none — extremely
            # unlikely unless the text was mutated between steps; log defensively.
            log.warning(
                "L4: Token '%s' was present at pre-scan but produced 0 "
                "substitutions. Text may have changed unexpectedly.", token,
            )

    elapsed_ms = (time.perf_counter() - t_start) * 1_000

    # ── 6. Post-substitution sanity sweep ─────────────────────────────────────
    residual_tokens = _find_residual_tokens(result_text)

    if residual_tokens:
        msg = (
            f"L4: ⚠ {len(residual_tokens)} unreversed token(s) remain "
            f"(not in PII map): {sorted(residual_tokens)}"
        )
        log.warning(msg)
        print(f"    {_YL}{msg}{_R}")
    elif not reversals:
        log.info("L4: No tokens found to reverse — text was already clean.")
        print(f"    {_GY}L4: No tokens found to reverse — text is already clean.{_R}")
    else:
        log.info(
            "L4: %d unique token(s) reversed (%d total occurrence(s)) in %.2f ms.",
            len(reversals), total_occurrences, elapsed_ms,
        )
        print(
            f"    L4: {len(reversals)} token(s) reversed, "
            f"{total_occurrences} occurrence(s) in {elapsed_ms:.2f} ms."
        )

    # ── 7. Assemble and return ────────────────────────────────────────────────
    stats = ReversalStats(
        tokens_found    = len(tokens_in_text),
        tokens_reversed = len(reversals),
        tokens_missed   = len(residual_tokens),
        occurrences     = total_occurrences,
    )

    return ReversalResult(
        text            = result_text,
        reversals       = tuple(reversals),
        residual_tokens = residual_tokens,
        elapsed_ms      = elapsed_ms,
        stats           = stats,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Internal utilities
# ─────────────────────────────────────────────────────────────────────────────

def _load_map_safe(ordis_id: str) -> Optional[Dict[str, str]]:
    """
    Attempt to load the PII map for *ordis_id*, catching all exceptions.

    Returns ``None`` on any failure instead of propagating so that the
    pipeline can degrade gracefully (tokens preserved rather than crash).
    """
    try:
        return load_pii_map(ordis_id)
    except FileNotFoundError:
        log.error("L4: PII map file not found for session '%s'.", ordis_id)
    except PermissionError:
        log.error("L4: Permission denied reading PII map for session '%s'.", ordis_id)
    except Exception as exc:  # noqa: BLE001
        log.exception("L4: Unexpected error loading PII map for '%s': %s", ordis_id, exc)
    return None


def _coerce_map_values(pii_map: Dict) -> Dict[str, str]:
    """
    Defensively coerce PII map keys and values to ``str``.

    Any entry where the value is ``None`` is dropped with a warning.
    Non-string keys/values of other types are stringified with a warning.
    """
    out: Dict[str, str] = {}
    for k, v in pii_map.items():
        if v is None:
            log.warning("L4: PII map entry %r has a None value — skipped.", k)
            continue
        str_k = k if isinstance(k, str) else str(k)
        str_v = v if isinstance(v, str) else str(v)
        if not isinstance(k, str):
            log.warning("L4: PII map key %r is not a str — coerced to '%s'.", k, str_k)
        if not isinstance(v, str):
            log.warning("L4: PII map value for '%s' is not a str — coerced to '%s'.", str_k, str_v)
        out[str_k] = str_v
    return out


def _warn(msg: str) -> None:
    """Emit *msg* via the logger (WARNING) and to stdout with ANSI colour."""
    log.warning(msg)
    print(f"    {_YL}{msg}{_R}")


# ─────────────────────────────────────────────────────────────────────────────
#  Self-contained test suite (run with:  python layer04.py --test)
# ─────────────────────────────────────────────────────────────────────────────

def _run_tests() -> None:  # pragma: no cover
    """
    Exhaustive edge-case test suite.  Executed when the module is invoked with
    the ``--test`` flag.  Uses ``assert`` for simplicity; no external framework
    required.
    """
    import sys

    PASS = "\033[32mPASS\033[0m"
    FAIL = "\033[31mFAIL\033[0m"
    results: List[Tuple[str, bool, str]] = []

    def check(name: str, got: str, want: str) -> None:
        ok = got == want
        results.append((name, ok, f"got={got!r}  want={want!r}"))

    BASE_MAP = {
        "PATIENT1":  "Marcus O'Reilly",
        "PATIENT2":  "Deidra",
        "PATIENT10": "Harriet Osei-Bonsu",
        "NURSE1":    "Dharani",
        "NURSE2":    "Anne-Marie",
        "DOCTOR1":   "Dr. Patel",
    }

    # ── Basic replacement ──────────────────────────────────────────────────────
    out, _ = run("PATIENT1 is stable.", pii_map=BASE_MAP)
    check("basic_replacement", out, "Marcus O'Reilly is stable.")

    # ── Multiple unique tokens ─────────────────────────────────────────────────
    out, _ = run("NURSE1 assessed PATIENT2.", pii_map=BASE_MAP)
    check("multiple_tokens", out, "Dharani assessed Deidra.")

    # ── Same token multiple occurrences ───────────────────────────────────────
    out, _ = run("PATIENT1 needs review. PATIENT1 is improving.", pii_map=BASE_MAP)
    check("multiple_occurrences", out, "Marcus O'Reilly needs review. Marcus O'Reilly is improving.")

    # ── Longest-first: PATIENT10 must not become PATIENT1 + "0" ──────────────
    out, _ = run("PATIENT10 and PATIENT1 both seen.", pii_map=BASE_MAP)
    check("longest_first", out, "Harriet Osei-Bonsu and Marcus O'Reilly both seen.")

    # ── Token followed by punctuation ─────────────────────────────────────────
    out, _ = run("Reviewed PATIENT1.", pii_map=BASE_MAP)
    check("token_before_fullstop", out, "Reviewed Marcus O'Reilly.")

    out, _ = run("(PATIENT1) discussed.", pii_map=BASE_MAP)
    check("token_in_parens", out, "(Marcus O'Reilly) discussed.")

    out, _ = run("PATIENT1, PATIENT2, NURSE1.", pii_map=BASE_MAP)
    check("token_before_comma", out, "Marcus O'Reilly, Deidra, Dharani.")

    out, _ = run('"PATIENT1" was assessed.', pii_map=BASE_MAP)
    check("token_in_quotes", out, '"Marcus O\'Reilly" was assessed.')

    out, _ = run("PATIENT1: NBM.", pii_map=BASE_MAP)
    check("token_before_colon", out, "Marcus O'Reilly: NBM.")

    out, _ = run("PATIENT1; reviewed.", pii_map=BASE_MAP)
    check("token_before_semicolon", out, "Marcus O'Reilly; reviewed.")

    # ── Token at very start / end of string ───────────────────────────────────
    out, _ = run("PATIENT1", pii_map=BASE_MAP)
    check("token_at_start_and_end", out, "Marcus O'Reilly")

    out, _ = run("PATIENT1\n", pii_map=BASE_MAP)
    check("token_at_end_with_newline", out, "Marcus O'Reilly\n")

    out, _ = run("\nPATIENT1", pii_map=BASE_MAP)
    check("token_at_start_with_newline", out, "\nMarcus O'Reilly")

    # ── Token must NOT match as substring of longer alphanumeric ─────────────
    out, _ = run("MYPATIENT1 arrived.", pii_map=BASE_MAP)
    check("no_match_alpha_prefix", out, "MYPATIENT1 arrived.")

    out, _ = run("PATIENT1X discharged.", pii_map=BASE_MAP)
    check("no_match_alpha_suffix", out, "PATIENT1X discharged.")

    out, _ = run("PATIENT1_NOTE updated.", pii_map=BASE_MAP)
    check("no_match_underscore_suffix", out, "PATIENT1_NOTE updated.")

    # ── Real name contains special chars (apostrophe, hyphen, dot) ────────────
    out, _ = run("NURSE2 on shift.", pii_map=BASE_MAP)
    check("name_with_hyphen", out, "Anne-Marie on shift.")

    out, _ = run("DOCTOR1 reviewed.", pii_map=BASE_MAP)
    check("name_with_dot", out, "Dr. Patel reviewed.")

    # ── Unicode real name ─────────────────────────────────────────────────────
    unicode_map = {"PATIENT1": "Søren Ångström"}
    out, _ = run("PATIENT1 is stable.", pii_map=unicode_map)
    check("unicode_real_name", out, "Søren Ångström is stable.")

    # ── Empty text ────────────────────────────────────────────────────────────
    out, _ = run("", pii_map=BASE_MAP)
    check("empty_text", out, "")

    # ── Text with no tokens ───────────────────────────────────────────────────
    out, _ = run("All stable on the ward.", pii_map=BASE_MAP)
    check("no_tokens_in_text", out, "All stable on the ward.")

    # ── Token in map but not in text ──────────────────────────────────────────
    out, reversals = run("All stable.", pii_map=BASE_MAP)
    check("token_in_map_not_in_text", out, "All stable.")
    assert len(reversals) == 0, f"Expected 0 reversals, got {reversals}"

    # ── Token in text but not in map → preserved as residual ─────────────────
    result = run_full("PATIENT99 reviewed.", pii_map=BASE_MAP)
    assert "PATIENT99" in result.residual_tokens, "PATIENT99 should be residual"
    check("residual_token_preserved", result.text, "PATIENT99 reviewed.")

    # ── Empty PII map ─────────────────────────────────────────────────────────
    out, _ = run("PATIENT1 is stable.", pii_map={})
    check("empty_pii_map", out, "PATIENT1 is stable.")

    # ── Map entry with None value → skipped ───────────────────────────────────
    map_with_none = {"PATIENT1": None, "PATIENT2": "Deidra"}
    out, _ = run("PATIENT1 and PATIENT2.", pii_map=map_with_none)
    check("none_value_skipped", out, "PATIENT1 and Deidra.")

    # ── Map entry with non-str key/value → coerced ───────────────────────────
    map_non_str = {"PATIENT1": 12345, 67890: "NonStrKey"}
    out, _ = run("PATIENT1 present.", pii_map=map_non_str)
    check("non_str_value_coerced", out, "12345 present.")

    # ── CRLF line endings ─────────────────────────────────────────────────────
    out, _ = run("PATIENT1 is\r\nstable.", pii_map=BASE_MAP)
    check("crlf_normalisation", out, "Marcus O'Reilly is\nstable.")

    # ── All-token sentence ────────────────────────────────────────────────────
    out, _ = run("PATIENT1 PATIENT2 NURSE1", pii_map=BASE_MAP)
    check("all_tokens_no_punctuation", out, "Marcus O'Reilly Deidra Dharani")

    # ── Missing ordis_id + no pii_map → ValueError ────────────────────────────
    raised = False
    try:
        run("PATIENT1 reviewed.", ordis_id="", pii_map=None)
    except ValueError:
        raised = True
    results.append(("missing_ordis_and_map_raises", raised, ""))

    # ── None text → ValueError ─────────────────────────────────────────────────
    raised = False
    try:
        run(None, pii_map=BASE_MAP)  # type: ignore[arg-type]
    except ValueError:
        raised = True
    results.append(("none_text_raises", raised, ""))

    # ── ReversalResult.is_clean flag ──────────────────────────────────────────
    result_clean = run_full("PATIENT1 reviewed.", pii_map=BASE_MAP)
    results.append(("is_clean_true", result_clean.is_clean, ""))

    result_dirty = run_full("PATIENT99 reviewed.", pii_map=BASE_MAP)
    results.append(("is_clean_false", not result_dirty.is_clean, ""))

    # ── Stats accuracy ────────────────────────────────────────────────────────
    result_stats = run_full(
        "PATIENT1 and PATIENT1 were seen. NURSE1 reported.",
        pii_map=BASE_MAP,
    )
    assert result_stats.stats.occurrences == 3, result_stats.stats
    assert result_stats.stats.tokens_found == 2, result_stats.stats
    assert result_stats.stats.tokens_reversed == 2, result_stats.stats
    results.append(("stats_occurrences", True, ""))

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "─" * 72)
    print("  LAYER 04  ·  Edge-Case Test Results")
    print("─" * 72)
    failed = 0
    for name, ok, detail in results:
        status = PASS if ok else FAIL
        print(f"  {status}  {name}")
        if not ok:
            print(f"         {detail}")
            failed += 1
    print("─" * 72)
    total = len(results)
    print(f"  {total - failed}/{total} passed")
    print("─" * 72 + "\n")
    sys.exit(0 if failed == 0 else 1)


# ─────────────────────────────────────────────────────────────────────────────
#  CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path
    from shared import OUTPUT_DIR

    logging.basicConfig(
        level  = logging.INFO,
        format = "%(levelname)-8s %(name)s: %(message)s",
    )

    if "--test" in sys.argv:
        _run_tests()
        sys.exit(0)

    sample = (
        "PATIENT1 (room 22) requires 1:1 continuous supervision and "
        "double-assist for all personal cares. NURSE1 reported that PATIENT2 "
        "is NBM (nil by mouth). PATIENT3 has a palliative care plan in place. "
        "PATIENT4 declined all cares (Resident Declined Help). "
        "PATIENT10 was reviewed by DOCTOR1."
    )

    text_in = sys.argv[1] if len(sys.argv) > 1 else sample
    session  = sys.argv[2] if len(sys.argv) > 2 else "test_session_001"

    # For standalone demo: bootstrap a dummy PII map if none exists on disk.
    pii_path = OUTPUT_DIR / f"pii_map_{session}.json"
    if not pii_path.exists():
        dummy_map = {
            "PATIENT1":  "Marcus O'Reilly",
            "PATIENT2":  "Deidra",
            "PATIENT3":  "Liam",
            "PATIENT4":  "William Oak",
            "PATIENT10": "Harriet Osei-Bonsu",
            "NURSE1":    "Dharani",
            "DOCTOR1":   "Dr. Patel",
        }
        pii_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pii_path, "w", encoding="utf-8") as fh:
            json.dump(dummy_map, fh, ensure_ascii=False, indent=2)
        print(f"  ℹ  Created dummy PII map at {pii_path}\n")

    result = run_full(text_in, ordis_id=session)

    print("\n── OUTPUT ───────────────────────────────────────────────────────────")
    print(result.text)
    print(f"\n── REVERSALS ({result.reversal_count}) ────────────────────────────────────────")
    for token, name in result.reversals:
        print(f"  {token}  →  '{name}'")
    if result.residual_tokens:
        print(f"\n── RESIDUAL TOKENS (not in map) ─────────────────────────────────────")
        for tok in sorted(result.residual_tokens):
            print(f"  {tok}")
    print(f"\n── STATS ────────────────────────────────────────────────────────────")
    print(f"  found:    {result.stats.tokens_found}")
    print(f"  reversed: {result.stats.tokens_reversed}")
    print(f"  missed:   {result.stats.tokens_missed}")
    print(f"  total Δ:  {result.stats.occurrences}")
    print(f"  time:     {result.elapsed_ms:.3f} ms")
    print(f"  clean:    {result.is_clean}")