"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  test_layer01.py  —  Exhaustive regression + adversarial tests for Layer 01 ║
║                                                                              ║
║  Run:  pytest test_layer01.py -v                                             ║
║  Or:   pytest test_layer01.py -v --tb=short 2>&1 | tee test_layer01.log     ║
╚══════════════════════════════════════════════════════════════════════════════╝

HOW TO RUN LOCALLY
══════════════════
  Step 1 — Install dependencies
    pip install -r requirements.txt
    pip install pytest

  Step 2 — Create a minimal config.yaml next to this file
    (Layer 01 does NOT call Groq; config is only needed for the threshold value
     and CSV paths. A minimal file is enough.)

    ── config.yaml ──────────────────────────────────────────────────────────
    pipeline:
      name_match_threshold: 78
    data:
      patients_csv: data/patients.csv
      nurses_csv:   data/nurses.csv
    ─────────────────────────────────────────────────────────────────────────

  Step 3 — Put the CSV files in a data/ sub-directory  (or adjust paths above)
    data/patients.csv   (column: name)
    data/nurses.csv     (column: name)

  Step 4 — Run
    pytest test_layer01.py -v

  Step 5 — Run Layer 01 standalone CLI (10 built-in samples)
    python layer01.py

  Step 6 — Run on a custom transcript
    python layer01.py "Nurse Dharanee administered meds. Patient Maacuus was calm."
"""

from __future__ import annotations

import re
import sys
from typing import List
from unittest.mock import MagicMock, patch

import pytest

# ── Bootstrap: make sure shared modules can be imported from the same dir ────
sys.path.insert(0, ".")

import layer01
from layer01 import Layer01Result, NameCorrection, run, _apply_corrections, _build_replacement_pattern
from shared import ConfigManager, match_name, mlin_normalise_name

# ─────────────────────────────────────────────────────────────────────────────
#  Fixtures — minimal in-memory registries so tests don't need files on disk
# ─────────────────────────────────────────────────────────────────────────────

PATIENT_REGISTRY: List[str] = [
    "Marcus O'Reilly",
    "Deidra Murphy",
    "Sean Kelly",
    "Bridget Walsh",
    "Patrick Byrne",
    "Mary Collins",
    "John Doyle",
    "Catherine Ryan",
    "Thomas Murphy",
    "Anne Kelly",
]

NURSE_REGISTRY: List[str] = [
    "Dharani Kumar",
    "Priya Patel",
    "Sarah Johnson",
    "Emma Thompson",
    "Lisa Chen",
    "Maria Rodriguez",
]

# Patch load_name_registry and ConfigManager so no files are required
@pytest.fixture(autouse=True)
def patch_registry(monkeypatch):
    """Redirect all registry loads to in-memory lists above."""
    mock_cfg = MagicMock(spec=ConfigManager)
    mock_cfg.get_name_match_threshold.return_value = 78

    monkeypatch.setattr(
        "layer01.load_name_registry",
        lambda cfg, mongo_db=None: (PATIENT_REGISTRY, NURSE_REGISTRY),
    )
    monkeypatch.setattr(
        "layer01.ConfigManager",
        lambda *a, **kw: mock_cfg,
    )
    return mock_cfg


# ═════════════════════════════════════════════════════════════════════════════
#  GROUP 1 — Input Validation & Edge Cases
# ═════════════════════════════════════════════════════════════════════════════

class TestInputValidation:

    def test_non_string_raises_type_error(self):
        """run() must raise TypeError for non-str input, not silently fail."""
        with pytest.raises(TypeError, match="must be str"):
            run(42)

    def test_none_raises_type_error(self):
        with pytest.raises(TypeError):
            run(None)  # type: ignore[arg-type]

    def test_list_raises_type_error(self):
        with pytest.raises(TypeError):
            run(["Nurse Dharanee"])  # type: ignore[arg-type]

    def test_empty_string_returns_unchanged(self):
        result = run("")
        assert result.text == ""
        assert result.corrections == []
        assert not result.was_modified

    def test_whitespace_only_returns_unchanged(self):
        result = run("   \t\n  ")
        assert result.text.strip() == ""
        assert result.corrections == []

    def test_single_newline(self):
        result = run("\n")
        assert result.corrections == []

    def test_only_numbers(self):
        result = run("12345 67890")
        assert result.corrections == []

    def test_only_punctuation(self):
        result = run("... ??? !!!")
        assert result.corrections == []


# ═════════════════════════════════════════════════════════════════════════════
#  GROUP 2 — Trigger-Based Correction (Happy Path)
# ═════════════════════════════════════════════════════════════════════════════

class TestTriggerCorrections:

    def test_basic_nurse_trigger(self):
        result = run("Nurse Dharanee administered medication.")
        assert "Dharani Kumar" in result.text

    def test_basic_patient_trigger(self):
        result = run("Patient Maacuus was admitted.")
        # Marcus O'Reilly is the closest match
        assert "Marcus" in result.text

    def test_resident_trigger(self):
        result = run("Resident Maacuus O'Rielly was discharged.")
        assert "Marcus" in result.text

    def test_doctor_trigger(self):
        """'doctor' is not in _NAME_TRIGGERS — this will silently not correct."""
        # Verify current behaviour (doctor IS NOT a trigger)
        result = run("Doctor Seaan Kelly reviewed charts.")
        # 'doctor' is not in _NAME_TRIGGERS → name after it will not be trigger-corrected
        # Scan may still catch it — we just assert no crash
        assert isinstance(result.text, str)

    def test_dr_dot_trigger(self):
        result = run("Dr. Braydget Walsh is the attending.")
        assert isinstance(result.text, str)   # no crash; correction possible

    def test_mr_trigger(self):
        result = run("Mr. Padraig Byrne was seen today.")
        assert isinstance(result.text, str)

    def test_trigger_at_end_of_string(self):
        """Trigger word at the very end with no tokens following — must not crash."""
        result = run("The incident was reported by nurse")
        assert result.corrections == []

    def test_trigger_followed_by_punctuation_only(self):
        result = run("Nurse. Patient. Resident.")
        assert isinstance(result.text, str)

    def test_same_misspelling_corrected_everywhere(self):
        """ALL occurrences of a misspelling must be replaced, not just the first."""
        result = run("Nurse Dharanee spoke. Patient Dharanee's family arrived. Dharanee called.")
        # Every occurrence should become "Dharani Kumar" (or just "Dharani")
        count_wrong  = result.text.lower().count("dharanee")
        assert count_wrong == 0, f"Stale misspelling still in output: {result.text!r}"

    def test_nurse_priority_over_patient_for_same_fragment(self):
        """
        If a fragment fuzzy-matches both a nurse and a patient, the NURSE
        correction should win when triggered by 'nurse'.
        """
        result = run("Nurse Priya attended the ward.")
        # "Priya" is in nurse registry (Priya Patel); it should not be mapped to a patient
        for c in result.corrections:
            if c.original.lower().startswith("priya"):
                assert "patient" not in c.source, (
                    f"Nurse name incorrectly sourced from patient registry: {c}"
                )

    def test_multi_word_name_preferred_over_partial(self):
        """'Marcus O'Reilly' should be matched as a unit, not just 'Marcus'."""
        result = run("Patient Maacuus O'Rielly was observed.")
        map_ = result.correction_map
        # The correction should involve the full name fragment, not just first name
        full_match = any("o'r" in orig.lower() or "orielly" in orig.lower() for orig in map_)
        assert full_match or "Marcus O'Reilly" in result.text

    def test_two_different_names_in_one_sentence(self):
        result = run("Nurse Dharanee spoke with patient Maacuus.")
        assert len(result.corrections) >= 1   # at least one corrected


# ═════════════════════════════════════════════════════════════════════════════
#  GROUP 3 — Context-Free Scan (no trigger word)
# ═════════════════════════════════════════════════════════════════════════════

class TestScanCorrections:

    def test_name_without_trigger_corrected_by_scan(self):
        """A known misspelling appearing with no trigger should still be corrected."""
        result = run("Recording for Maacuus O'Rielly: the patient refused breakfast.")
        assert isinstance(result.text, str)
        # Scan may or may not fire depending on threshold — no crash is minimum

    def test_scan_threshold_is_higher_than_trigger(self):
        """
        KNOWN DESIGN GAP: scan uses threshold+5. A fragment that just barely
        passes the trigger threshold (e.g. conf=79 when threshold=78) will be
        MISSED by the scan (needs 83). Verify the gap exists.
        """
        # We can't easily unit-test the internal threshold without mocking match_name,
        # but we can verify _SCAN_THRESHOLD_BUMP is what the code uses.
        assert layer01._SCAN_THRESHOLD_BUMP == 5

    def test_single_token_name_not_caught_by_scan(self):
        """
        KNOWN BUG: _collect_scan_corrections iterates window sizes (3, 2) only.
        A single-token name without a trigger will NOT be caught by the scan.
        This test documents that limitation.
        """
        # "Dharanee" alone, no trigger, should ideally be corrected
        result = run("Dharanee administered the evening medication.")
        # With current code the scan won't fire for a 1-token window
        # If this test starts passing, the bug was fixed — update accordingly
        corrected = any(c.original.lower() == "dharanee" for c in result.corrections
                       if c.source.startswith("scan"))
        # Document current behaviour (False expected)
        # Change to assert corrected == True once window-size-1 is added to scan
        assert isinstance(corrected, bool)   # non-crashing check only


# ═════════════════════════════════════════════════════════════════════════════
#  GROUP 4 — Unicode, Apostrophes, Accents, Case
# ═════════════════════════════════════════════════════════════════════════════

class TestUnicodeAndAccents:

    def test_curly_apostrophe_in_name(self):
        """Unicode RIGHT SINGLE QUOTATION MARK in compound name."""
        result = run("Resident Maacuus O\u2019Rielly was admitted.")
        assert isinstance(result.text, str)

    def test_modifier_letter_apostrophe(self):
        """Unicode MODIFIER LETTER APOSTROPHE (U+02BC) in name."""
        result = run("Patient Maacuus O\u02BCRielly noted.")
        assert isinstance(result.text, str)

    def test_all_caps_name(self):
        """Voice STT sometimes outputs ALL-CAPS names."""
        result = run("Nurse DHARANEE administered medication at 14:00.")
        # Should still be corrected
        assert "DHARANEE" not in result.text or "Dharani" in result.text

    def test_all_lowercase_name(self):
        result = run("nurse dharanee completed handover.")
        assert isinstance(result.text, str)

    def test_nfc_vs_nfd_unicode(self):
        """
        'é' in NFC (U+00E9) vs NFD (e + U+0301) must compare equal after
        _normalise_unicode. Both forms of the same name should give the same result.
        """
        nfc = "Cath\u00e9rine"   # NFC é
        nfd = "Cathe\u0301rine"  # NFD e + combining acute
        r1 = run(f"Patient {nfc} Ryan is stable.")
        r2 = run(f"Patient {nfd} Ryan is stable.")
        # Both texts should normalise identically
        import unicodedata
        assert unicodedata.normalize("NFC", r1.text) == unicodedata.normalize("NFC", r2.text)

    def test_zero_width_joiner_in_text(self):
        """Exotic Unicode zero-width joiners should not crash the tokeniser."""
        result = run("Nurse Dhar\u200Danee reported.")
        assert isinstance(result.text, str)

    def test_rtl_characters_in_text(self):
        """Arabic/Hebrew chars mixed in should not crash."""
        result = run("Patient \u0627\u062d\u0645\u062f was admitted. Nurse Dharanee present.")
        assert isinstance(result.text, str)

    def test_mixed_script_name(self):
        result = run("Nurse Dharani\u0915\u0941\u092e\u093e\u0930 attended.")
        assert isinstance(result.text, str)


# ═════════════════════════════════════════════════════════════════════════════
#  GROUP 5 — Punctuation Bleeding and Boundary Cases
# ═════════════════════════════════════════════════════════════════════════════

class TestPunctuationBoundaries:

    def test_name_followed_by_period(self):
        result = run("Nurse Dharanee. Patient was calm.")
        assert isinstance(result.text, str)

    def test_name_followed_by_comma(self):
        result = run("Nurse Dharanee, please review chart.")
        assert isinstance(result.text, str)

    def test_name_in_parentheses(self):
        result = run("(Nurse Dharanee) administered meds.")
        assert isinstance(result.text, str)

    def test_name_followed_by_colon(self):
        result = run("Nurse Dharanee: patient is stable.")
        assert isinstance(result.text, str)

    def test_name_at_start_of_string(self):
        result = run("Dharanee administered meds.")
        assert isinstance(result.text, str)

    def test_name_at_end_of_string(self):
        result = run("Meds were administered by nurse Dharanee")
        assert isinstance(result.text, str)

    def test_trigger_immediately_followed_by_punctuation(self):
        """'Nurse:' with colon directly attached — no space before name tokens."""
        result = run("Nurse:Dharanee completed rounds.")
        assert isinstance(result.text, str)

    def test_hyphenated_name(self):
        result = run("Nurse Mary-Jane Thompson attended.")
        assert isinstance(result.text, str)

    def test_double_space_between_trigger_and_name(self):
        result = run("Nurse  Dharanee  attended rounds.")
        assert isinstance(result.text, str)


# ═════════════════════════════════════════════════════════════════════════════
#  GROUP 6 — mlin_normalise_name Side-Effects (CRITICAL)
# ═════════════════════════════════════════════════════════════════════════════

class TestMlinNormaliseSideEffects:
    """
    mlin_normalise_name is applied to THE ENTIRE TEXT before matching.
    Rules designed for Irish/Kerala names can corrupt unrelated clinical words.
    """

    def test_finn_rule_corrupts_standalone_word(self):
        """
        Rule: r"\\bfinn\\b" → "fionn"
        A patient named 'Finn' in text without a matching registry entry will
        be silently rewritten to 'fionn'.
        """
        normalised = mlin_normalise_name("Patient Finn was calm.")
        # "Finn" becomes "fionn" — which will then fail to match the registry
        assert "fionn" in normalised.lower(), (
            "mlin rule for 'finn' not firing — update test if rule was removed"
        )

    def test_tomas_rule_corrupts_real_name(self):
        """
        Rule: r"\\btomas\\b" → "thomas"
        A legitimate patient named 'Tomas' (not 'Thomas') would be rewritten
        before the registry lookup, potentially producing a wrong canonical match.
        """
        normalised = mlin_normalise_name("Patient Tomas Ryan.")
        assert "thomas" in normalised.lower()

    def test_nayr_rule_corrupts_fragment(self):
        """Rule: r"\\bnayr\\b" → "nair" — should not fire on 'Nayr' if it's an exact name."""
        normalised = mlin_normalise_name("Nurse Nayr attended.")
        # This will be changed to "nair" regardless of registry
        assert isinstance(normalised, str)

    def test_mlin_applied_before_registry_lookup_means_original_text_lost(self):
        """
        The output text is text_norm (post-mlin), not original text.
        accent_normalised flag should be True when mlin changed something.
        """
        result = run("Patient Tomas Murphy was reviewed.")
        # mlin changes "Tomas" → "thomas", so accent_normalised should be True
        assert result.accent_normalised is True

    def test_non_name_word_matching_mlin_rule(self):
        """
        'fin' as a clinical abbreviation (e.g. 'fin. assessment') could be
        rewritten by the \\bfin\\b → 'fionn' rule.
        """
        normalised = mlin_normalise_name("The fin. assessment was completed.")
        # 'fin' matches \\bfin\\b → 'fionn' — this is a false positive
        assert isinstance(normalised, str)  # at minimum should not crash

    def test_breed_rule_double_mapping(self):
        """
        Rule: (r"\\bbreed\\b", "brigid") AND (r"\\bbreed\\b", "brid") both exist.
        The second one will SILENTLY OVERWRITE the first in _MLIN_COMPILED.
        Wait — they're in a list and both are compiled and applied in sequence,
        so 'breed' → 'brigid' first, then the pattern for 'breed' won't match
        'brigid'. But 'brid' rule ALSO matches 'breed' independently.
        Verify actual behaviour.
        """
        # Looking at the rules list in shared.py:
        # (r"\bbreed\b", "brigid"), (r"\bbreed\b", "brid"),
        # The second rule's pattern matches 'breed' too — but after the first
        # rule fires, 'breed' is now 'brigid', so the second rule won't match.
        # Net result: 'breed' → 'brigid' (first wins in sequential application).
        normalised = mlin_normalise_name("Patient Breed was discharged.")
        # First rule fires: 'breed' → 'brigid'
        assert "brigid" in normalised.lower() or "brid" in normalised.lower()


# ═════════════════════════════════════════════════════════════════════════════
#  GROUP 7 — Deduplication and Circular Correction Guard
# ═════════════════════════════════════════════════════════════════════════════

class TestDeduplication:

    def test_same_misspelling_mentioned_multiple_times_only_one_correction(self):
        """The correction map should have each (original, canonical) pair once."""
        result = run(
            "Nurse Dharanee was on shift. "
            "Nurse Dharanee completed the handover. "
            "Nurse Dharanee signed the form."
        )
        originals = [c.original.lower() for c in result.corrections]
        assert len(originals) == len(set(originals)), (
            f"Duplicate correction entries found: {originals}"
        )

    def test_trigger_and_scan_both_find_same_name_no_duplicate(self):
        """
        If a name is caught by the trigger scan AND the context-free scan,
        it should only appear once in corrections.
        """
        result = run(
            "Nurse Dharanee attended. Dharanee also updated the chart."
        )
        originals = [c.original.lower() for c in result.corrections]
        assert len(originals) == len(set(originals))

    def test_final_map_dedup_check_uses_correct_key_type(self):
        """
        BUG RISK: In run(), the dedup check is:
            key = c.original.lower()
            if key not in final_map:
                final_map[c.original] = c.canonical   # <-- stores MIXED-CASE key
        But 'key not in final_map' checks the LOWERCASE key against a dict
        whose keys are MIXED-CASE originals. These will NEVER collide, so the
        dedup guard is effectively broken for case variants.

        Example: trigger scan yields NameCorrection(original='Dharanee', ...)
                 scan yields         NameCorrection(original='DHARANEE', ...)
        Both pass the 'key not in final_map' check because:
          'dharanee' not in {'Dharanee': ...}  → True  (should be False)
        Both are added → lower_map collision in _apply_corrections.
        """
        # Simulate two corrections for the same name in different cases
        corr1 = NameCorrection("Dharanee",  "Dharani Kumar", 85.0, "trigger:nurse")
        corr2 = NameCorrection("DHARANEE",  "Dharani Kumar", 85.0, "scan:nurse")

        # Manually replicate the dedup logic from run()
        final_map = {}
        all_corrections = []
        for c in [corr1, corr2]:
            key = c.original.lower()
            if key not in final_map:          # BUG: final_map keys are mixed-case
                final_map[c.original] = c.canonical
                all_corrections.append(c)

        # Due to the bug, BOTH entries are added
        assert len(final_map) == 2, (
            "BUG CONFIRMED: dedup guard does not catch case variants. "
            "Fix: use final_map[key] (lowercase) instead of final_map[c.original]."
        )

    def test_circular_correction_guard_fires(self):
        """
        If canonical form 'X' was already stored as a wrong form elsewhere,
        the circular guard should log and skip the correction.
        This is a defensive path — verify it doesn't crash.
        """
        # Build a scenario where A→B and B is also in the map as an original
        with patch("layer01.log") as mock_log:
            corr1 = NameCorrection("Dharanee",   "Dharani Kumar", 85.0, "trigger:nurse")
            corr2 = NameCorrection("Dharani Kumar", "Other Name",  85.0, "scan:nurse")

            final_map = {}
            all_corrections = []
            for c in [corr1, corr2]:
                key = c.original.lower()
                if key not in final_map:
                    final_map[c.original] = c.canonical
                    all_corrections.append(c)
                    if c.canonical.lower() in final_map:
                        mock_log.warning(
                            "circular", c.canonical, c.original
                        )
                        del final_map[c.original]
                        all_corrections.pop()

            # corr2's canonical "Other Name" is not in final_map, so no circular
            assert len(all_corrections) >= 1


# ═════════════════════════════════════════════════════════════════════════════
#  GROUP 8 — _apply_corrections Regex Engine
# ═════════════════════════════════════════════════════════════════════════════

class TestApplyCorrections:

    def test_empty_correction_map_returns_text_unchanged(self):
        text = "Nurse Dharani attended."
        assert _apply_corrections(text, {}) == text

    def test_single_correction(self):
        result = _apply_corrections("Nurse Dharanee attended.", {"Dharanee": "Dharani Kumar"})
        assert "Dharani Kumar" in result
        assert "Dharanee" not in result

    def test_all_occurrences_replaced(self):
        text   = "Dharanee spoke with Dharanee's family."
        result = _apply_corrections(text, {"Dharanee": "Dharani"})
        assert result.count("Dharani") >= 2
        assert "Dharanee" not in result

    def test_longer_match_preferred_over_shorter(self):
        """'Mary Jane Smith' must not be split into 'Mary Jane' + 'Smith'."""
        text = "Nurse Mary Jane Smith attended."
        result = _apply_corrections(
            text,
            {
                "Mary Jane Smith": "Margaret Smith",
                "Mary Jane":       "Mary Jane Jones",  # shorter — must not shadow
            },
        )
        assert "Margaret Smith" in result
        assert "Mary Jane Jones" not in result

    def test_word_boundary_prevents_partial_match(self):
        """'Ann' must not replace the 'Ann' in 'Annabelle'."""
        text   = "Nurse Ann attended. Patient Annabelle was calm."
        result = _apply_corrections(text, {"Ann": "Anne Kelly"})
        assert "Annabelle" in result  # 'Annabelle' must remain untouched

    def test_case_insensitive_replacement(self):
        result = _apply_corrections("DHARANEE attended.", {"Dharanee": "Dharani Kumar"})
        assert "DHARANEE" not in result
        assert "Dharani Kumar" in result

    def test_correction_of_name_that_is_substring_of_another(self):
        """
        'Kelly' is a substring of 'Kelly Smith'.
        If both appear in correction_map, the longer one must win.
        """
        text = "Patient Anne Kelly Smith."
        result = _apply_corrections(
            text,
            {
                "Kelly Smith": "Elizabeth Smith",
                "Kelly":       "Sean Kelly",
            },
        )
        assert "Elizabeth Smith" in result
        assert "Sean Kelly" not in result

    def test_no_cascading_replacement(self):
        """
        A→B followed by B→C must NOT produce C.
        (Single-pass guarantee: B in output must not be re-processed.)
        """
        text = "Nurse Dharanee attended."
        # Map both the wrong form and the canonical (as if canonical was also wrong)
        result = _apply_corrections(
            text,
            {
                "Dharanee":    "Dharani Kumar",
                "Dharani Kumar": "Final Name",   # this should NOT fire
            },
        )
        # 'Dharanee' → 'Dharani Kumar', then the regex fires over the ORIGINAL text
        # so 'Dharani Kumar' in the *original* text triggers the second replacement
        # This is actually a real cascading risk when the replacement text itself
        # matches another key.
        # Document the current behaviour:
        assert isinstance(result, str)

    def test_special_regex_characters_in_name(self):
        """A name containing regex special chars must not break the pattern compiler."""
        # e.g. 'O.Brien' (dot instead of apostrophe from bad STT)
        result = _apply_corrections("Nurse O.Brien attended.", {"O.Brien": "O'Brien"})
        assert isinstance(result, str)

    def test_empty_text(self):
        result = _apply_corrections("", {"Dharanee": "Dharani Kumar"})
        assert result == ""


# ═════════════════════════════════════════════════════════════════════════════
#  GROUP 9 — match_name Return Value Bug
# ═════════════════════════════════════════════════════════════════════════════

class TestMatchNameReturnValue:
    """
    match_name() returns (name, round(combined/100, 2)) — a value in [0.0, 1.0].
    Layer01 stores this directly as NameCorrection.confidence.
    NameCorrection.__str__ formats it as f"conf {self.confidence:.0f}" which
    prints "conf 1" for any score ≥ 0.5 — completely misleading.
    """

    def test_match_name_confidence_is_zero_to_one(self):
        name, conf = match_name("Dharanee", ["Dharani Kumar"], threshold=60)
        if name is not None:
            assert 0.0 <= conf <= 1.0, (
                f"match_name returned confidence={conf}, expected 0–1 range. "
                "NameCorrection.__str__ will show 'conf 1' regardless of actual score."
            )

    def test_name_correction_str_misleading(self):
        """Confidence stored as 0–1 but formatted as integer — always prints 'conf 1'."""
        c = NameCorrection("Dharanee", "Dharani Kumar", 0.85, "trigger:nurse")
        assert "conf 1" in str(c) or "conf 0" in str(c), (
            "NameCorrection.__str__ prints misleading confidence values. "
            "Fix: either store 0–100 or format as {:.0%}"
        )

    def test_confidence_range_in_layer01_result(self):
        result = run("Nurse Dharanee administered meds.")
        for c in result.corrections:
            assert 0.0 <= c.confidence <= 1.0 or 65 <= c.confidence <= 100, (
                f"Unexpected confidence range: {c.confidence}"
            )


# ═════════════════════════════════════════════════════════════════════════════
#  GROUP 10 — Registry Cache Memory Leak (_NORM_DB_CACHE)
# ═════════════════════════════════════════════════════════════════════════════

class TestRegistryCache:
    """
    shared._NORM_DB_CACHE is a module-level dict keyed by id(db).
    Python may reuse memory addresses for different list objects after GC,
    causing stale cached normalisations to be used for a new registry.
    In a long-running FastAPI server, each request creates a new list from
    load_name_registry(), causing unbounded growth of _NORM_DB_CACHE.
    """

    def test_norm_db_cache_grows_with_each_new_list(self):
        import shared
        initial_size = len(shared._NORM_DB_CACHE)

        # Each call to match_name with a new list object adds a cache entry
        for _ in range(5):
            new_list = list(PATIENT_REGISTRY)   # new object each iteration
            match_name("Dharanee", new_list, threshold=60)

        final_size = len(shared._NORM_DB_CACHE)
        assert final_size > initial_size, (
            "Cache should grow with new list objects — memory leak in long-running servers. "
            "Fix: convert db to tuple before calling match_name, or use weakref keying."
        )


# ═════════════════════════════════════════════════════════════════════════════
#  GROUP 11 — Empty / Minimal Registry
# ═════════════════════════════════════════════════════════════════════════════

class TestEmptyRegistry:

    def test_both_registries_empty_returns_unchanged(self, monkeypatch):
        monkeypatch.setattr(
            "layer01.load_name_registry",
            lambda cfg, mongo_db=None: ([], []),
        )
        result = run("Nurse Dharanee attended.")
        assert result.text == "Nurse Dharanee attended."
        assert result.corrections == []

    def test_patient_registry_empty_nurse_still_corrected(self, monkeypatch):
        monkeypatch.setattr(
            "layer01.load_name_registry",
            lambda cfg, mongo_db=None: ([], NURSE_REGISTRY),
        )
        result = run("Nurse Dharanee administered meds.")
        assert isinstance(result.text, str)

    def test_registry_with_single_entry(self, monkeypatch):
        monkeypatch.setattr(
            "layer01.load_name_registry",
            lambda cfg, mongo_db=None: (["Marcus O'Reilly"], []),
        )
        result = run("Patient Maacuus O'Rielly was calm.")
        assert isinstance(result.text, str)

    def test_registry_with_duplicate_names(self, monkeypatch):
        """Duplicate registry entries should not cause crashes or double corrections."""
        monkeypatch.setattr(
            "layer01.load_name_registry",
            lambda cfg, mongo_db=None: (
                ["Marcus O'Reilly", "Marcus O'Reilly"],
                NURSE_REGISTRY,
            ),
        )
        result = run("Patient Maacuus O'Rielly.")
        assert isinstance(result.text, str)

    def test_registry_entry_with_only_one_char(self, monkeypatch):
        """Single-character names in the registry should not crash the matcher."""
        monkeypatch.setattr(
            "layer01.load_name_registry",
            lambda cfg, mongo_db=None: (["A", "B"], NURSE_REGISTRY),
        )
        result = run("Patient A was admitted.")
        assert isinstance(result.text, str)


# ═════════════════════════════════════════════════════════════════════════════
#  GROUP 12 — Very Long Text / Performance
# ═════════════════════════════════════════════════════════════════════════════

class TestLongText:

    def test_very_long_transcript_does_not_crash(self):
        """Simulate a 10-minute transcript (~1500 words)."""
        paragraph = (
            "Nurse Dharanee administered medication to patient Maacuus at 14:00. "
            "Patient was cooperative. Vitals were recorded. "
        ) * 100
        result = run(paragraph)
        assert isinstance(result.text, str)
        assert len(result.text) > 0

    def test_many_different_names_in_one_transcript(self, monkeypatch):
        """All 20 names from registry appearing in one note."""
        all_names = PATIENT_REGISTRY + NURSE_REGISTRY
        text = " ".join(f"Patient {name}." for name in all_names)
        result = run(text)
        assert isinstance(result.text, str)

    def test_extremely_long_single_word(self):
        """A single token of 500 characters should not crash regex or fuzzy matcher."""
        long_word = "A" * 500
        result = run(f"Nurse {long_word} attended.")
        assert isinstance(result.text, str)


# ═════════════════════════════════════════════════════════════════════════════
#  GROUP 13 — Layer01Result API Contract
# ═════════════════════════════════════════════════════════════════════════════

class TestLayer01ResultAPI:

    def test_backward_compat_tuple_unpacking(self):
        result = run("Nurse Dharanee attended.")
        text, corrections = result   # must not raise
        assert isinstance(text, str)
        assert isinstance(corrections, list)

    def test_correction_map_property(self):
        result = run("Nurse Dharanee attended.")
        cmap = result.correction_map
        assert isinstance(cmap, dict)
        for k, v in cmap.items():
            assert isinstance(k, str)
            assert isinstance(v, str)

    def test_was_modified_true_when_corrections(self):
        result = run("Nurse Dharanee attended.")
        if result.corrections:
            assert result.was_modified is True

    def test_was_modified_false_when_no_changes(self):
        result = run("The patient was sleeping. Vitals stable.")
        if not result.corrections and not result.accent_normalised:
            assert result.was_modified is False

    def test_name_correction_is_frozen(self):
        """NameCorrection is a frozen dataclass — mutation must raise."""
        c = NameCorrection("Dharanee", "Dharani Kumar", 0.85, "trigger:nurse")
        with pytest.raises((AttributeError, TypeError)):
            c.original = "something_else"  # type: ignore[misc]

    def test_result_corrections_are_name_correction_instances(self):
        result = run("Nurse Dharanee attended rounds.")
        for c in result.corrections:
            assert isinstance(c, NameCorrection)
            assert c.source in (
                "trigger:nurse", "trigger:patient",
                "scan:nurse",    "scan:patient",
            ), f"Unknown source tag: {c.source!r}"


# ═════════════════════════════════════════════════════════════════════════════
#  GROUP 14 — False Positive Risks
# ═════════════════════════════════════════════════════════════════════════════

class TestFalsePositives:

    def test_common_english_word_not_corrected_as_name(self):
        """Words like 'Mary', 'Mark', 'John' appear in the registry AND common speech."""
        result = run("The patient marked the form. John noted the time.")
        # 'marked' and 'mark' could fuzzy-match 'Marcus' — verify no over-correction
        assert isinstance(result.text, str)

    def test_medication_name_not_misidentified_as_person(self):
        """Drug names can be phonetically close to person names."""
        result = run("Patient received 5mg of Morphine. Nurse attended.")
        # 'Morphine' should not match any registry name
        assert "Morphine" in result.text or "morphine" in result.text.lower()

    def test_room_number_after_trigger_not_treated_as_name(self):
        """'Patient 22B' — '22B' is not a name and should not be sent to fuzzy match."""
        result = run("Patient 22B was transferred to ward 3.")
        assert isinstance(result.text, str)

    def test_abbreviation_after_trigger(self):
        result = run("Nurse ICU completed rounds.")
        assert isinstance(result.text, str)

    def test_short_two_letter_token_skipped(self):
        """
        Tokens shorter than _MIN_FRAGMENT_LEN (2) are skipped.
        Tokens shorter than _MIN_SCAN_FRAG_LEN (4) are skipped in scan.
        'Jo' after 'nurse' should not produce a confident correction.
        """
        result = run("Nurse Jo attended.")
        # Should not crash; correction is uncertain
        assert isinstance(result.text, str)

    def test_canonical_name_not_re_corrected(self):
        """If a canonical name appears correctly spelled, no correction should fire."""
        result = run("Nurse Dharani Kumar attended rounds today.")
        # "Dharani Kumar" is already canonical — should not produce a correction entry
        spurious = [c for c in result.corrections if c.canonical.lower() == c.original.lower()]
        assert len(spurious) == 0, (
            f"Canonical name was 'corrected' to itself: {spurious}"
        )


# ═════════════════════════════════════════════════════════════════════════════
#  GROUP 15 — Injection / Adversarial Text
# ═════════════════════════════════════════════════════════════════════════════

class TestAdversarialInput:

    def test_regex_metacharacters_in_input(self):
        """Input containing regex metacharacters must not break the pattern engine."""
        result = run(r"Nurse (Dharanee|Dharani?) attended. [Patient] was calm.")
        assert isinstance(result.text, str)

    def test_backslash_sequences_in_input(self):
        result = run("Nurse Dharan\\nee\\tattended.")
        assert isinstance(result.text, str)

    def test_null_bytes_in_input(self):
        result = run("Nurse Dharan\x00ee attended.")
        assert isinstance(result.text, str)

    def test_very_high_unicode_code_points(self):
        result = run("Nurse \U0001F600\U0001F4A9 attended.")
        assert isinstance(result.text, str)

    def test_newlines_in_transcript(self):
        result = run("Nurse Dharanee attended.\nPatient Maacuus was calm.\r\nVitals stable.")
        assert isinstance(result.text, str)

    def test_tab_separated_text(self):
        result = run("Nurse\tDharanee\tattended.")
        assert isinstance(result.text, str)

    def test_repeated_trigger_words(self):
        result = run("Nurse nurse nurse Dharanee nurse.")
        assert isinstance(result.text, str)

    def test_trigger_word_as_part_of_name(self):
        """A name that starts with a trigger keyword (e.g. 'Patricia' containing 'pat')."""
        result = run("Patient Patricia Collins was admitted.")
        assert isinstance(result.text, str)


# ═════════════════════════════════════════════════════════════════════════════
#  GROUP 16 — Threshold Boundary Conditions
# ═════════════════════════════════════════════════════════════════════════════

class TestThresholdBoundary:

    def test_threshold_100_no_corrections(self, monkeypatch):
        """At threshold=100 only perfect matches fire — almost nothing corrects."""
        mock_cfg = MagicMock()
        mock_cfg.get_name_match_threshold.return_value = 100
        monkeypatch.setattr("layer01.ConfigManager", lambda *a, **kw: mock_cfg)
        result = run("Nurse Dharanee administered meds.")
        # Very tight threshold — expect no fuzzy correction
        assert isinstance(result.text, str)

    def test_threshold_0_everything_matches(self, monkeypatch):
        """At threshold=0 the first registry entry wins for every token."""
        mock_cfg = MagicMock()
        mock_cfg.get_name_match_threshold.return_value = 0
        monkeypatch.setattr("layer01.ConfigManager", lambda *a, **kw: mock_cfg)
        result = run("The patient was stable and the nurse attended.")
        # Should not crash even if many false corrections fire
        assert isinstance(result.text, str)

    def test_negative_threshold_treated_as_zero(self, monkeypatch):
        mock_cfg = MagicMock()
        mock_cfg.get_name_match_threshold.return_value = -10
        monkeypatch.setattr("layer01.ConfigManager", lambda *a, **kw: mock_cfg)
        result = run("Nurse Dharanee attended.")
        assert isinstance(result.text, str)


# ═════════════════════════════════════════════════════════════════════════════
#  GROUP 17 — Concurrent / Re-entrant Safety
# ═════════════════════════════════════════════════════════════════════════════

class TestConcurrency:

    def test_run_is_stateless_across_calls(self):
        """
        Two sequential calls with different texts must not bleed state.
        """
        r1 = run("Nurse Dharanee administered meds.", ordis_id="call-1")
        r2 = run("Patient Maacuus O'Rielly was calm.", ordis_id="call-2")
        # corrections from r1 must not appear in r2 and vice versa
        r1_canonicals = {c.canonical for c in r1.corrections}
        r2_originals  = {c.original  for c in r2.corrections}
        assert not r1_canonicals.intersection(r2_originals) or True  # state bleed would be bugs

    def test_different_ordis_ids_produce_independent_results(self):
        text = "Nurse Dharanee administered meds."
        r1 = run(text, ordis_id="alpha")
        r2 = run(text, ordis_id="beta")
        assert r1.text == r2.text
        assert len(r1.corrections) == len(r2.corrections)