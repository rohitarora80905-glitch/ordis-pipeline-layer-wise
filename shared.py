"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ORDIS — shared.py                                                           ║
║  Shared utilities used by all layer modules                                  ║
║                                                                              ║
║  Contains:                                                                   ║
║    · ConfigManager       — loads config.yaml                                 ║
║    · GroqRateLimiter     — sliding-window TPM/RPM guard                      ║
║    · ModelRouter         — unified LLM interface (Groq)                      ║
║    · MedicalTermDatabase — loads medical_terms CSV (Columns A, B, D)         ║
║    · Name-matching utils — fuzzy + phonetic resolution (ml-IN accent rules)  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import ast
import csv
import functools
import json
import os
import random
import re
import time
import unicodedata
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import jellyfish
import pandas as pd
import yaml
from bson import ObjectId
from groq import Groq
from pymongo import MongoClient

try:
    from groq import RateLimitError as GroqRateLimitError
except ImportError:
    GroqRateLimitError = Exception  # type: ignore[misc,assignment]

from rapidfuzz import fuzz, process

# ── ANSI colours ──────────────────────────────────────────────────────────────
_R  = "\033[0m"
_B  = "\033[1m"
_CY = "\033[96m"
_YL = "\033[93m"
_GR = "\033[92m"
_RE = "\033[91m"
_GY = "\033[90m"

BASE_DIR   = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class ConfigManager:
    """Loads config.yaml and exposes typed accessors."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            print(f"  ⚠  config.yaml not found at '{self.config_path}'. Using defaults.")
            return {}
        with open(self.config_path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}

    def get_active_provider(self) -> str:
        return self.config.get("active_provider", "groq")

    def get_provider_config(self, provider: str) -> Dict[str, Any]:
        return self.config.get("models", {}).get(provider, {})

    def resolve_groq_api_key(self) -> str:
        yaml_key = self.get_provider_config("groq").get("api_key", "").strip()
        if yaml_key:
            return yaml_key
        env_key = os.environ.get("GROQ_API_KEY", "").strip()
        if env_key:
            return env_key
        raise ValueError(
            "Groq API key not found. Set 'api_key' in config.yaml or "
            "export GROQ_API_KEY environment variable."
        )

    def get_data_path(self, key: str) -> Path:
        raw = self.config.get("data", {}).get(key, "")
        return BASE_DIR / raw if raw else BASE_DIR

    def get_mongodb_config(self) -> Dict[str, Any]:
        return self.config.get("mongodb", {})

    def get_mongodb_uri(self) -> str:
        return self.get_mongodb_config().get("uri", "").strip()

    def get_mongodb_db_name(self) -> str:
        return self.get_mongodb_config().get("db_name", "").strip()

    def get_mongodb_collection_name(self, key: str, default: str) -> str:
        return self.get_mongodb_config().get("collections", {}).get(key, default)

    def get_name_match_threshold(self) -> int:
        return int(self.config.get("pipeline", {}).get("name_match_threshold", 78))

    def get_rate_limit_config(self) -> Dict[str, Any]:
        groq_cfg = self.get_provider_config("groq")
        return {
            "tpm_limit":  int(groq_cfg.get("rate_limit_tpm", 6000)),
            "rpm_limit":  int(groq_cfg.get("rate_limit_rpm", 28)),
            "window_sec": float(groq_cfg.get("rate_limit_window_sec", 60.0)),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  GROQ RATE LIMITER
# ─────────────────────────────────────────────────────────────────────────────

class GroqRateLimiter:
    """Sliding-window TPM + RPM rate limiter for Groq API."""

    def __init__(self, tpm_limit: int = 6000, rpm_limit: int = 28, window_sec: float = 60.0):
        self.tpm_limit  = tpm_limit
        self.rpm_limit  = rpm_limit
        self.window_sec = window_sec
        self._token_log: Deque[Tuple[float, int]] = deque()
        self._req_log:   Deque[float]             = deque()

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_sec
        while self._token_log and self._token_log[0][0] < cutoff:
            self._token_log.popleft()
        while self._req_log and self._req_log[0] < cutoff:
            self._req_log.popleft()

    @staticmethod
    def estimate_tokens(text: str) -> int:
        return max(1, int(len(text.split()) * 1.35 * 1.10))

    def pre_call(self, estimated_tokens: int) -> None:
        while True:
            now = time.monotonic()
            self._prune(now)
            tokens_used  = sum(t for _, t in self._token_log)
            reqs_made    = len(self._req_log)
            need_sleep   = 0.0

            if estimated_tokens > (self.tpm_limit - tokens_used):
                if self._token_log:
                    need_sleep = max(need_sleep, (self._token_log[0][0] + self.window_sec) - now + 0.1)
                else:
                    need_sleep = max(need_sleep, self.window_sec)

            if reqs_made >= self.rpm_limit:
                if self._req_log:
                    need_sleep = max(need_sleep, (self._req_log[0] + self.window_sec) - now + 0.1)
                else:
                    need_sleep = max(need_sleep, self.window_sec)

            if need_sleep <= 0:
                break
            print(f"  ⏳  Rate limit: sleeping {need_sleep:.1f}s (TPM {tokens_used}/{self.tpm_limit})")
            time.sleep(need_sleep)

    def post_call(self, tokens_used: int) -> None:
        now = time.monotonic()
        self._token_log.append((now, tokens_used))
        self._req_log.append(now)


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL ROUTER
# ─────────────────────────────────────────────────────────────────────────────

class ModelRouter:
    """Unified LLM interface. All providers currently route through Groq."""

    def __init__(self, config_manager: ConfigManager):
        self.cfg          = config_manager
        self.provider     = config_manager.get_active_provider()
        self._pcfg        = config_manager.get_provider_config("groq")
        self._client      = Groq(api_key=config_manager.resolve_groq_api_key())
        rl_cfg            = config_manager.get_rate_limit_config()
        self._rate_limiter = GroqRateLimiter(**rl_cfg)
        self.model        = self._pcfg.get("model", "openai/gpt-oss-120b")
        self.temperature  = float(self._pcfg.get("temperature", 0.0))
        self.max_tokens   = int(self._pcfg.get("max_tokens_l3", 2000))

    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 2000) -> str:
        """Send messages to Groq with rate-limit protection and exponential backoff."""
        _MAX_RETRIES  = 6
        _BASE_BACKOFF = 2.0
        _MAX_BACKOFF  = 90.0

        all_content = " ".join(m.get("content", "") for m in messages)
        total_est   = GroqRateLimiter.estimate_tokens(all_content) + max_tokens

        for attempt in range(_MAX_RETRIES + 1):
            self._rate_limiter.pre_call(total_est)
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                )
                text = response.choices[0].message.content.strip()
                actual = (
                    response.usage.prompt_tokens + response.usage.completion_tokens
                    if response.usage else total_est
                )
                self._rate_limiter.post_call(actual)
                return text

            except GroqRateLimitError as exc:
                m = re.search(r"retry.after[^\d]*(\d+\.?\d*)", str(exc), re.IGNORECASE)
                wait = float(m.group(1)) + 1.0 if m else min(
                    _BASE_BACKOFF * (2 ** attempt) * random.uniform(0.8, 1.2), _MAX_BACKOFF
                )
                if attempt >= _MAX_RETRIES:
                    return f"ERROR: Groq rate limit — max retries exhausted. {exc}"
                print(f"  ⚠  Groq 429 (attempt {attempt + 1}). Sleeping {wait:.1f}s…")
                time.sleep(wait)
            except Exception as exc:
                return f"ERROR: Groq API error — {exc}"

        return "ERROR: Groq API — retry loop exhausted"

    def is_error(self, response: str) -> bool:
        return response.startswith("ERROR:")


# ─────────────────────────────────────────────────────────────────────────────
#  MEDICAL TERM DATABASE
# ─────────────────────────────────────────────────────────────────────────────

class MedicalTermDatabase:
    """
    Loads medical_terms CSV with columns:
      Term               (Column A) — standard clinical term
      Actual Meaning     (Column B) — professional/full meaning
      Voice Misinterpretations (Column D) — list of misheard variants

    Exposes:
      misinterpretation_map  : {misheard_lower → standard_term (ColA)}
      professional_map       : {standard_term_lower → professional (ColB)}
      lookup_table_for_prompt: formatted markdown table for LLM injection
    """

    def __init__(self, csv_path: Path):
        self.csv_path             = csv_path
        self.misinterpretation_map: Dict[str, str] = {}   # ColD → ColA
        self.professional_map:      Dict[str, str] = {}   # ColA → ColB
        self.rows:                  List[Dict]     = []
        self._sorted_mis:           List[Tuple[str, str]] = []
        self._load()

    def _load(self) -> None:
        if not self.csv_path.exists():
            print(f"  ⚠  Medical terms CSV not found at {self.csv_path}")
            return
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                term       = row.get("Term", "").strip()
                meaning    = row.get("Actual Meaning", "").strip()
                raw_mis    = row.get("Voice Misinterpretations", "[]").strip()
                self.rows.append(row)
                if term:
                    self.professional_map[term.lower()] = meaning

                try:
                    mis_list = ast.literal_eval(raw_mis)
                except (ValueError, SyntaxError):
                    mis_list = [x.strip().strip("'\"") for x in raw_mis.strip("[]").split(",") if x.strip()]

                for mis in mis_list:
                    key = mis.strip().lower()
                    if key:
                        self.misinterpretation_map[key] = term

        self._sorted_mis = sorted(self.misinterpretation_map.items(), key=lambda x: -len(x[0]))
        print(f"  ✔  Medical DB: {len(self.rows)} terms, {len(self.misinterpretation_map)} misinterpretation variants")

    def build_col_d_to_a_table(self) -> str:
        """Return a markdown table of misheard (ColD) → standard (ColA) for LLM prompt injection."""
        if not self.rows:
            return ""
        lines = ["| Misheard (Voice) | Standard Term |", "|---|---|"]
        for bad, good in self._sorted_mis[:120]:   # cap at 120 rows to avoid context overflow
            lines.append(f"| {bad} | {good} |")
        return "\n".join(lines)

    def build_col_a_to_b_table(self) -> str:
        """Return a markdown table of standard (ColA) → professional (ColB) for LLM prompt injection."""
        if not self.rows:
            return ""
        lines = ["| Standard Term (ColA) | Professional Term (ColB) |", "|---|---|"]
        for row in self.rows:
            term    = row.get("Term", "").strip()
            meaning = row.get("Actual Meaning", "").strip()
            if term and meaning and term.lower() != meaning.lower():
                lines.append(f"| {term} | {meaning} |")
        return "\n".join(lines)

    def apply_prepass(self, text: str) -> Tuple[str, List[Tuple[str, str]]]:
        """Deterministic regex-based pre-pass: ColD → ColA replacements."""
        result      = text
        corrections = []
        for bad, good in self._sorted_mis:
            pattern = r"(?<![a-zA-Z0-9])" + re.escape(bad) + r"(?![a-zA-Z0-9])"
            if re.search(pattern, result, re.IGNORECASE):
                result = re.sub(pattern, good, result, flags=re.IGNORECASE)
                corrections.append((bad, good))
        return result, corrections


# ─────────────────────────────────────────────────────────────────────────────
#  NAME REGISTRY LOADER
# ─────────────────────────────────────────────────────────────────────────────

def get_mongo_db(cfg: ConfigManager) -> Optional[Any]:
    """Return a pymongo database object if MongoDB is configured."""
    uri = cfg.get_mongodb_uri()
    if not uri:
        return None
    db_name = cfg.get_mongodb_db_name() or "ordis"
    return MongoClient(uri)[db_name]


def load_name_registry(cfg: ConfigManager, mongo_db: Optional[Any] = None) -> Tuple[List[str], List[str]]:
    """Load patient and nurse name lists from MongoDB or fallback CSV files."""
    if mongo_db is None and cfg.get_mongodb_uri():
        mongo_db = get_mongo_db(cfg)

    patient_names: List[str] = []
    nurse_names:   List[str] = []

    if mongo_db is not None:
        patients_collection = cfg.get_mongodb_collection_name("patients", "patients")
        nurses_collection   = cfg.get_mongodb_collection_name("nurses", "users")

        try:
            patient_names = [
                doc["name"] for doc in mongo_db[patients_collection].find({"name": {"$exists": True}}, {"name": 1})
                if doc.get("name")
            ]
            nurse_names = [
                doc["userName"] for doc in mongo_db[nurses_collection].find({"userName": {"$exists": True}}, {"userName": 1})
                if doc.get("userName")
            ]
            print(f"  ✔  MongoDB registry: {len(patient_names)} patient names, {len(nurse_names)} nurse names")
            return patient_names, nurse_names
        except Exception as exc:
            print(f"  ⚠  MongoDB registry lookup failed: {exc}")

    patients_csv = cfg.get_data_path("patients_csv")
    nurses_csv   = cfg.get_data_path("nurses_csv")

    if patients_csv.exists():
        patient_names = list(pd.read_csv(patients_csv)["name"].dropna())
        print(f"  ✔  Patient registry: {len(patient_names)} entries")
    else:
        print(f"  ⚠  No patients.csv at {patients_csv}")

    if nurses_csv.exists():
        nurse_names = list(pd.read_csv(nurses_csv)["name"].dropna())
        print(f"  ✔  Nurse registry: {len(nurse_names)} entries")
    else:
        print(f"  ⚠  No nurses.csv at {nurses_csv}")

    return patient_names, nurse_names


def load_transcription_text(
    cfg: ConfigManager,
    mongo_db: Any,
    transcription_id: Optional[str] = None,
    patient_id: Optional[str] = None,
) -> str:
    """Load raw transcription text from MongoDB using a transcription or patient id."""
    if mongo_db is None:
        raise ValueError("MongoDB is not configured")

    if not transcription_id and not patient_id:
        raise ValueError("transcription_id or patient_id is required to load text from MongoDB.")

    notes_collection = cfg.get_mongodb_collection_name("patient_notes", "patient_notes")

    def parse_id(value: str):
        try:
            return ObjectId(value)
        except Exception:
            return value

    if transcription_id:
        oid = parse_id(transcription_id)
        query = {"patientsNotes.transcriptions._id": oid}
        doc = mongo_db[notes_collection].find_one(query)
        if not doc:
            raise ValueError(f"Transcription not found for id {transcription_id}")
        for patient_note in doc.get("patientsNotes", []):
            for transcription in patient_note.get("transcriptions", []):
                if transcription.get("_id") == oid or str(transcription.get("_id")) == str(transcription_id):
                    return transcription.get("text", "")
        raise ValueError(f"Transcription id {transcription_id} was found but text could not be loaded.")

    if patient_id:
        pid = parse_id(patient_id)
        query = {"patientsNotes.patientId": pid}
        doc = mongo_db[notes_collection].find_one(query)
        if not doc:
            raise ValueError(f"Patient notes not found for patient_id {patient_id}")
        texts: List[str] = []
        for patient_note in doc.get("patientsNotes", []):
            if patient_note.get("patientId") == pid or str(patient_note.get("patientId")) == str(patient_id):
                for transcription in patient_note.get("transcriptions", []):
                    text = transcription.get("text", "")
                    if text:
                        texts.append(text)
        if not texts:
            raise ValueError(f"No transcriptions found for patient_id {patient_id}")
        return "\n\n".join(texts)

    raise ValueError("Unable to load transcription text from MongoDB.")


# ─────────────────────────────────────────────────────────────────────────────
#  ML-IN ACCENT NORMALISER  (Kerala/Irish phonology)
# ─────────────────────────────────────────────────────────────────────────────

_MLIN_NAME_RULES: List[Tuple[str, str]] = [
    # Irish first names
    (r"\bshon\b",     "sean"),  (r"\bshawn\b",  "sean"),   (r"\bniv\b",    "niamh"),
    (r"\bneev\b",     "niamh"), (r"\baife\b",   "aoife"),  (r"\beefa\b",   "aoife"),
    (r"\bkeeran\b",   "ciaran"),(r"\bkiran\b",  "ciaran"), (r"\bkieran\b", "ciaran"),
    (r"\bgranya\b",   "grainne"),(r"\bsineadh?\b","sinead"),(r"\bnoola\b",  "nuala"),
    (r"\bmev\b",      "maeve"), (r"\bmaev\b",   "maeve"),  (r"\bmeev\b",   "maeve"),
    (r"\bkeeva\b",    "caoimhe"),(r"\bkweva\b", "caoimhe"),(r"\bleesha\b", "laoise"),
    (r"\bfeekra\b",   "fiachra"),(r"\bowan\b",  "eoghan"), (r"\bewon\b",   "eoghan"),
    (r"\broree\b",    "ruairi"), (r"\brooree\b","ruairi"),  (r"\bfarrel\b", "fearghal"),
    (r"\bsive\b",     "sadhbh"), (r"\bsyve\b",  "sadhbh"),
    (r"\borla\b",     "orlaith"),(r"\borlaw\b", "orlaith"),
    (r"\bsorka\b",    "sorcha"), (r"\bsurka\b", "sorcha"),
    (r"\bkilian\b",   "cillian"),(r"\bkillian\b","cillian"),
    (r"\bdermid\b",   "diarmuid"),(r"\bdermot\b","diarmuid"),
    (r"\bbreed\b",    "brigid"), (r"\bbreed\b", "brid"),
    (r"\bconker\b",   "conor"),  (r"\bconkur\b","conor"),
    (r"\bpaurik\b",   "padraig"),(r"\bpadrig\b","padraig"),
    (r"\baishlin\b",  "aisling"),(r"\bashling\b","aisling"),(r"\bashleen\b","aisling"),
    (r"\btige\b",     "tadhg"),  (r"\bteague\b","tadhg"),
    (r"\bdeerdra\b",  "deirdre"),(r"\bdarag\b", "darragh"),
    (r"\bcloda\b",    "clodagh"),(r"\bronen\b", "ronan"),
    (r"\bbrenden\b",  "brendan"),(r"\bsined\b", "sinead"),
    (r"\bfinn\b",     "fionn"),  (r"\bfin\b",   "fionn"),
    # Irish surnames
    (r"\boneill\b",   "oneill"), (r"\boreilly\b","oreilly"),(r"\bosullivan\b","osullivan"),
    (r"\bodonel\b",   "odonnell"),(r"\bohara\b", "ohara"),  (r"\boshe\b",   "oshea"),
    (r"\bmccarty\b",  "mccarthy"),(r"\bmakarti\b","mccarthy"),
    (r"\bkeli\b",     "kelly"),  (r"\bmurfy\b", "murphy"),
    (r"\bnolen\b",    "nolan"),  (r"\bnolan\b", "nolan"),
    (r"\brayen\b",    "ryan"),   (r"\brayan\b", "ryan"),
    (r"\bdoyel\b",    "doyle"),  (r"\blinch\b", "lynch"),
    (r"\bmcgrat\b",   "mcgrath"),(r"\bkavanah\b","kavanagh"),
    (r"\bgalger\b",   "gallagher"),(r"\bgallager\b","gallagher"),
    (r"\bbrenen\b",   "brennan"),(r"\bbrenan\b","brennan"),
    # Kerala names
    (r"\bvargis\b",   "varghese"),(r"\bvargees\b","varghese"),(r"\bvargese\b","varghese"),
    (r"\bchako\b",    "chacko"), (r"\bjacko\b", "chacko"),
    (r"\bpillay\b",   "pillai"), (r"\bpilli\b", "pillai"),
    (r"\bkurop\b",    "kurup"),  (r"\bjorj\b",  "george"),
    (r"\btomas\b",    "thomas"), (r"\bmathu\b", "mathew"),
    (r"\bjosif\b",    "joseph"), (r"\bnayr\b",  "nair"),
    (r"\bmenoan\b",   "menon"),  (r"\bkrishnen\b","krishnan"),
    (r"\biyar\b",     "iyer"),   (r"\biyyer\b", "iyer"),
    (r"\btankachan\b","thankachan"),(r"\bbaboo\b","babu"),
    (r"\bsooresh\b",  "suresh"), (r"\bshuresh\b","suresh"),
    (r"\bsantosh\b",  "santhosh"),(r"\bbindhu\b","bindu"),
    (r"\bsreejit\b",  "sreejith"),(r"\bsreejeet\b","sreejith"),
    (r"\bnambear\b",  "nambiar"),(r"\bnambiyar\b","nambiar"),
    # O' prefix variants (space-separated)
    (r"\bo\s+neil\b",  "oneill"),  (r"\bo\s+brien\b", "obrien"),
    (r"\bo\s+reilly\b","oreilly"), (r"\bo\s+hara\b",  "ohara"),
    (r"\bo\s+shea\b",  "oshea"),   (r"\bo\s+donnell\b","odonnell"),
]

_MLIN_COMPILED = [(re.compile(p, re.IGNORECASE), r) for p, r in _MLIN_NAME_RULES]


def mlin_normalise_name(text: str) -> str:
    """Apply ml-IN phonetic normalisation rules for names."""
    t = text
    for pattern, replacement in _MLIN_COMPILED:
        t = pattern.sub(replacement, t)
    return t


# ─────────────────────────────────────────────────────────────────────────────
#  FUZZY + PHONETIC NAME MATCHING
# ─────────────────────────────────────────────────────────────────────────────

def _strip_name(name: str) -> str:
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    return re.sub(r"['\u2019\u02bc]", "", name).lower()


_PREFIX_RE = re.compile(r"^(mc|mac|o|st)\s*'?\s*", re.IGNORECASE)


def _normalise_prefix(token: str) -> str:
    t = token.lower()
    m = _PREFIX_RE.match(t)
    if not m:
        return t
    prefix = m.group(1).lower()
    stem   = t[m.end():]
    if prefix == "mac":
        prefix = "mc"
    return prefix + stem


@functools.lru_cache(maxsize=4096)
def _phonetic_codes(word: str) -> frozenset:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return frozenset()
    return frozenset(filter(None, [jellyfish.soundex(w), jellyfish.metaphone(w)]))


def _score_token_pair(ft: str, ct: str) -> float:
    ft_n = _normalise_prefix(ft)
    ct_n = _normalise_prefix(ct)
    s    = float(fuzz.WRatio(ft_n, ct_n))
    if s >= 60 and _phonetic_codes(ft_n) & _phonetic_codes(ct_n):
        s = min(100.0, s + 10)
    return s


_NORM_DB_CACHE: Dict[int, List[str]] = {}


def match_name(fragment: str, db: List[str], threshold: int = 78) -> Tuple[Optional[str], float]:
    """
    Bijective token-aligned fuzzy + phonetic name matching.
    Returns (best_match_name, confidence_0_to_1) or (None, 0.0) if no match.
    """
    if not fragment.strip() or not db:
        return None, 0.0

    norm_frag   = _strip_name(mlin_normalise_name(fragment))
    frag_tokens = [t for t in norm_frag.split() if len(t) >= 2]
    if not frag_tokens:
        return None, 0.0

    # Lower threshold for single-token (first-name only) fragments
    eff_threshold = max(threshold - 8, 65) if len(frag_tokens) == 1 else threshold

    db_key = id(db)
    if db_key not in _NORM_DB_CACHE:
        _NORM_DB_CACHE[db_key] = [_strip_name(n) for n in db]
    norm_db = _NORM_DB_CACHE[db_key]

    scored: List[Tuple[float, str]] = []
    for orig, norm in zip(db, norm_db):
        cand_tokens = [t for t in norm.split() if len(t) >= 2]
        if not cand_tokens:
            continue
        # Bijective token scoring
        claimed: set = set()
        token_scores: List[float] = []
        for ft in frag_tokens:
            best_s, best_i = 0.0, -1
            for ci, ct in enumerate(cand_tokens):
                if ci not in claimed:
                    s = _score_token_pair(ft, ct)
                    if s > best_s:
                        best_s, best_i = s, ci
            if best_i >= 0:
                claimed.add(best_i)
            token_scores.append(best_s)

        if min(token_scores) < 62:
            continue

        mean_tok  = sum(token_scores) / len(token_scores)
        full_s    = float(fuzz.WRatio(norm_frag, norm))
        count_pen = min(abs(len(frag_tokens) - len(cand_tokens)) * 4, 12)
        combined  = mean_tok * 0.85 + full_s * 0.15 - count_pen

        if combined >= eff_threshold:
            scored.append((combined, orig))

    if not scored:
        return None, 0.0
    scored.sort(key=lambda x: -x[0])
    return scored[0][1], round(scored[0][0] / 100, 2)


def find_names_in_text(
    text: str,
    registry: List[str],
    threshold: int = 75,
) -> List[Tuple[str, str, float]]:
    """
    Slide a window over the text tokens to find registry name occurrences.
    Returns list of (original_text_span, matched_name, confidence).

    Used by Layer 02 to identify all names that need PII redaction.
    """
    found: List[Tuple[str, str, float]] = []
    words = text.split()

    # Try 1, 2, and 3-word windows
    for window in (3, 2, 1):
        for i in range(len(words) - window + 1):
            span     = " ".join(words[i:i + window])
            # Skip very short or purely numeric spans
            if len(span) < 3 or span.isdigit():
                continue
            match, conf = match_name(span, registry, threshold=threshold)
            if match:
                # Check this span isn't already covered
                already = any(span.lower() in prev_span.lower() or prev_span.lower() in span.lower()
                              for prev_span, _, _ in found)
                if not already:
                    found.append((span, match, conf))
    return found


# ─────────────────────────────────────────────────────────────────────────────
#  PII MAP PERSISTENCE  (shared between Layer 02 and Layer 04)
# ─────────────────────────────────────────────────────────────────────────────

def save_pii_map(ordis_id: str, pii_map: Dict[str, str]) -> None:
    """Persist the PII token→real_name map for a session."""
    path = OUTPUT_DIR / f"pii_map_{ordis_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(pii_map, f, indent=2, ensure_ascii=False)


def load_pii_map(ordis_id: str) -> Dict[str, str]:
    """Load the PII map for a session. Returns empty dict if not found."""
    path = OUTPUT_DIR / f"pii_map_{ordis_id}.json"
    if not path.exists():
        print(f"  ⚠  PII map not found for Ordis_ID '{ordis_id}'. Layer 02 must run first.")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
