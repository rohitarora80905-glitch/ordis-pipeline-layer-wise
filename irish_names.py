"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ORDIS — irish_names.py                                                      ║
║  Curated Irish Name Registry for Layer 01                                    ║
║                                                                              ║
║  Purpose                                                                     ║
║    Provides a bootstrapped set of common Irish first names, surnames, and    ║
║    full name combinations.  Integrated by Layer 01 to supplement the live    ║
║    patient registry so phonetic/fuzzy matching works even for names not yet  ║
║    enrolled — particularly important when Malayali nurses pronounce Irish     ║
║    names phonetically (e.g. "Siobhán" said as "Shivawn").                   ║
║                                                                              ║
║  Contents                                                                    ║
║    IRISH_MALE_FIRST       — 60 common Irish male given names                ║
║    IRISH_FEMALE_FIRST     — 60 common Irish female given names              ║
║    IRISH_SURNAMES         — 80 common Irish family names                    ║
║    IRISH_PHONETIC_VARIANTS— phonetic → canonical map (Malayali-accent safe) ║
║    get_all_irish_names()  — flat list of realistic full-name combinations    ║
║    get_phonetic_variants()— returns the variant map                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
from typing import Dict, List

# ─────────────────────────────────────────────────────────────────────────────
#  First names — Male
# ─────────────────────────────────────────────────────────────────────────────

IRISH_MALE_FIRST: List[str] = [
    "Aidan", "Aaron", "Barry", "Brian", "Brendan",
    "Cathal", "Cian", "Ciarán", "Cillian", "Colm",
    "Conor", "Cormac", "Darragh", "Declan", "Denis",
    "Dermot", "Diarmuid", "Donal", "Dylan", "Eamon",
    "Eoin", "Fergal", "Fiachra", "Fionn", "Gary",
    "Gerard", "Hugh", "James", "John", "Kevin",
    "Kieran", "Liam", "Mark", "Michael", "Muiris",
    "Niall", "Oisín", "Pádraig", "Patrick", "Paul",
    "Peter", "Philip", "Rónán", "Ronan", "Ruairí",
    "Ryan", "Seamus", "Sean", "Shane", "Stephen",
    "Thomas", "Tim", "Tony", "Vincent", "William",
    "Conal", "Donnacha", "Fearghal", "Lorcan", "Tadhg",
]

# ─────────────────────────────────────────────────────────────────────────────
#  First names — Female
# ─────────────────────────────────────────────────────────────────────────────

IRISH_FEMALE_FIRST: List[str] = [
    "Áine", "Aisling", "Aoife", "Bríd", "Brigid",
    "Caitlin", "Caoimhe", "Caroline", "Ciara", "Claire",
    "Clodagh", "Deirdre", "Eileen", "Eimear", "Emer",
    "Emma", "Fiona", "Fionnuala", "Gráinne", "Helen",
    "Joan", "Kathleen", "Lauren", "Maeve", "Máire",
    "Margaret", "Mary", "Michelle", "Niamh", "Nóra",
    "Nuala", "Orla", "Patricia", "Rachel", "Roisin",
    "Rónán", "Saoirse", "Sharon", "Sinéad", "Siobhán",
    "Sorcha", "Susan", "Tracey", "Una", "Áine",
    "Blathnaid", "Clíona", "Eithne", "Laoise", "Méabh",
    "Muireann", "Niamh", "Orlaith", "Sadhbh", "Treasa",
    "Ann", "Anne", "Catherine", "Christine", "Lisa",
]

# ─────────────────────────────────────────────────────────────────────────────
#  Surnames
# ─────────────────────────────────────────────────────────────────────────────

IRISH_SURNAMES: List[str] = [
    "Barry", "Brady", "Brennan", "Burke", "Byrne",
    "Campbell", "Clarke", "Collins", "Connolly", "Connor",
    "Cullen", "Daly", "Donnelly", "Doyle", "Dunne",
    "Farrell", "Fitzgerald", "Fitzpatrick", "Flynn", "Foley",
    "Gallagher", "Griffin", "Hayes", "Healy", "Hughes",
    "Johnston", "Kelly", "Kennedy", "Lynch", "Maher",
    "Malone", "Mathews", "McCarthy", "McDonagh", "McGrath",
    "McLoughlin", "Moore", "Moran", "Moriarty", "Morris",
    "Murphy", "Murray", "Nolan", "O'Brien", "O'Callaghan",
    "O'Carroll", "O'Connor", "O'Doherty", "O'Donnell", "O'Donoghue",
    "O'Gorman", "O'Herlihy", "O'Keeffe", "O'Mahony", "O'Neill",
    "O'Reilly", "O'Riordan", "O'Sullivan", "Power", "Quinn",
    "Riordan", "Ryan", "Sheridan", "Smith", "Sweeney",
    "Thompson", "Walsh", "White", "Wilson", "Fitzgibbon",
    "Bourke", "Broderick", "Delaney", "Fanning", "Hogan",
    "Keane", "Kearney", "Kinsella", "Lawlor", "Lennon",
    "Loftus", "Meagher", "Moloney", "Mulcahy", "Nagle",
]

# ─────────────────────────────────────────────────────────────────────────────
#  Phonetic variants
#  Key  : phonetic spelling a Malayali speaker would use when transcribing
#  Value: the correct canonical Irish form
#
#  This feeds mlin_normalise_name() in shared.py via the accent normalisation
#  pipeline, enabling Layer 01's L2 pass to correct ASR artefacts from
#  Malayalam-accented English pronunciation of Irish names.
# ─────────────────────────────────────────────────────────────────────────────

IRISH_PHONETIC_VARIANTS: Dict[str, str] = {
    # Irish spellings that diverge from English phonetics
    "shivawn":       "Siobhán",
    "shivon":        "Siobhán",
    "shavon":        "Siobhán",
    "shivaan":       "Siobhán",
    "keeva":         "Caoimhe",
    "kweeva":        "Caoimhe",
    "caoimhe":       "Caoimhe",
    "neev":          "Niamh",
    "neeav":         "Niamh",
    "niv":           "Niamh",
    "eersha":        "Saoirse",
    "searsha":       "Saoirse",
    "seersha":       "Saoirse",
    "ashling":       "Aisling",
    "ashleen":       "Aisling",
    "ayshling":      "Aisling",
    "eefa":          "Aoife",
    "eefah":         "Aoife",
    "aifa":          "Aoife",
    "orla":          "Orla",
    "orlagh":        "Orla",
    "sorcha":        "Sorcha",
    "surra":         "Sorcha",
    "granya":        "Gráinne",
    "grainne":       "Gráinne",
    "grawn ya":      "Gráinne",
    "maeve":         "Maeve",
    "mayve":         "Maeve",
    "sinead":        "Sinéad",
    "shinade":       "Sinéad",
    "shunaid":       "Sinéad",
    "deirdre":       "Deirdre",
    "derdree":       "Deirdre",
    "nuala":         "Nuala",
    "noola":         "Nuala",
    "fionnuala":     "Fionnuala",
    "finnoola":      "Fionnuala",
    "finula":        "Fionnuala",
    "eimer":         "Eimear",
    "aimer":         "Eimear",
    "emer":          "Emer",
    "clodagh":       "Clodagh",
    "cloda":         "Clodagh",
    "blathnaid":     "Blathnaid",
    "blahnid":       "Blathnaid",
    "taig":          "Tadhg",
    "tig":           "Tadhg",
    "tague":         "Tadhg",
    "tierna":        "Tiernan",
    "caran":         "Ciarán",
    "kiaran":        "Ciarán",
    "kieran":        "Ciarán",
    "conan":         "Conal",
    "oishin":        "Oisín",
    "ushin":         "Oisín",
    "rori":          "Ruairí",
    "rory":          "Ruairí",
    "ruree":         "Ruairí",
    "padraig":       "Pádraig",
    "podraig":       "Pádraig",
    "pawrick":       "Pádraig",
    "donokha":       "Donnacha",
    "donnakha":      "Donnacha",
    "feargal":       "Fearghal",
    "fergle":        "Fearghal",
    "lorcan":        "Lorcan",
    "lurcan":        "Lorcan",
    "diarmid":       "Diarmuid",
    "dermid":        "Diarmuid",
    "jarmed":        "Diarmuid",
    "laisha":        "Laoise",
    "leesha":        "Laoise",
    "meave":         "Méabh",
    "medb":          "Méabh",
    "mave":          "Méabh",
    "eamon":         "Eamon",
    "aymon":         "Eamon",
    "aimon":         "Eamon",
    "shaymus":       "Seamus",
    "shamus":        "Seamus",
    "shaun":         "Sean",
    "shawn":         "Sean",
    "shan":          "Sean",
    # O' prefix variants (Malayali speakers often drop the O' or merge it)
    "reilly":        "O'Reilly",
    "oreilly":       "O'Reilly",
    "brien":         "O'Brien",
    "obrien":        "O'Brien",
    "connor":        "O'Connor",
    "oconnor":       "O'Connor",
    "sullivan":      "O'Sullivan",
    "osullivan":     "O'Sullivan",
    "neill":         "O'Neill",
    "oneill":        "O'Neill",
    "donnell":       "O'Donnell",
    "odonnell":      "O'Donnell",
    "mahony":        "O'Mahony",
    "omahony":       "O'Mahony",
    "keeffe":        "O'Keeffe",
    "okeeffe":       "O'Keeffe",
    "callaghan":     "O'Callaghan",
    "ocallaghan":    "O'Callaghan",
    "donoghue":      "O'Donoghue",
    "odonoghue":     "O'Donoghue",
    "doherty":       "O'Doherty",
    "odoherty":      "O'Doherty",
    "riordan":       "O'Riordan",
    "oriordan":      "O'Riordan",
    "carroll":       "O'Carroll",
    "ocarroll":      "O'Carroll",
    # Mc / Mac prefix variants
    "macarthy":      "McCarthy",
    "mccarthy":      "McCarthy",
    "mcloughlin":    "McLoughlin",
    "mclafflin":     "McLoughlin",
    "mcgrath":       "McGrath",
    "makgrath":      "McGrath",
    "mcdonagh":      "McDonagh",
    "makdonagh":     "McDonagh",
}


# ─────────────────────────────────────────────────────────────────────────────
#  Pre-built curated full name combinations
#  These represent realistic Irish care-home resident names.
# ─────────────────────────────────────────────────────────────────────────────

_CURATED_FULL_NAMES: List[str] = [
    # Classic combinations common in Irish care homes
    "Mary Murphy",       "Kathleen Kelly",    "Margaret Walsh",
    "Brigid O'Sullivan", "Joan Ryan",         "Ann Byrne",
    "Patricia O'Brien",  "Helen Collins",     "Eileen Doyle",
    "Nóra Lynch",        "Anne Murray",       "Máire Quinn",
    "Sinéad Moore",      "Siobhán Gallagher", "Niamh Kennedy",
    "Aoife Farrell",     "Aisling Nolan",     "Caoimhe Burke",
    "Roisin Hayes",      "Clodagh Campbell",  "Saoirse Griffin",
    "Orla Sweeney",      "Ciara Brennan",     "Fiona Fitzgerald",
    "Deirdre McCarthy",  "Gráinne McLoughlin","Nuala O'Carroll",
    "Eimear Cullen",     "Maeve Power",       "Sorcha Sheridan",
    "John Murphy",       "Patrick Kelly",     "Michael Walsh",
    "James O'Sullivan",  "Thomas Ryan",       "William Byrne",
    "David O'Brien",     "Stephen Collins",   "Kevin Doyle",
    "Brian Lynch",       "Paul Murray",       "Mark Quinn",
    "Conor Moore",       "Seamus Gallagher",  "Ciarán Kennedy",
    "Liam Farrell",      "Eoin Nolan",        "Cormac Burke",
    "Declan Hayes",      "Ronan Campbell",    "Darragh Griffin",
    "Barry Sweeney",     "Shane Brennan",     "Gary Fitzgerald",
    "Niall McCarthy",    "Oisín McLoughlin",  "Pádraig O'Carroll",
    "Tadhg Cullen",      "Ruairí Power",      "Fionn Sheridan",
    "Daniel Ryan",       "John Doyle",        "Marcus O'Reilly",
    "Anne Kelly",        "Emma Thompson",     "Mary Collins",
    "Sarah Johnson",     "Lisa Chen",         "Peter O'Brien",
    "Catherine Walsh",   "Elizabeth Murphy",  "Frances Byrne",
    "Gerard Connolly",   "Hugh Daly",         "Ivan Donnelly",
    "Joseph Flynn",      "Kenneth Foley",     "Lawrence Hughes",
    "Martin Johnston",   "Noel Keane",        "Oliver Kearney",
    "Raymond Kinsella",  "Samuel Lawlor",     "Victor Lennon",
    "Walter Malone",     "Arthur McGrath",    "Bernard Moriarty",
    "Charles Moran",     "Dennis Maher",      "Edward Riordan",
    "Francis White",     "George Wilson",     "Harold Bourke",
    "Irene Broderick",   "Jacqueline Delaney","Katharine Fanning",
    "Lorraine Hogan",    "Monica Meagher",    "Nancy Moloney",
    "Olive Mulcahy",     "Pamela Nagle",      "Queenie Fitzgibbon",
    "Rosemary Loftus",   "Teresa Fitzpatrick","Ursula O'Gorman",
    "Veronica O'Herlihy","Winifred O'Keeffe", "Yvonne O'Mahony",
]


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_all_irish_names() -> List[str]:
    """
    Return a flat list of realistic Irish full-name combinations.

    Used by Layer 01 to supplement the live patient registry.
    Names here augment (never override) registry entries.

    Returns
    -------
    List[str]
        Unique full names (First Surname format).
    """
    return list(dict.fromkeys(_CURATED_FULL_NAMES))  # dedup, preserve order


def get_irish_first_names() -> List[str]:
    """All Irish first names (male + female), deduplicated."""
    return list(dict.fromkeys(IRISH_MALE_FIRST + IRISH_FEMALE_FIRST))


def get_irish_surnames() -> List[str]:
    """All Irish surnames."""
    return list(IRISH_SURNAMES)


def get_phonetic_variants() -> Dict[str, str]:
    """
    Return the phonetic variant map.

    Keys   : lowercase phonetic spellings (as a Malayali speaker might say/
             transcribe an Irish name).
    Values : canonical Irish spelling.

    Intended to be merged into shared.py's mlin_normalise_name() accent
    rules or used directly in the Layer 01 L2 (accent_norm) pass.
    """
    return dict(IRISH_PHONETIC_VARIANTS)


def apply_phonetic_variant(name: str) -> str:
    """
    Apply Irish phonetic variant normalisation to a single name token.

    Checks each whitespace-separated token against the variant map and
    replaces any matched tokens.  Case-insensitive lookup, output preserves
    the canonical casing from the map.

    Example
    -------
    >>> apply_phonetic_variant("shivawn obrien")
    "Siobhán O'Brien"
    """
    tokens = name.strip().split()
    out = []
    i = 0
    while i < len(tokens):
        # Try 2-token match first ("o brien" → "O'Brien")
        if i + 1 < len(tokens):
            two = (tokens[i] + " " + tokens[i + 1]).lower()
            if two in IRISH_PHONETIC_VARIANTS:
                out.append(IRISH_PHONETIC_VARIANTS[two])
                i += 2
                continue
        one = tokens[i].lower()
        out.append(IRISH_PHONETIC_VARIANTS.get(one, tokens[i]))
        i += 1
    return " ".join(out)
