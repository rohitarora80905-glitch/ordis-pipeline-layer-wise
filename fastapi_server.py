"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ORDIS — fastapi_server.py                                                   ║
║  FastAPI wrapper — one endpoint per layer                                    ║
║                                                                              ║
║  Endpoints:                                                                  ║
║    POST  /api/layer01   (txt, accent, Ordis_ID) → name correction            ║
║    POST  /api/layer02   (txt, accent, Ordis_ID) → PII redaction              ║
║    POST  /api/layer03a  (txt, accent, Ordis_ID) → voice mis → standard       ║
║    POST  /api/layer03b  (txt, accent, Ordis_ID) → standard → professional    ║
║    POST  /api/layer04   (txt, accent, Ordis_ID) → PII reversal               ║
║    POST  /api/pipeline  (txt, accent, Ordis_ID) → run all 5 layers in order  ║
║    GET   /health                                → service health check        ║
║                                                                              ║
║  Run:                                                                        ║
║    uvicorn fastapi_server:app --host 0.0.0.0 --port 8080 --reload            ║
║                                                                              ║
║  State:                                                                      ║
║    The PII map created by Layer 02 is persisted to                           ║
║    output/pii_map_{Ordis_ID}.json and read by Layer 04.                      ║
║    Use the SAME Ordis_ID across all layer calls for a single transcript.     ║
╚══════════════════════════════════════════════════════════════════════════════╝

Typical call sequence for one transcript
──────────────────────────────────────────
  1. POST /api/layer01  { txt: "<raw>", accent: "ml_In", Ordis_ID: "abc123" }
  2. POST /api/layer02  { txt: "<L1 output>", accent: "ml_In", Ordis_ID: "abc123" }
  3. POST /api/layer03a { txt: "<L2 output>", accent: "ml_In", Ordis_ID: "abc123" }
  4. POST /api/layer03b { txt: "<L3A output>", accent: "ml_In", Ordis_ID: "abc123" }
  5. POST /api/layer04  { txt: "<L3B output>", accent: "ml_In", Ordis_ID: "abc123" }

Or use /api/pipeline to run all steps in one call.
"""

from __future__ import annotations

import uuid
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import layer01
import layer02
import layer03a
import layer03b
import layer04
from shared import (
    ConfigManager,
    MedicalTermDatabase,
    ModelRouter,
    get_mongo_db,
    load_transcription_text,
)

# ── Startup — initialise shared resources once ───────────────────────────────
_cfg        = ConfigManager()
_router     = ModelRouter(_cfg)
_medical_db = MedicalTermDatabase(_cfg.get_data_path("medical_terms_csv"))
_mongo_db   = get_mongo_db(_cfg)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Ordis Clinical Transcription API",
    description = (
        "Irish Elderly Care — 5-layer voice transcription correction pipeline.\n\n"
        "Each layer is available as a separate endpoint, or use /api/pipeline "
        "to run all layers in sequence."
    ),
    version     = "1.0.0",
)


# ── Request / Response models ─────────────────────────────────────────────────

class LayerRequest(BaseModel):
    txt:              str  = Field("", description="Input text to process")
    transcription_id: Optional[str] = Field(
        None,
        description="Mongo transcription _id to load text directly from the database",
    )
    patient_id:       Optional[str] = Field(
        None,
        description="Mongo patientId to load all text for a patient from the database",
    )
    accent:           str  = Field("ml_In", description="Accent profile (default: ml_In)")
    Ordis_ID:         str  = Field(default="", description="Session ID — must be consistent across all layers for one transcript")

    class Config:
        json_schema_extra = {
            "example": {
                "txt":      "Resident Maacuus O'Rielly room 22 was observed Sundowing...",
                "accent":   "ml_In",
                "Ordis_ID": "session_20240407_001",
            }
        }


class LayerResponse(BaseModel):
    Ordis_ID:    str
    layer:       str
    output_txt:  str
    corrections: List[List[str]]   # list of [original, corrected] pairs
    message:     str


class PipelineResponse(BaseModel):
    Ordis_ID:    str
    layer01_out: str
    layer02_out: str
    layer03a_out: str
    layer03b_out: str
    final_out:   str
    all_corrections: Dict[str, List[List[str]]]
    message:     str


# ── Helper ────────────────────────────────────────────────────────────────────

def _ensure_ordis_id(ordis_id: str) -> str:
    """Generate a session ID if none was provided."""
    return ordis_id.strip() or str(uuid.uuid4())


def _resolve_text(req: LayerRequest) -> str:
    if req.txt.strip():
        return req.txt

    if _mongo_db is None:
        raise HTTPException(
            status_code=500,
            detail="MongoDB is not configured. Provide txt or configure mongodb.uri in config.yaml.",
        )

    try:
        return load_transcription_text(
            cfg=_cfg,
            mongo_db=_mongo_db,
            transcription_id=req.transcription_id,
            patient_id=req.patient_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Status"])
async def health():
    """Service health check — returns LLM provider info."""
    return {
        "status":       "ok",
        "llm_provider": _router.provider,
        "llm_model":    _router.model,
        "version":      app.version,
    }


@app.post("/api/layer01", response_model=LayerResponse, tags=["Layers"])
async def api_layer01(req: LayerRequest):
    """
    **Layer 01 — Name Correction**

    Fixes garbled nurse/patient names using fuzzy + phonetic matching
    against the patient and nurse registries.

    - Input: Raw voice-transcribed text
    - Output: Text with names corrected to canonical registry form
    """
    ordis_id = _ensure_ordis_id(req.Ordis_ID)
    text = _resolve_text(req)
    try:
        output, corrections = layer01.run(
            text=text, accent=req.accent, ordis_id=ordis_id, cfg=_cfg, mongo_db=_mongo_db
        )
        return LayerResponse(
            Ordis_ID    = ordis_id,
            layer       = "01 - Name Correction",
            output_txt  = output,
            corrections = [[b, g] for b, g in corrections],
            message     = f"{len(corrections)} name(s) corrected.",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Layer 01 error: {exc}")


@app.post("/api/layer02", response_model=LayerResponse, tags=["Layers"])
async def api_layer02(req: LayerRequest):
    """
    **Layer 02 — PII Redaction**

    Replaces all real patient/nurse names with anonymous tokens
    (PATIENT1, PATIENT2, NURSE1 …).

    Saves the token→name map to disk keyed by Ordis_ID so Layer 04
    can reverse the redaction later.

    - Input: Text from Layer 01 (corrected names)
    - Output: Text with names replaced by tokens
    """
    ordis_id = _ensure_ordis_id(req.Ordis_ID)
    text = _resolve_text(req)
    try:
        output, pii_map = layer02.run(
            text=text, accent=req.accent, ordis_id=ordis_id, cfg=_cfg, mongo_db=_mongo_db
        )
        corrections = [[token, name] for token, name in pii_map.items()]
        return LayerResponse(
            Ordis_ID    = ordis_id,
            layer       = "02 - PII Redaction",
            output_txt  = output,
            corrections = corrections,
            message     = f"{len(pii_map)} name(s) redacted. Use same Ordis_ID in Layer 04 to restore.",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Layer 02 error: {exc}")


@app.post("/api/layer03a", response_model=LayerResponse, tags=["Layers"])
async def api_layer03a(req: LayerRequest):
    """
    **Layer 03A — Voice Misinterpretation → Standard Term**

    LLM Call 1: uses the Column D → Column A lookup table to replace
    voice misinterpretations with correct standard clinical terms.

    - Input: PII-redacted text from Layer 02
    - Output: Text with misheard terms replaced by standard terms
    """
    ordis_id = _ensure_ordis_id(req.Ordis_ID)
    text = _resolve_text(req)
    try:
        output, corrections = layer03a.run(
            text=text, accent=req.accent, ordis_id=ordis_id,
            cfg=_cfg, router=_router, medical_db=_medical_db,
        )
        return LayerResponse(
            Ordis_ID    = ordis_id,
            layer       = "03A - Voice Mis → Standard Term",
            output_txt  = output,
            corrections = [[b, g] for b, g in corrections],
            message     = f"{len(corrections)} misinterpretation(s) corrected.",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Layer 03A error: {exc}")


@app.post("/api/layer03b", response_model=LayerResponse, tags=["Layers"])
async def api_layer03b(req: LayerRequest):
    """
    **Layer 03B — Standard Term → Professional Clinical Language**

    LLM Call 2: uses the Column A → Column B lookup table to upgrade
    terminology to professional clinical language, plus grammar and
    drug name correction.

    - Input: Text from Layer 03A (standard terms in place)
    - Output: Fully professional clinical note (names still tokenised)
    """
    ordis_id = _ensure_ordis_id(req.Ordis_ID)
    text = _resolve_text(req)
    try:
        output, corrections = layer03b.run(
            text=text, accent=req.accent, ordis_id=ordis_id,
            cfg=_cfg, router=_router, medical_db=_medical_db,
        )
        return LayerResponse(
            Ordis_ID    = ordis_id,
            layer       = "03B - Standard → Professional",
            output_txt  = output,
            corrections = [[b, g] for b, g in corrections],
            message     = f"{len(corrections)} professional term upgrade(s).",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Layer 03B error: {exc}")


@app.post("/api/layer04", response_model=LayerResponse, tags=["Layers"])
async def api_layer04(req: LayerRequest):
    """
    **Layer 04 — PII Reversal**

    Restores real names from PII tokens using the map saved by Layer 02.
    No LLM call — purely deterministic find-replace.

    Requires the SAME Ordis_ID used in Layer 02.

    - Input: Professional text from Layer 03B (tokens still in place)
    - Output: Final polished note with real names restored
    """
    ordis_id = _ensure_ordis_id(req.Ordis_ID)
    if not req.Ordis_ID.strip():
        raise HTTPException(
            status_code=400,
            detail="Ordis_ID is required for Layer 04 — must match the ID used in Layer 02.",
        )
    text = _resolve_text(req)
    try:
        output, reversals = layer04.run(
            text=text, accent=req.accent, ordis_id=ordis_id, cfg=_cfg
        )
        return LayerResponse(
            Ordis_ID    = ordis_id,
            layer       = "04 - PII Reversal",
            output_txt  = output,
            corrections = [[token, name] for token, name in reversals],
            message     = f"{len(reversals)} PII token(s) restored to real names.",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Layer 04 error: {exc}")


@app.post("/api/pipeline", response_model=PipelineResponse, tags=["Pipeline"])
async def api_pipeline(req: LayerRequest):
    """
    **Full Pipeline — All 5 Layers in Sequence**

    Runs Layer 01 → 02 → 03A → 03B → 04 in order and returns all
    intermediate outputs plus the final corrected note.

    Equivalent to calling each layer endpoint in order with the same Ordis_ID.

    - Input: Raw voice-transcribed clinical note
    - Output: Final professional note with real names, plus all intermediate steps
    """
    ordis_id = _ensure_ordis_id(req.Ordis_ID)
    all_corrections: Dict[str, List[List[str]]] = {}

    try:
        # L1 — Name correction
        text = _resolve_text(req)
        l1_out, l1_corr = layer01.run(
            text=text, accent=req.accent, ordis_id=ordis_id, cfg=_cfg, mongo_db=_mongo_db
        )
        all_corrections["layer01"] = [[b, g] for b, g in l1_corr]

        # L2 — PII redaction
        l2_out, pii_map = layer02.run(
            text=l1_out, accent=req.accent, ordis_id=ordis_id, cfg=_cfg, mongo_db=_mongo_db
        )
        all_corrections["layer02"] = [[t, n] for t, n in pii_map.items()]

        # L3A — Voice mis → standard
        l3a_out, l3a_corr = layer03a.run(
            text=l2_out, accent=req.accent, ordis_id=ordis_id,
            cfg=_cfg, router=_router, medical_db=_medical_db,
        )
        all_corrections["layer03a"] = [[b, g] for b, g in l3a_corr]

        # L3B — Standard → professional
        l3b_out, l3b_corr = layer03b.run(
            text=l3a_out, accent=req.accent, ordis_id=ordis_id,
            cfg=_cfg, router=_router, medical_db=_medical_db,
        )
        all_corrections["layer03b"] = [[b, g] for b, g in l3b_corr]

        # L4 — PII reversal
        final_out, l4_rev = layer04.run(
            text=l3b_out, accent=req.accent, ordis_id=ordis_id,
            cfg=_cfg, pii_map=pii_map,  # pass in-memory to avoid disk round-trip
        )
        all_corrections["layer04"] = [[t, n] for t, n in l4_rev]

        total = sum(len(v) for v in all_corrections.values())
        return PipelineResponse(
            Ordis_ID      = ordis_id,
            layer01_out   = l1_out,
            layer02_out   = l2_out,
            layer03a_out  = l3a_out,
            layer03b_out  = l3b_out,
            final_out     = final_out,
            all_corrections = all_corrections,
            message       = f"Pipeline complete. {total} total correction(s) across all layers.",
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")
