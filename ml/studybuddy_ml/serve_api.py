from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import LABELS
from .temporal import SessionSummaryAccumulator, TemporalConfig, TemporalFocusEngine

MODEL_PATH = Path(os.getenv("STUDYBUDDY_MODEL_PATH", "/app/artifacts/export/best_model.keras"))
IMAGE_SIZE_RAW = os.getenv("STUDYBUDDY_IMAGE_SIZE")
if IMAGE_SIZE_RAW and IMAGE_SIZE_RAW.strip():
    IMAGE_SIZE_OVERRIDE: int | None = int(IMAGE_SIZE_RAW)
else:
    IMAGE_SIZE_OVERRIDE = None
CORS_ORIGINS_RAW = os.getenv(
    "STUDYBUDDY_CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:5174,http://127.0.0.1:5174,http://localhost:5175,http://127.0.0.1:5175",
)
if CORS_ORIGINS_RAW.strip() == "*":
    CORS_ORIGINS = ["*"]
else:
    CORS_ORIGINS = [
        origin.strip()
        for origin in CORS_ORIGINS_RAW.split(",")
        if origin.strip()
    ]


@dataclass
class SessionContext:
    session_id: str
    created_at_ms: int
    metadata: dict[str, Any] = field(default_factory=dict)
    engine: TemporalFocusEngine = field(default_factory=lambda: TemporalFocusEngine(TemporalConfig()))
    accumulator: SessionSummaryAccumulator | None = None


class StartSessionRequest(BaseModel):
    session_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class StartSessionResponse(BaseModel):
    session_id: str
    created_at_ms: int


class PredictionPayload(BaseModel):
    session_id: str
    timestamp_ms: int
    raw_label: str
    raw_confidence: float
    smoothed_label: str
    smoothed_confidence: float
    state: str
    transitioned: bool


class StopSessionResponse(BaseModel):
    session_id: str
    summary: dict[str, Any]


app = FastAPI(title="StudyBuddy GPU Inference API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
_model: tf.keras.Model | None = None
_sessions: dict[str, SessionContext] = {}


def _load_model() -> tf.keras.Model:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file does not exist at {MODEL_PATH}. Run export-best first."
        )
    # Inference only needs forward pass; compile state may contain
    # custom training losses that are not required at serve time.
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def _ensure_model() -> tf.keras.Model:
    global _model
    if _model is None:
        _model = _load_model()
    return _model


def _infer_model_image_size(model: tf.keras.Model) -> int | None:
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        if not input_shape:
            return None
        input_shape = input_shape[0]
    if not isinstance(input_shape, tuple) or len(input_shape) < 3:
        return None
    height = input_shape[1]
    width = input_shape[2]
    if isinstance(height, int) and isinstance(width, int) and height > 0 and width > 0:
        # Current models are square; use height for resize target.
        return int(height)
    return None


def _effective_image_size(model: tf.keras.Model) -> int:
    if IMAGE_SIZE_OVERRIDE is not None and IMAGE_SIZE_OVERRIDE > 0:
        return IMAGE_SIZE_OVERRIDE
    inferred = _infer_model_image_size(model)
    if inferred is not None:
        return inferred
    return 224


def _decode_image(image_bytes: bytes, image_size: int) -> tf.Tensor:
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, [image_size, image_size])
    image = tf.cast(image, tf.float32)
    image = tf.expand_dims(image, axis=0)
    return image


def _predict_probs(image_bytes: bytes) -> np.ndarray:
    model = _ensure_model()
    image_size = _effective_image_size(model)
    image = _decode_image(image_bytes, image_size)
    probs = model(image, training=False).numpy()[0]
    return probs


def _session_prediction(session: SessionContext, image_bytes: bytes) -> PredictionPayload:
    timestamp_ms = int(time.time() * 1000)
    probs = _predict_probs(image_bytes)
    raw_idx = int(np.argmax(probs))
    raw_label = LABELS[raw_idx]
    raw_conf = float(probs[raw_idx])

    temporal = session.engine.update(probs=probs, timestamp_ms=timestamp_ms)
    if session.accumulator is None:
        session.accumulator = SessionSummaryAccumulator(start_ms=timestamp_ms)
    session.accumulator.update(temporal)

    return PredictionPayload(
        session_id=session.session_id,
        timestamp_ms=timestamp_ms,
        raw_label=raw_label,
        raw_confidence=round(raw_conf, 4),
        smoothed_label=temporal.smoothed_label,
        smoothed_confidence=round(temporal.smoothed_confidence, 4),
        state=temporal.state,
        transitioned=temporal.transitioned,
    )


def _get_session(session_id: str) -> SessionContext:
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Unknown session_id: {session_id}")
    return session


@app.on_event("startup")
async def startup_event() -> None:
    _ensure_model()


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"ok": True, "model_loaded": _model is not None, "labels": LABELS}


@app.get("/model-info")
async def model_info() -> dict[str, Any]:
    model = _ensure_model()
    image_size = _effective_image_size(model)
    return {
        "model_path": str(MODEL_PATH),
        "image_size": image_size,
        "num_classes": len(LABELS),
        "labels": LABELS,
        "input_shape": model.input_shape,
        "output_shape": model.output_shape,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file payload")
    probs = _predict_probs(data)
    idx = int(np.argmax(probs))
    return {
        "label": LABELS[idx],
        "confidence": float(round(float(probs[idx]), 4)),
        "probs": {label: float(round(float(prob), 4)) for label, prob in zip(LABELS, probs)},
    }


@app.post("/session/start", response_model=StartSessionResponse)
async def start_session(payload: StartSessionRequest) -> StartSessionResponse:
    session_id = payload.session_id or str(uuid.uuid4())
    now_ms = int(time.time() * 1000)
    _sessions[session_id] = SessionContext(
        session_id=session_id,
        created_at_ms=now_ms,
        metadata=payload.metadata,
    )
    return StartSessionResponse(session_id=session_id, created_at_ms=now_ms)


@app.post("/session/{session_id}/frame", response_model=PredictionPayload)
async def ingest_frame(session_id: str, file: UploadFile = File(...)) -> PredictionPayload:
    session = _get_session(session_id)
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file payload")
    return _session_prediction(session, data)


@app.post("/session/{session_id}/stop", response_model=StopSessionResponse)
async def stop_session(session_id: str) -> StopSessionResponse:
    session = _get_session(session_id)
    _sessions.pop(session_id, None)
    end_ms = int(time.time() * 1000)
    if session.accumulator is None:
        summary = {
            "startTs": int(session.created_at_ms / 1000),
            "endTs": int(end_ms / 1000),
            "focusedMs": 0,
            "distractedMs": 0,
            "longestFocusedMs": 0,
            "longestDistractedMs": 0,
            "distractions": 0,
            "avgFocusBeforeDistractMs": 0.0,
            "focusPercent": 0.0,
            "attentionLabelCounts": {},
        }
    else:
        summary = session.accumulator.finalize(end_ms=end_ms)
    return StopSessionResponse(session_id=session_id, summary=summary)


@app.websocket("/ws/session/{session_id}")
async def ws_session(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    session = _sessions.get(session_id)
    if session is None:
        now_ms = int(time.time() * 1000)
        session = SessionContext(session_id=session_id, created_at_ms=now_ms)
        _sessions[session_id] = session

    try:
        while True:
            message = await websocket.receive()
            if "bytes" in message and message["bytes"] is not None:
                payload = _session_prediction(session, message["bytes"])
                await websocket.send_text(payload.model_dump_json())
            elif "text" in message and message["text"] is not None:
                text = message["text"].strip()
                if text.lower() == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
                elif text.lower() == "stop":
                    summary_payload = (await stop_session(session_id)).model_dump()
                    await websocket.send_text(json.dumps({"type": "stopped", **summary_payload}))
                    break
                else:
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "error",
                                "message": "Expected binary JPEG frames, 'ping', or 'stop'.",
                            }
                        )
                    )
    except WebSocketDisconnect:
        return
