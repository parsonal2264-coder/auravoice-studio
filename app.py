from __future__ import annotations

import base64
import io
import logging
import os
import re
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import requests
from flask import Flask, jsonify, render_template, request, send_file
from flask_cors import CORS

ROOT = Path(__file__).resolve().parent
MODEL_ID = "eleven_v3"
PORT = 7860
TIMEOUT = 120
MAX_SCRIPT_CHARS = 5000

app = Flask(__name__, template_folder=str(ROOT / "templates"))
CORS(app, resources={r"/api/*": {"origins": os.environ.get("FRONTEND_ORIGIN", "*")}})
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
log = logging.getLogger("auravoice")
lock = threading.Lock()
state: dict[str, Any] = {"active": 0, "usage": [], "renders": 0, "errors": 0}
voice_cache: dict[str, Any] = {"voices": [], "expires": 0.0}

VOICE_SLOTS = [
    ("Brian", "Deep, resonant and comforting", "VOICE_BRIAN_ID"),
    ("Bill", "Adult professional", "VOICE_BILL_ID"),
    ("Bill Oxley", "Documentary commentator", "VOICE_BILL_OXLEY_ID"),
    ("Bunty Reel", "Punchy, crisp and reel king", "VOICE_BUNTY_REEL_ID"),
    ("Bunty Funny", "Funny best friend", "VOICE_BUNTY_FUNNY_ID"),
    ("Ellen", "Serious, direct and confident", "VOICE_ELLEN_ID"),
    ("Amy", "Natural and sweet", "VOICE_AMY_ID"),
]

def load_keys() -> list[str]:
    values: list[str] = []
    for index in range(1, 101):
        value = (
            os.environ.get(f"ELEVENLABS_API_KEY_{index}", "").strip()
            or os.environ.get(f"ELEVENLABS_KEY_{index}", "").strip()
            or os.environ.get(f"ELAVENLABS_API_KEY_{index}", "").strip()
        )
        if value:
            values.append(value)
    if values:
        return values
    return [
        x.strip() for x in os.environ.get("ELEVENLABS_API_KEYS", "").split(",") if x.strip()
    ]

def get_keys() -> list[str]:
    values = load_keys()
    with lock:
        if len(state["usage"]) != len(values):
            state["usage"] = [0] * len(values)
    return values

def provider_error(response: requests.Response) -> str:
    try:
        payload = response.json()
        detail = payload.get("detail", payload)
        return detail if isinstance(detail, str) else str(detail)
    except ValueError:
        return response.text[:300] or "ElevenLabs request failed."

def elevenlabs(
    method: str,
    url: str,
    body: dict[str, Any] | None = None,
    audio: bool = False,
):
    available = get_keys()
    if not available:
        raise RuntimeError(
            "No API key found. Add ELEVENLABS_API_KEY_1 in Space Secrets."
        )
    last_error = "ElevenLabs request failed."
    for attempt in range(len(available)):
        with lock:
            key_index = (state["active"] + attempt) % len(available)
        try:
            response = requests.request(
                method,
                url,
                headers={
                    "xi-api-key": available[key_index],
                    "accept": "audio/mpeg" if audio else "application/json",
                    "content-type": "application/json",
                },
                json=body,
                timeout=TIMEOUT,
            )
            if response.ok:
                with lock:
                    state["active"] = key_index
                return (response.content if audio else response.json()), key_index
            last_error = provider_error(response)
            if response.status_code not in {401, 402, 429, 500, 502, 503, 504}:
                break
        except requests.RequestException as exc:
            last_error = str(exc)
    with lock:
        state["errors"] += 1
    raise RuntimeError(last_error)

def load_account_voices(force: bool = False) -> list[dict[str, Any]]:
    now = time.time()
    if not force and voice_cache["voices"] and voice_cache["expires"] > now:
        return voice_cache["voices"]
    for endpoint in (
        "https://api.elevenlabs.io/v2/voices",
        "https://api.elevenlabs.io/v1/voices?show_legacy=true",
    ):
        try:
            payload, _ = elevenlabs("GET", endpoint)
            voices = payload.get("voices", []) if isinstance(payload, dict) else []
            if voices:
                voice_cache.update({"voices": voices, "expires": now + 300})
                return voices
        except Exception:
            pass
    raise RuntimeError("Voice loading failed.")

def combined_catalog() -> list[dict[str, Any]]:
    account = load_account_voices()
    known = {voice.get("voice_id") for voice in account}
    result = [
        {
            "voice_id": v.get("voice_id"),
            "name": v.get("name", "Unnamed"),
            "category": v.get("category", "account"),
        }
        for v in account
    ]
    for name, description, secret_name in VOICE_SLOTS:
        voice_id = os.environ.get(secret_name, "").strip()
        if voice_id and voice_id not in known:
            result.append(
                {"voice_id": voice_id, "name": name, "category": description}
            )
    return result

def apply_pronunciation(text: str, rules: str) -> str:
    for line in str(rules or "").splitlines():
        if "=" not in line:
            continue
        word, spoken = line.split("=", 1)
        word, spoken = word.strip(), spoken.strip()
        if word and spoken:
            text = re.sub(
                rf"\b{re.escape(word)}\b", spoken, text, flags=re.IGNORECASE
            )
    return text

def synthesis_text(script: str, tags: list[str], rules: str, speed: float) -> str:
    clean_tags = [str(tag).strip() for tag in (tags or []) if str(tag).strip()]
    prefix = (
        f"Delivery directions: {', '.join(clean_tags)}. "
        "Apply naturally and strictly.\n\n"
        if clean_tags
        else ""
    )
    speed_value = float(speed or 1.0)
    pace = (
        "Read at a measured pace.\n\n"
        if speed_value < 0.9
        else "Read with brisk pacing.\n\n"
        if speed_value > 1.1
        else ""
    )
    return prefix + pace + apply_pronunciation(script.strip(), rules)

def synthesize(
    voice_id: str,
    text: str,
    stability: float = 0.8,
    clarity: float = 0.6,
):
    body = {
        "text": text,
        "model_id": MODEL_ID,
        "voice_settings": {
            "stability": max(0.0, min(1.0, float(stability))),
            "similarity_boost": max(0.0, min(1.0, float(clarity))),
            "use_speaker_boost": True,
        },
    }
    audio, key_index = elevenlabs(
        "POST",
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
        body,
        True,
    )
    with lock:
        state["usage"][key_index] += len(text)
        state["renders"] += 1
    return audio, key_index

@app.get("/")
def home():
    return render_template("index.html")

@app.get("/api/health")
def health():
    return jsonify(
        {
            "ok": True,
            "model": MODEL_ID,
            "configured_keys": len(get_keys()),
            "active_key": f"Key {state['active'] + 1:02d}",
            "renders": state["renders"],
            "errors": state["errors"],
        }
    )

@app.get("/api/voices")
def voices():
    try:
        return jsonify({"voices": combined_catalog()})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 503

@app.get("/api/credits")
def credits():
    return jsonify(
        {
            "status": "Credit tracking enabled",
            "total_credits": "View Hugging Face logs",
        }
    )

@app.post("/api/preview")
def preview():
    payload = request.get_json(silent=True) or {}
    voice_id = str(payload.get("voice_id", "")).strip()
    if not voice_id:
        return jsonify({"error": "Select a voice."}), 400
    try:
        audio, _ = synthesize(
            voice_id, "This is a short AuraVoice Studio preview."
        )
        return jsonify(
            {
                "audio_base64": base64.b64encode(audio).decode("ascii"),
                "mime_type": "audio/mpeg",
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 503

@app.post("/api/generate")
def generate():
    payload = request.get_json(silent=True) or {}
    voice_cache.clear()
    script = str(payload.get("script", "")).strip()
    voice_id = str(payload.get("voice_id", "")).strip()
    
    if not script:
        return jsonify({"error": "Script is required."}), 400
    if not voice_id:
        return jsonify({"error": "Voice ID is required."}), 400
    
    try:
        text = synthesis_text(
            script,
            payload.get("tags", []),
            payload.get("pronunciation", ""),
            payload.get("speed", 1.0),
        )
        audio, key_index = synthesize(
            voice_id,
            text,
            payload.get("stability", 0.8),
            payload.get("similarity_boost", 0.6),
        )
        return jsonify(
            {
                "audio_base64": base64.b64encode(audio).decode("ascii"),
                "mime_type": "audio/mpeg",
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 503

def convert_audio(mp3: bytes, output_format: str) -> io.BytesIO:
    options = {
        "wav": ["-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le", "-f", "wav"],
        "ogg": ["-c:a", "libvorbis", "-q:a", "5", "-f", "ogg"],
        "flac": ["-c:a", "flac", "-f", "flac"],
        "m4a": ["-c:a", "aac", "-b:a", "192k", "-f", "ipod"],
        "webm": ["-c:a", "libopus", "-b:a", "128k", "-f", "webm"],
    }
    args = options.get(output_format)
    if not args:
        args = ["-c:a", "libmp3lame", "-f", "mp3"]
        
    with tempfile.NamedTemporaryFile(suffix=".mp3") as source:
        source.write(mp3)
        source.flush()
        process = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                source.name,
                *args,
                "pipe:1",
            ],
            capture_output=True,
            check=False,
        )
    return io.BytesIO(process.stdout)

@app.post("/api/export")
def export_audio():
    payload = request.get_json(silent=True) or {}
    output_format = str(payload.get("format", "mp3")).lower()
    raw = base64.b64decode(str(payload.get("audio_base64", "")))
    output = (
        io.BytesIO(raw) if output_format == "mp3" else convert_audio(raw, output_format)
    )
    output.seek(0)
    mime = {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "ogg": "audio/ogg",
        "flac": "audio/flac",
        "m4a": "audio/mp4",
        "webm": "audio/webm",
    }.get(output_format, "audio/mpeg")
    return send_file(
        output,
        mimetype=mime,
        as_attachment=True,
        download_name=f"auravoice.{output_format}",
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
