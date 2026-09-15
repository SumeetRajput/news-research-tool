"""
audio_io.py
-----------
Voice in, voice out. Both run on Groq, so the whole application uses exactly
one AI provider.

* Speech to text  -> Whisper Large v3 Turbo (free tier: 2,000 requests/day)
* Text to speech  -> Orpheus v1 English

Both endpoints are OpenAI-compatible, which is why the `groq` SDK calls look
identical to OpenAI's. Groq runs them on their LPU hardware, so transcription
of a 10-second clip comes back in well under a second.
"""

from __future__ import annotations

import os
import re

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

STT_MODEL = "whisper-large-v3-turbo"   # fastest; use whisper-large-v3 for max accuracy
TTS_MODEL = "canopylabs/orpheus-v1-english"
TTS_VOICES = ["troy", "hannah"]

MAX_AUDIO_BYTES = 25 * 1024 * 1024     # Groq's hard limit on upload size
MAX_TTS_CHARS = 1800                   # keep synthesis snappy


class AudioError(RuntimeError):
    """Raised when transcription or synthesis fails."""


def _client() -> Groq:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise AudioError("GROQ_API_KEY is not set in .env")
    return Groq(api_key=key)


def transcribe(audio_bytes: bytes, filename: str = "input.wav", language: str = "en") -> str:
    """Turn recorded microphone audio into text.

    `audio_bytes` comes straight from Streamlit's st.audio_input() widget.
    """
    if not audio_bytes:
        raise AudioError("No audio was recorded.")
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise AudioError("Recording is over Groq's 25 MB limit. Keep it shorter.")

    try:
        result = _client().audio.transcriptions.create(
            file=(filename, audio_bytes),
            model=STT_MODEL,
            language=language,
            response_format="json",
            temperature=0.0,
        )
    except Exception as exc:  # noqa: BLE001 - surface the vendor error
        raise AudioError(f"Transcription failed: {exc}") from exc

    return (getattr(result, "text", "") or "").strip()


def _strip_for_speech(text: str) -> str:
    """Clean up an answer so it sounds right when read aloud.

    Citation markers, markdown bullets and bold asterisks all read badly. The
    listener has the written answer on screen, so we drop them rather than
    having the voice say "open bracket one close bracket".
    """
    text = re.sub(r"\[\d+\]", "", text)          # citation markers
    text = re.sub(r"[*_#`]+", "", text)          # markdown emphasis
    text = re.sub(r"^\s*[-•]\s*", "", text, flags=re.M)
    text = re.sub(r"\n{2,}", ". ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def synthesize(text: str, voice: str = "troy") -> bytes:
    """Read an answer aloud. Returns WAV bytes ready for st.audio()."""
    clean = _strip_for_speech(text)
    if not clean:
        raise AudioError("Nothing to read aloud.")
    if len(clean) > MAX_TTS_CHARS:
        # Cut at a sentence boundary so it doesn't stop mid-word.
        cut = clean[:MAX_TTS_CHARS]
        clean = cut[: cut.rfind(". ") + 1] or cut

    try:
        response = _client().audio.speech.create(
            model=TTS_MODEL,
            voice=voice,
            input=clean,
            response_format="wav",
        )
        return response.read()
    except Exception as exc:  # noqa: BLE001
        raise AudioError(f"Speech synthesis failed: {exc}") from exc
