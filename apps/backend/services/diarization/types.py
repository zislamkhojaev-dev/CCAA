"""Shared types for diarization providers (ASR segments → speaker turns)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TranscriptSegment:
    """One timed fragment from ASR (Whisper segment, Deepgram word group, etc.)."""

    start_sec: float
    end_sec: float
    text: str
