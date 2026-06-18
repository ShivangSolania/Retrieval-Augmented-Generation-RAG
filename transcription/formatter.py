"""
Transcript formatting for RAG-ready Markdown.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any

from openai import OpenAI

from config import config

SEGMENTS_PER_LLM_CHUNK = 200
GAP_SECONDS = 2.0

_STRUCTURE_SYSTEM = """\
You convert speech-to-text segment data into clean Markdown for RAG ingestion.

Rules:
- Output ONLY valid Markdown (no code fences, no preamble).
- Start with a single # title (humanize the suggested title).
- Then a metadata block using bold labels on separate lines:
  **Source:** `filename`
  **Duration:** HH:MM:SS
  **Transcribed:** YYYY-MM-DD
  **Speakers:** comma-separated speaker labels (omit if none are available)
- Use ## headings for semantic sections. Each heading MUST end with a timestamp \
in brackets from the first segment in that section, e.g. ## Budget review [00:12:34].
- If speaker labels are provided in the segments, attribute speech with **speaker_label:** \
prefix on each paragraph or turn. If not, output paragraphs normally without labels.
- Group sections by topic. Preserve factual content; \
do not invent information not present in the segments.
- Timestamps in headings must be HH:MM:SS (hours optional if under 1 hour: MM:SS is ok).
- Do not duplicate the metadata block inside section bodies.
"""

_STRUCTURE_USER = """\
Suggested title: {title}
Source audio: {source_name}
Total duration: {duration}
Transcription date: {transcribed_date}
Speakers: {speakers}
{chunk_note}

Segments (JSON array of {{index, start, end, text, speaker}}):
{segments_json}

Produce structured Markdown with semantic ## sections, speaker attribution, \
and accurate timestamps.
"""


def format_timestamp(seconds: float) -> str:
    """Format *seconds* as HH:MM:SS or MM:SS."""
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def humanize_filename(stem: str) -> str:
    """Turn a file stem into a readable title."""
    text = stem.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text.title() if text else "Transcript"


def segments_from_transcription_response(
    data: dict[str, Any],
) -> list[dict[str, Any]]:
    """Normalize diarized_json (or legacy verbose_json) segments."""
    raw = data.get("segments") or []
    out: list[dict[str, Any]] = []
    for i, seg in enumerate(raw):
        if not isinstance(seg, dict):
            continue
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        entry: dict[str, Any] = {
            "index": i,
            "start": float(seg.get("start", 0)),
            "end": float(seg.get("end", 0)),
            "text": text,
        }
        speaker = seg.get("speaker")
        if speaker is not None:
            entry["speaker"] = str(speaker)
        out.append(entry)
    return out


def unique_speakers(segments: list[dict[str, Any]]) -> list[str]:
    """Return ordered unique speaker labels from segments."""
    seen: set[str] = set()
    speakers: list[str] = []
    for seg in segments:
        speaker = seg.get("speaker")
        if speaker and speaker not in seen:
            seen.add(speaker)
            speakers.append(speaker)
    return speakers


def compact_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip segments to fields needed for LLM input."""
    out: list[dict[str, Any]] = []
    for s in segments:
        entry: dict[str, Any] = {
            "index": s["index"],
            "start": round(s["start"], 2),
            "end": round(s["end"], 2),
            "text": s["text"],
        }
        if s.get("speaker"):
            entry["speaker"] = s["speaker"]
        out.append(entry)
    return out


def total_duration_seconds(segments: list[dict[str, Any]]) -> float:
    """Last segment end time, or 0."""
    if not segments:
        return 0.0
    return max(float(s.get("end", 0)) for s in segments)


def metadata_block(
    source_name: str,
    duration_sec: float,
    segments: list[dict[str, Any]] | None = None,
    transcribed: date | None = None,
) -> str:
    """Build the standard metadata lines (without H1)."""
    when = (transcribed or date.today()).isoformat()
    lines = [
        f"**Source:** `{source_name}`  ",
        f"**Duration:** {format_timestamp(duration_sec)}  ",
        f"**Transcribed:** {when}",
    ]
    if segments:
        speakers = unique_speakers(segments)
        if speakers:
            lines.append(f"**Speakers:** {', '.join(speakers)}")
    return "\n".join(lines) + "\n"


def format_unstructured(
    audio_path: Path,
    segments: list[dict[str, Any]],
    transcribed: date | None = None,
) -> str:
    """Group segments by pause gaps into timestamped ## blocks (no LLM)."""
    title = humanize_filename(audio_path.stem)
    duration = total_duration_seconds(segments)
    lines = [
        f"# {title}\n",
        metadata_block(audio_path.name, duration, segments, transcribed),
    ]

    if not segments:
        lines.append("\n_(No speech detected.)_\n")
        return "\n".join(lines)

    groups: list[list[dict[str, Any]]] = [[segments[0]]]
    for seg in segments[1:]:
        prev = groups[-1][-1]
        gap = float(seg["start"]) - float(prev["end"])
        if gap > GAP_SECONDS:
            groups.append([seg])
        else:
            groups[-1].append(seg)

    for n, group in enumerate(groups, start=1):
        ts = format_timestamp(float(group[0]["start"]))
        speaker = group[0].get("speaker", "speaker")
        heading = f"## {speaker} — Segment {n} [{ts}]"
        body_parts: list[str] = []
        for seg in group:
            label = seg.get("speaker", speaker)
            body_parts.append(f"**{label}:** {seg['text']}")
        lines.append(f"\n{heading}\n\n" + "\n\n".join(body_parts) + "\n")

    return "\n".join(lines)


def _structure_chunk(
    client: OpenAI,
    *,
    title: str,
    source_name: str,
    duration: str,
    transcribed_date: str,
    speakers: str,
    segments: list[dict[str, Any]],
    chunk_note: str,
) -> str:
    """Call LLM to structure one segment batch into Markdown body sections."""
    user = _STRUCTURE_USER.format(
        title=title,
        source_name=source_name,
        duration=duration,
        transcribed_date=transcribed_date,
        speakers=speakers,
        chunk_note=chunk_note,
        segments_json=json.dumps(compact_segments(segments), ensure_ascii=False),
    )
    response = client.chat.completions.create(
        model=config.llm_model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": _STRUCTURE_SYSTEM},
            {"role": "user", "content": user},
        ],
    )
    content = response.choices[0].message.content or ""
    return content.strip()


def _merge_structured_parts(header: str, parts: list[str]) -> str:
    """Combine multi-chunk LLM outputs under one document header."""
    if len(parts) == 1:
        text = parts[0]
        if text.startswith("#"):
            return text
        return f"{header}\n{text}"

    bodies: list[str] = []
    for part in parts:
        lines = part.splitlines()
        start = 0
        if lines and lines[0].startswith("#"):
            start = 1
            while start < len(lines) and (
                not lines[start].strip() or lines[start].startswith("**")
            ):
                start += 1
        bodies.append("\n".join(lines[start:]).strip())

    return f"{header}\n" + "\n\n".join(b for b in bodies if b)


def format_structured(
    client: OpenAI,
    audio_path: Path,
    segments: list[dict[str, Any]],
    transcribed: date | None = None,
) -> str:
    """Use LLM to produce semantic Markdown sections from diarized segments."""
    title = humanize_filename(audio_path.stem)
    duration_sec = total_duration_seconds(segments)
    duration_str = format_timestamp(duration_sec)
    transcribed_date = (transcribed or date.today()).isoformat()
    speakers_str = ", ".join(unique_speakers(segments)) or "unknown"

    if not segments:
        return (
            f"# {title}\n\n"
            f"{metadata_block(audio_path.name, duration_sec, segments, transcribed)}\n"
            "_(No speech detected.)_\n"
        )

    compact = compact_segments(segments)
    chunks = [
        compact[i : i + SEGMENTS_PER_LLM_CHUNK]
        for i in range(0, len(compact), SEGMENTS_PER_LLM_CHUNK)
    ]

    parts: list[str] = []
    for idx, chunk in enumerate(chunks):
        if len(chunks) == 1:
            note = ""
        else:
            note = (
                f"This is part {idx + 1} of {len(chunks)} of a longer recording. "
                "Continue semantic sections; do not repeat the document title or metadata block."
            )
        parts.append(
            _structure_chunk(
                client,
                title=title,
                source_name=audio_path.name,
                duration=duration_str,
                transcribed_date=transcribed_date,
                speakers=speakers_str,
                segments=chunk,
                chunk_note=note,
            )
        )

    if len(parts) == 1:
        return parts[0]

    header = (
        f"# {title}\n\n"
        f"{metadata_block(audio_path.name, duration_sec, segments, transcribed)}"
    )
    return _merge_structured_parts(header, parts)
