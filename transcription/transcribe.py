"""
Transcribe audio files to Markdown via OpenAI speech-to-text API.

Uses gpt-4o-transcribe-diarize by default for speaker-labeled, timestamped segments.
Writes ``.md`` files to the project root (or ``--output-dir``) for later RAG ingestion.

Usage (from the InternShip parent directory)::

    python llamaindex_rag/transcription/transcribe.py path/to/audio.mp3
    python llamaindex_rag/transcription/transcribe.py path/to/audio_folder/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _PACKAGE_ROOT.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv

_ENV_PATH = _PACKAGE_ROOT / ".env"
load_dotenv(_ENV_PATH)

from openai import OpenAI

from config import config

from transcription.formatter import (
    format_structured,
    format_unstructured,
    segments_from_transcription_response,
)

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".webm", ".mp4", ".mpeg", ".mpga"}
MAX_FILE_BYTES = 25 * 1024 * 1024


def _require_api_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print(
            "Error: OPENAI_API_KEY is not set. Add it to "
            f"{_ENV_PATH} or your environment.",
            file=sys.stderr,
        )
        sys.exit(1)


def discover_audio_paths(paths: list[Path]) -> list[Path]:
    """Expand files and directories into a sorted list of audio paths."""
    found: list[Path] = []
    for p in paths:
        if not p.exists():
            print(f"[skip] Not found: {p}", file=sys.stderr)
            continue
        if p.is_file():
            if p.suffix.lower() in AUDIO_EXTENSIONS:
                found.append(p.resolve())
            else:
                print(f"[skip] Unsupported extension: {p}", file=sys.stderr)
        elif p.is_dir():
            for child in sorted(p.rglob("*")):
                if child.is_file() and child.suffix.lower() in AUDIO_EXTENSIONS:
                    found.append(child.resolve())
    seen: set[Path] = set()
    unique: list[Path] = []
    for f in found:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    return unique


def transcribe_file(
    client: OpenAI,
    audio_path: Path,
    *,
    model: str,
    language: str | None = None,
) -> dict:
    """Run transcription API on *audio_path*; returns dict of segments."""
    kwargs: dict = {"model": model}
    
    if "whisper" in model:
        kwargs["response_format"] = "verbose_json"
    else:
        kwargs["response_format"] = "diarized_json"
        kwargs["chunking_strategy"] = "auto"
    if language:
        kwargs["language"] = language

    with audio_path.open("rb") as f:
        result = client.audio.transcriptions.create(file=f, **kwargs)

    if hasattr(result, "model_dump"):
        return result.model_dump()
    if isinstance(result, dict):
        return result
    if isinstance(result, str):
        return {"text": result, "segments": []}
    return json.loads(str(result))


def process_file(
    client: OpenAI,
    audio_path: Path,
    output_dir: Path,
    *,
    model: str,
    language: str | None = None,
    use_structure: bool = True,
    force: bool = False,
) -> Path | None:
    """Transcribe one audio file and write Markdown. Returns output path or None."""
    size = audio_path.stat().st_size
    if size > MAX_FILE_BYTES:
        mb = size / (1024 * 1024)
        print(
            f"[skip] {audio_path.name}: {mb:.1f} MB exceeds 25 MB API limit.",
            file=sys.stderr,
        )
        return None

    out_path = output_dir / f"{audio_path.stem}.md"
    if out_path.exists() and not force:
        print(f"[skip] {out_path.name} exists (use --force to overwrite).")
        return None

    print(f"[>] Transcribing ({model}): {audio_path}")
    data = transcribe_file(client, audio_path, model=model, language=language)
    segments = segments_from_transcription_response(data)

    if use_structure:
        markdown = format_structured(client, audio_path, segments)
    else:
        markdown = format_unstructured(audio_path, segments)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8")
    print(f"[ok] Wrote: {out_path}")
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe audio to structured Markdown "
            "(OpenAI gpt-4o-transcribe-diarize)."
        ),
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Audio file(s) and/or folder(s) containing audio.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_PACKAGE_ROOT,
        help=f"Directory for .md output (default: {_PACKAGE_ROOT}).",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="ISO-639-1 language code (optional).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            f"Transcription model override "
            f"(default: {config.transcription_model} or TRANSCRIPTION_MODEL env)."
        ),
    )
    parser.add_argument(
        "--no-structure",
        action="store_true",
        help="Skip LLM structuring; use pause-based timestamp sections only.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing Markdown files.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    _require_api_key()
    parser = build_parser()
    args = parser.parse_args(argv)

    audio_files = discover_audio_paths(args.paths)
    if not audio_files:
        print("No audio files found.", file=sys.stderr)
        return 1

    model = args.model or config.transcription_model
    client = OpenAI()
    output_dir = args.output_dir.resolve()
    written = 0

    for audio_path in audio_files:
        result = process_file(
            client,
            audio_path,
            output_dir,
            model=model,
            language=args.language,
            use_structure=not args.no_structure,
            force=args.force,
        )
        if result is not None:
            written += 1

    print(f"\nDone. {written} Markdown file(s) in {output_dir}")
    return 0 if written > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
