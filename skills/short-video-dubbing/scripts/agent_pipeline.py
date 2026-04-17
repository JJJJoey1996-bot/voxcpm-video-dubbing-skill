#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def venv_python() -> str:
    if sys.platform.startswith("win"):
        return str(REPO_ROOT / ".venv" / "Scripts" / "python.exe")
    return str(REPO_ROOT / ".venv" / "bin" / "python")


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified entrypoint for the short-video dubbing skill.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    install_parser = subparsers.add_parser("install")
    install_parser.add_argument("--download-source", default="hf")
    install_parser.add_argument("--skip-system-deps", action="store_true")

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--video", required=True)
    prepare_parser.add_argument("--output-dir", required=True)
    prepare_parser.add_argument("--source-language", default="en")
    prepare_parser.add_argument("--target-language", default="zh")
    prepare_parser.add_argument("--chunk-merge-gap-ms", default="350")
    prepare_parser.add_argument("--max-chunk-duration-ms", default="18000")

    render_parser = subparsers.add_parser("render")
    render_parser.add_argument("--video", required=True)
    render_parser.add_argument("--probe-dir", required=True)
    render_parser.add_argument("--output-dir", required=True)
    render_parser.add_argument("--translated-json", required=True)
    render_parser.add_argument("--device", default="auto")

    args = parser.parse_args()

    if args.command == "install":
        cmd = [sys.executable, str(REPO_ROOT / "skills" / "short-video-dubbing" / "scripts" / "install.py")]
        cmd += ["--download-source", args.download_source]
        if args.skip_system_deps:
            cmd.append("--skip-system-deps")
        run(cmd)
        return

    if args.command == "prepare":
        run(
            [
                venv_python(),
                "-m",
                "voxcpm.video_dub",
                "--video",
                args.video,
                "--output-dir",
                args.output_dir,
                "--source-language",
                args.source_language,
                "--target-language",
                args.target_language,
                "--chunk-merge-gap-ms",
                args.chunk_merge_gap_ms,
                "--max-chunk-duration-ms",
                args.max_chunk_duration_ms,
            ]
        )
        return

    run(
        [
            venv_python(),
            str(REPO_ROOT / "scripts" / "render_pretranslated_video_dub.py"),
            "--video",
            args.video,
            "--probe-dir",
            args.probe_dir,
            "--output-dir",
            args.output_dir,
            "--translated-json",
            args.translated_json,
            "--device",
            args.device,
        ]
    )


if __name__ == "__main__":
    main()
