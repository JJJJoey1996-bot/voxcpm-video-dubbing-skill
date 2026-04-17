#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
WHISPER_MODEL_FILE = "ggml-medium.en.bin"
SUPPORTED_PYTHONS = {(3, 11), (3, 12)}
PREFERRED_PYTHON = (3, 12)


def venv_python() -> Path:
    if platform.system() == "Windows":
        return REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    return REPO_ROOT / ".venv" / "bin" / "python"


def whisper_cli_candidates() -> list[Path]:
    whisper_root = REPO_ROOT / "tools" / "whisper.cpp" / "build" / "bin"
    return [
        whisper_root / "whisper-cli",
        whisper_root / "whisper-cli.exe",
        whisper_root / "Release" / "whisper-cli.exe",
        whisper_root / "Release" / "whisper-cli",
    ]


def find_whisper_cli() -> Path | None:
    for candidate in whisper_cli_candidates():
        if candidate.exists():
            return candidate
    return None


def run_capture(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )


def package_probe() -> dict[str, object]:
    py = venv_python()
    if not py.exists():
        return {"ok": False, "error": "managed .venv is missing"}

    code = """
import importlib.util, json, sys
packages = ["torch", "torchaudio", "transformers", "demucs", "misaki", "huggingface_hub", "modelscope"]
status = {name: bool(importlib.util.find_spec(name)) for name in packages}
device = "cpu"
details = {"python": f"{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}"}
try:
    import torch
    details["torch"] = torch.__version__
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
except Exception as exc:
    details["torch_error"] = str(exc)
details["device"] = device
print(json.dumps({"packages": status, "details": details}))
""".strip()
    result = run_capture([str(py), "-c", code])
    if result.returncode != 0:
        return {"ok": False, "error": result.stderr.strip() or result.stdout.strip()}
    try:
        payload = json.loads(result.stdout.strip())
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": f"invalid package probe output: {exc}"}
    payload["ok"] = True
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-check for the short-video dubbing skill.")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args()

    report: dict[str, object] = {
        "repo_root": str(REPO_ROOT),
        "platform": platform.platform(),
        "checks": {},
        "summary": {"ok": True, "failures": []},
    }

    checks: dict[str, object] = report["checks"]  # type: ignore[assignment]
    failures: list[str] = report["summary"]["failures"]  # type: ignore[index]

    checks["system_tools"] = {
        name: bool(shutil.which(name)) for name in ("git", "ffmpeg", "cmake")
    }
    for name, ok in checks["system_tools"].items():  # type: ignore[union-attr]
        if not ok:
            failures.append(f"Missing system tool: {name}")

    py = venv_python()
    venv_ok = py.exists()
    python_info: dict[str, object] = {
        "path": str(py),
        "exists": venv_ok,
    }
    if venv_ok:
        result = run_capture([str(py), "-c", "import sys; print(sys.version_info[0], sys.version_info[1])"])
        if result.returncode == 0:
            major, minor = result.stdout.strip().split()
            python_info["version"] = f"{major}.{minor}"
            current = (int(major), int(minor))
            python_info["supported"] = current in SUPPORTED_PYTHONS
            python_info["preferred"] = current == PREFERRED_PYTHON
            if not python_info["supported"]:
                failures.append(
                    f"Managed .venv uses Python {major}.{minor}; expected one of "
                    + ", ".join(f"{major}.{minor}" for major, minor in sorted(SUPPORTED_PYTHONS))
                )
        else:
            python_info["supported"] = False
            python_info["error"] = result.stderr.strip() or result.stdout.strip()
            failures.append("Unable to inspect managed .venv Python")
    else:
        python_info["supported"] = False
        failures.append("Managed .venv is missing")
    checks["managed_python"] = python_info

    package_status = package_probe()
    checks["python_packages"] = package_status
    if not package_status.get("ok"):
        failures.append("Unable to verify Python packages inside managed .venv")
    else:
        packages = package_status["packages"]  # type: ignore[index]
        missing_packages = [name for name, ok in packages.items() if not ok]  # type: ignore[union-attr]
        if missing_packages:
            failures.append(f"Missing Python packages: {', '.join(sorted(missing_packages))}")

    whisper_cli = find_whisper_cli()
    whisper_model = REPO_ROOT / "tools" / "whisper.cpp" / "models" / WHISPER_MODEL_FILE
    checks["whisper_cpp"] = {
        "binary": str(whisper_cli) if whisper_cli else None,
        "binary_exists": bool(whisper_cli),
        "model": str(whisper_model),
        "model_exists": whisper_model.exists(),
    }
    if not whisper_cli:
        failures.append("whisper.cpp binary is missing")
    if not whisper_model.exists():
        failures.append("whisper.cpp model is missing")

    voxcpm_model_dir = REPO_ROOT / "models" / "VoxCPM2"
    model_ok = all((voxcpm_model_dir / filename).exists() for filename in ("config.json", "model.safetensors"))
    checks["voxcpm_model"] = {
        "path": str(voxcpm_model_dir),
        "ready": model_ok,
    }
    if not model_ok:
        failures.append("VoxCPM2 model is missing or incomplete")

    report["summary"]["ok"] = not failures  # type: ignore[index]

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        raise SystemExit(0 if not failures else 1)

    print("Short Video Dubbing Skill Doctor")
    print(f"Repo root: {REPO_ROOT}")
    print(f"Managed Python: {python_info.get('path')}")
    if package_status.get("ok"):
        details = package_status.get("details", {})
        print(
            "Runtime probe: "
            f"python={details.get('python', 'unknown')} "
            f"torch={details.get('torch', 'missing')} "
            f"device={details.get('device', 'unknown')}"
        )
    if failures:
        print("Status: FAILED")
        for item in failures:
            print(f"- {item}")
        print("Recommended fix: run `python skills/short-video-dubbing/scripts/agent_pipeline.py install` again.")
        raise SystemExit(1)

    print("Status: OK")
    print("- Managed Python is pinned and ready")
    print("- Core Python packages are available")
    print("- whisper.cpp binary and model are present")
    print("- VoxCPM2 model files are ready")


if __name__ == "__main__":
    main()
