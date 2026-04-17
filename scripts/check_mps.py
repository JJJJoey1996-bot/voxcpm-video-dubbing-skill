#!/usr/bin/env python3
from __future__ import annotations

import json
import platform
import sys


def main() -> int:
    try:
        import torch
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"torch import failed: {exc}"}, ensure_ascii=False, indent=2))
        return 1

    report: dict[str, object] = {
        "python": sys.version.split()[0],
        "system": platform.system(),
        "machine": platform.machine(),
        "mac_ver": platform.mac_ver()[0],
        "torch": torch.__version__,
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
    }

    if report["mps_available"]:
        try:
            report["mps_name"] = torch.backends.mps.get_name()
        except Exception as exc:
            report["mps_name"] = f"unavailable: {exc}"

        try:
            x = torch.ones((4, 4), device="mps")
            y = torch.ones((4, 4), device="mps")
            z = x @ y
            report["tensor_test"] = {
                "ok": True,
                "device": str(z.device),
                "sum": float(z.sum().item()),
            }
        except Exception as exc:
            report["tensor_test"] = {
                "ok": False,
                "error": str(exc),
            }
    else:
        report["mps_name"] = "unavailable"
        report["tensor_test"] = {
            "ok": False,
            "error": "Skipped because torch.backends.mps.is_available() returned False.",
        }

    report["ok"] = bool(report["mps_built"] and report["mps_available"] and report["tensor_test"]["ok"])
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
