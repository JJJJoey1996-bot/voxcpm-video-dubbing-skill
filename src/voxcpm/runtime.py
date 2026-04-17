from __future__ import annotations

import logging
import os
import platform
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from .core import VoxCPM

logger = logging.getLogger(__name__)


DEFAULT_PYTORCH_MODEL_ID = "openbmb/VoxCPM2"
DEFAULT_ASR_MODEL_ID = "iic/SenseVoiceSmall"
DEFAULT_MODELSCOPE_MODEL_ID = "OpenBMB/VoxCPM2"


@dataclass(slots=True)
class GenerationRequest:
    text: str
    control: str = ""
    prompt_audio: Optional[str] = None
    reference_audio: Optional[str] = None
    prompt_text: Optional[str] = None
    cfg_value: float = 2.0
    inference_timesteps: int = 10
    normalize: bool = False
    denoise: bool = False

    @property
    def cleaned_text(self) -> str:
        return (self.text or "").strip()

    @property
    def cleaned_control(self) -> str:
        return (self.control or "").strip()

    @property
    def cleaned_prompt_text(self) -> Optional[str]:
        text = (self.prompt_text or "").strip()
        return text or None

    @property
    def final_text(self) -> str:
        if self.cleaned_control:
            return f"({self.cleaned_control}){self.cleaned_text}"
        return self.cleaned_text

    @property
    def mode(self) -> str:
        if self.prompt_audio and self.cleaned_prompt_text:
            return "ultimate_clone"
        if self.reference_audio:
            return "clone"
        return "design"


@dataclass(slots=True)
class EngineConfig:
    model_id: str = DEFAULT_PYTORCH_MODEL_ID
    modelscope_model_id: str = DEFAULT_MODELSCOPE_MODEL_ID
    device: str = "auto"
    enable_denoiser: bool = True
    optimize: bool = False
    cache_dir: Optional[str] = None
    local_files_only: bool = False
    zipenhancer_model_id: Optional[str] = "iic/speech_zipenhancer_ans_multiloss_16k_base"
    asr_model_id: str = DEFAULT_ASR_MODEL_ID


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


class PyTorchVoxCPMBackend:
    kind = "pytorch"

    def __init__(self, config: EngineConfig):
        self.config = config
        self._model: Optional[VoxCPM] = None
        self._last_profile: dict[str, float | bool] = {
            "cold_start": True,
            "load_seconds": 0.0,
            "generate_seconds": 0.0,
            "total_seconds": 0.0,
        }

    @property
    def model_id(self) -> str:
        return self.config.model_id

    def _tune_torch_runtime(self) -> None:
        try:
            import torch
        except Exception:
            return

        if self.config.device == "mps" or (
            self.config.device == "auto"
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            # Keep CPU helper threads modest to reduce launch overhead and heat on Apple Silicon.
            torch.set_num_threads(max(1, min(4, (os.cpu_count() or 4) // 2)))
            if hasattr(torch, "set_num_interop_threads"):
                try:
                    torch.set_num_interop_threads(1)
                except RuntimeError:
                    pass

    def get_model(self) -> VoxCPM:
        if self._model is None:
            start = time.perf_counter()
            self._tune_torch_runtime()
            logger.info("Loading PyTorch VoxCPM backend from %s", self.model_id)
            self._model = VoxCPM.from_pretrained(
                hf_model_id=self.model_id,
                load_denoiser=self.config.enable_denoiser,
                zipenhancer_model_id=self.config.zipenhancer_model_id,
                cache_dir=self.config.cache_dir,
                local_files_only=self.config.local_files_only,
                optimize=self.config.optimize,
                device=self.config.device,
            )
            self._last_profile["load_seconds"] = time.perf_counter() - start
        return self._model

    def transcribe(self, audio_path: Optional[str]) -> str:
        if not audio_path:
            return ""

        try:
            from funasr import AutoModel
        except Exception as exc:
            raise RuntimeError(
                "Optional dependency 'funasr' is not installed, so reference-audio auto-transcription is unavailable. "
                "Install the 'reference_asr' extra on Python versions below 3.13 if you need this feature."
            ) from exc

        if not hasattr(self, "_asr_model"):
            device = "cpu"
            try:
                import torch

                if torch.cuda.is_available():
                    device = "cuda:0"
            except Exception:
                device = "cpu"

            logger.info("Loading ASR model %s on %s", self.config.asr_model_id, device)
            self._asr_model = AutoModel(
                model=self.config.asr_model_id,
                disable_update=True,
                log_level="ERROR",
                device=device,
            )

        result = self._asr_model.generate(input=audio_path, language="auto", use_itn=True)
        return result[0]["text"].split("|>")[-1].strip()

    def generate(self, request: GenerationRequest) -> tuple[int, np.ndarray]:
        cold_start = self._model is None
        total_start = time.perf_counter()
        model = self.get_model()
        generate_start = time.perf_counter()
        wav = model.generate(
            text=request.final_text,
            prompt_wav_path=request.prompt_audio if request.cleaned_prompt_text else None,
            prompt_text=request.cleaned_prompt_text,
            reference_wav_path=request.reference_audio,
            cfg_value=float(request.cfg_value),
            inference_timesteps=int(request.inference_timesteps),
            normalize=bool(request.normalize),
            denoise=bool(request.denoise and request.reference_audio),
            retry_badcase=True,
            retry_badcase_max_times=2,
            retry_badcase_ratio_threshold=4.0,
        )
        self._last_profile = {
            "cold_start": cold_start,
            "load_seconds": float(self._last_profile.get("load_seconds", 0.0) if cold_start else 0.0),
            "generate_seconds": time.perf_counter() - generate_start,
            "total_seconds": time.perf_counter() - total_start,
        }
        return model.tts_model.sample_rate, wav

    def last_profile(self) -> dict[str, float | bool]:
        return dict(self._last_profile)


class VoxCPMEngine:
    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        self.backend = PyTorchVoxCPMBackend(self.config)

    def describe(self) -> dict[str, str | bool]:
        return {
            "backend": self.backend.kind,
            "device": self.config.device,
            "model_id": self.backend.model_id,
            "apple_silicon": _is_apple_silicon(),
            "modelscope_model_id": self.config.modelscope_model_id,
        }

    def generate(self, request: GenerationRequest) -> tuple[int, np.ndarray]:
        if not request.cleaned_text:
            raise ValueError("Please enter the target text to synthesize.")
        return self.backend.generate(request)

    def transcribe(self, audio_path: Optional[str]) -> str:
        return self.backend.transcribe(audio_path)

    def last_profile(self) -> dict[str, float | bool]:
        return self.backend.last_profile()


@lru_cache(maxsize=1)
def default_cache_dir() -> str:
    return str(Path.home() / ".cache" / "voxcpm")
