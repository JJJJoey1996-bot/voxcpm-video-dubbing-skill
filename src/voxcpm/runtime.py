from __future__ import annotations

import inspect
import logging
import os
import platform
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .core import VoxCPM

logger = logging.getLogger(__name__)


DEFAULT_PYTORCH_MODEL_ID = "openbmb/VoxCPM2"
DEFAULT_MLX_MODEL_ID = "mlx-community/VoxCPM2-bf16"
DEFAULT_ASR_MODEL_ID = "iic/SenseVoiceSmall"
DEFAULT_MLX_ASR_MODEL_ID = "mlx-community/whisper-large-v3-turbo"


@dataclass(slots=True)
class GenerationRequest:
    text: str
    control: str = ""
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
        if self.cleaned_control and not self.cleaned_prompt_text:
            return f"({self.cleaned_control}){self.cleaned_text}"
        return self.cleaned_text

    @property
    def mode(self) -> str:
        if self.reference_audio and self.cleaned_prompt_text:
            return "ultimate_clone"
        if self.reference_audio:
            return "clone"
        return "design"


@dataclass(slots=True)
class EngineConfig:
    backend: str = "auto"
    model_id: str = DEFAULT_PYTORCH_MODEL_ID
    mlx_model_id: str = DEFAULT_MLX_MODEL_ID
    device: str = "auto"
    enable_denoiser: bool = True
    optimize: bool = True
    cache_dir: Optional[str] = None
    local_files_only: bool = False
    zipenhancer_model_id: Optional[str] = "iic/speech_zipenhancer_ans_multiloss_16k_base"
    asr_model_id: str = DEFAULT_ASR_MODEL_ID
    mlx_asr_model_id: str = DEFAULT_MLX_ASR_MODEL_ID


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _looks_like_mlx_repo(model_id: str) -> bool:
    lowered = (model_id or "").lower()
    return "mlx" in lowered or lowered.startswith("mlx-community/")


def _supports_mlx() -> bool:
    try:
        import mlx.core as mx  # noqa: F401
        import mlx_audio  # noqa: F401

        return True
    except Exception:
        return False


def select_backend(config: EngineConfig) -> str:
    explicit = (config.backend or "auto").strip().lower()
    if explicit != "auto":
        return explicit

    preferred_model = config.model_id or ""
    mlx_model = config.mlx_model_id or ""
    if _is_apple_silicon() and _supports_mlx():
        if _looks_like_mlx_repo(preferred_model) or _looks_like_mlx_repo(mlx_model):
            return "mlx"
    return "pytorch"


class PyTorchVoxCPMBackend:
    kind = "pytorch"

    def __init__(self, config: EngineConfig):
        self.config = config
        self._model: Optional[VoxCPM] = None

    @property
    def model_id(self) -> str:
        return self.config.model_id

    def get_model(self) -> VoxCPM:
        if self._model is None:
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
        return self._model

    def transcribe(self, audio_path: Optional[str]) -> str:
        if not audio_path:
            return ""

        try:
            from funasr import AutoModel
        except Exception as exc:
            raise RuntimeError("funasr is not installed, cannot auto-transcribe reference audio.") from exc

        if not hasattr(self, "_asr_model"):
            device = "cpu"
            try:
                import torch

                if torch.cuda.is_available():
                    device = "cuda:0"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device = "cpu"
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
        model = self.get_model()
        wav = model.generate(
            text=request.final_text,
            prompt_wav_path=request.reference_audio if request.cleaned_prompt_text else None,
            prompt_text=request.cleaned_prompt_text,
            reference_wav_path=request.reference_audio,
            cfg_value=float(request.cfg_value),
            inference_timesteps=int(request.inference_timesteps),
            normalize=bool(request.normalize),
            denoise=bool(request.denoise and request.reference_audio),
        )
        return model.tts_model.sample_rate, wav


class MLXVoxCPMBackend:
    kind = "mlx"

    def __init__(self, config: EngineConfig):
        self.config = config
        self._generator = None
        self._sample_rate = 24000

    @property
    def model_id(self) -> str:
        if _looks_like_mlx_repo(self.config.model_id):
            return self.config.model_id
        return self.config.mlx_model_id

    def _get_generator(self):
        if self._generator is not None:
            return self._generator

        try:
            from mlx_audio.tts.generate import generate_audio as fallback_generate_audio
            from mlx_audio.tts.utils import load_model
        except Exception as exc:
            raise RuntimeError(
                "MLX backend requested, but `mlx-audio` is not installed. "
                "Run the Mac setup script or install `mlx-audio` manually."
            ) from exc

        logger.info("Loading MLX VoxCPM backend from %s", self.model_id)
        model, processor = load_model(self.model_id)
        self._generator = {
            "model": model,
            "processor": processor,
            "generate_audio": fallback_generate_audio,
        }
        self._sample_rate = int(getattr(model, "sample_rate", getattr(processor, "sampling_rate", 24000)))
        return self._generator

    def transcribe(self, audio_path: Optional[str]) -> str:
        if not audio_path:
            return ""

        try:
            from mlx_audio.tts.utils import transcribe_audio
        except Exception as exc:
            raise RuntimeError(
                "MLX ASR helper is unavailable. Install a recent `mlx-audio` release for auto transcription."
            ) from exc

        return str(transcribe_audio(audio_path, repo=self.config.mlx_asr_model_id)).strip()

    def _build_candidate_kwargs(self, request: GenerationRequest) -> list[dict[str, Any]]:
        shared = {
            "text": request.cleaned_text,
            "cfg_value": float(request.cfg_value),
            "inference_timesteps": int(request.inference_timesteps),
        }
        if request.mode == "design":
            if request.cleaned_control:
                return [
                    {**shared, "instruct": request.cleaned_control},
                    {**shared, "control": request.cleaned_control},
                    {**shared, "prompt": request.cleaned_control},
                    {**shared, "text": request.final_text},
                ]
            return [shared]

        candidate_audio_keys = ["reference_audio", "reference_audio_path", "ref_audio", "audio_path"]
        candidate_text_keys = ["prompt_text", "reference_text", "ref_text", "transcript"]
        candidates: list[dict[str, Any]] = []

        for audio_key in candidate_audio_keys:
            base = {**shared, audio_key: request.reference_audio}
            if request.cleaned_control:
                candidates.append({**base, "instruct": request.cleaned_control})
                candidates.append({**base, "control": request.cleaned_control})
                candidates.append({**base, "prompt": request.cleaned_control})
            candidates.append(base)

            if request.cleaned_prompt_text:
                for text_key in candidate_text_keys:
                    candidates.append({**base, text_key: request.cleaned_prompt_text})
                    candidates.append(
                        {**base, text_key: request.cleaned_prompt_text, "instruct": request.cleaned_control}
                    )
        return candidates

    def generate(self, request: GenerationRequest) -> tuple[int, np.ndarray]:
        runtime = self._get_generator()
        model = runtime["model"]
        processor = runtime["processor"]
        generate_audio = runtime["generate_audio"]

        errors: list[str] = []
        for kwargs in self._build_candidate_kwargs(request):
            try:
                signature = None
                try:
                    signature = inspect.signature(generate_audio)
                except (TypeError, ValueError):
                    signature = None

                if signature is not None:
                    accepted = {
                        key: value for key, value in kwargs.items() if key in signature.parameters and value is not None
                    }
                else:
                    accepted = {key: value for key, value in kwargs.items() if value is not None}

                audio = generate_audio(model=model, processor=processor, **accepted)
                array = np.asarray(audio, dtype=np.float32).squeeze()
                return self._sample_rate, array
            except TypeError as exc:
                errors.append(f"{kwargs.keys()}: {exc}")
                continue
            except Exception:
                raise

        joined = "; ".join(errors[-3:]) if errors else "unknown MLX invocation error"
        raise RuntimeError(f"Unable to call MLX generation API with the installed mlx-audio version: {joined}")


class VoxCPMEngine:
    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        self.backend_name = select_backend(self.config)
        logger.info("Selected backend: %s", self.backend_name)

        if self.backend_name == "mlx":
            self.backend = MLXVoxCPMBackend(self.config)
        elif self.backend_name == "pytorch":
            self.backend = PyTorchVoxCPMBackend(self.config)
        else:
            raise ValueError(f"Unsupported backend: {self.backend_name}")

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.backend.kind,
            "device": self.config.device,
            "model_id": self.backend.model_id,
            "apple_silicon": _is_apple_silicon(),
            "mlx_available": _supports_mlx(),
            "cwd": os.getcwd(),
        }

    def generate(self, request: GenerationRequest) -> tuple[int, np.ndarray]:
        if not request.cleaned_text:
            raise ValueError("Please enter the target text to synthesize.")
        return self.backend.generate(request)

    def transcribe(self, audio_path: Optional[str]) -> str:
        return self.backend.transcribe(audio_path)


@lru_cache(maxsize=1)
def default_cache_dir() -> str:
    return str(Path.home() / ".cache" / "voxcpm")
