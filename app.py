import argparse
import logging
import os
import platform
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import gradio as gr
import numpy as np
import soundfile as sf

from voxcpm.presets import COOKBOOK_GUIDE_HTML, VOICE_PERSONA_PRESETS, VOICE_TAG_PRESETS
from voxcpm.runtime import DEFAULT_MODELSCOPE_MODEL_ID, DEFAULT_PYTORCH_MODEL_ID, EngineConfig, GenerationRequest, VoxCPMEngine

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _default_model_id() -> str:
    local_model_dir = Path(__file__).resolve().parent / "models" / "VoxCPM2"
    if (local_model_dir / "config.json").exists():
        return str(local_model_dir)
    return DEFAULT_PYTORCH_MODEL_ID


APP_CSS = """
:root {
    --bg: #f6f7fb;
    --surface: rgba(255, 255, 255, 0.92);
    --surface-strong: rgba(255, 255, 255, 0.98);
    --line: rgba(15, 23, 42, 0.08);
    --text: #0f172a;
    --muted: #5b6475;
    --primary: #111827;
    --primary-soft: #eef2ff;
    --accent: #2563eb;
    --accent-soft: rgba(37, 99, 235, 0.08);
    --success: #0f766e;
}

body, .gradio-container {
    background:
        radial-gradient(circle at 20% 0%, rgba(59, 130, 246, 0.08), transparent 26%),
        radial-gradient(circle at 100% 0%, rgba(15, 23, 42, 0.05), transparent 18%),
        linear-gradient(180deg, #fbfcfe 0%, #f4f6fb 100%);
    color: var(--text);
}

.app-shell {
    max-width: 1320px;
    margin: 0 auto;
    padding-bottom: 40px;
}

.hero {
    padding: 30px 32px;
    border-radius: 28px;
    background:
        linear-gradient(140deg, rgba(255, 255, 255, 0.94), rgba(248, 250, 252, 0.98)),
        linear-gradient(135deg, #111827, #1f2937);
    border: 1px solid rgba(255, 255, 255, 0.55);
    box-shadow: 0 24px 70px rgba(15, 23, 42, 0.10);
    position: relative;
    overflow: hidden;
}

.hero::after {
    content: "";
    position: absolute;
    inset: -20% -10% auto auto;
    width: 280px;
    height: 280px;
    background: radial-gradient(circle, rgba(37, 99, 235, 0.16), transparent 65%);
}

.hero h1 {
    margin: 0;
    font-size: 2.5rem;
    letter-spacing: -0.05em;
    color: #0f172a;
}

.status-bar {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
    margin-top: 16px;
}

.status-card {
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.78);
    border: 1px solid var(--line);
    padding: 14px 16px;
}

.status-label {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #6b7280;
}

.status-value {
    margin-top: 6px;
    font-size: 1rem;
    font-weight: 600;
    color: #111827;
}

.panel {
    border-radius: 26px;
    background: var(--surface);
    border: 1px solid var(--line);
    box-shadow: 0 16px 40px rgba(15, 23, 42, 0.06);
    backdrop-filter: blur(14px);
}

.panel-title {
    font-size: 1.08rem;
    font-weight: 700;
    color: #111827;
}

.mode-hint {
    color: var(--muted);
    line-height: 1.65;
    font-size: 0.95rem;
}

.tag-cloud {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 4px;
}

.tag-chip {
    border: 1px solid rgba(15, 23, 42, 0.08);
    border-radius: 999px;
    background: white;
    color: #111827;
    padding: 8px 12px;
    font-size: 0.92rem;
    cursor: pointer;
    transition: all 0.18s ease;
}

.tag-chip:hover {
    border-color: rgba(37, 99, 235, 0.3);
    background: rgba(37, 99, 235, 0.06);
    transform: translateY(-1px);
}

.tag-chip code {
    background: transparent;
    font-size: 0.88rem;
    color: #2563eb;
}

.tips {
    color: var(--muted);
    line-height: 1.75;
    font-size: 0.95rem;
}

#generate-btn {
    background: linear-gradient(135deg, #111827, #1f2937);
    border: none;
}

#generate-btn:hover {
    filter: brightness(1.03);
}

@media (max-width: 960px) {
    .status-bar {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}

@media (max-width: 640px) {
    .hero {
        padding: 22px 20px;
    }

    .hero h1 {
        font-size: 2rem;
    }

    .status-bar {
        grid-template-columns: 1fr;
    }
}
"""


APP_THEME = gr.themes.Base(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Instrument Sans"), gr.themes.GoogleFont("Noto Sans SC"), "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "monospace"],
)


APP_HEAD = """
<script>
(() => {
  const state = { active: null };

  function isEditable(el) {
    return !!el && (
      el.tagName === "TEXTAREA" ||
      (el.tagName === "INPUT" && ["text", "search"].includes(el.type))
    );
  }

  function rememberActive(el) {
    if (!isEditable(el)) return;
    const wrapper = el.closest("#target-textbox, #control-textbox");
    if (wrapper) state.active = el;
  }

  function insertAtCursor(target, text) {
    if (!target) return false;
    const start = typeof target.selectionStart === "number" ? target.selectionStart : target.value.length;
    const end = typeof target.selectionEnd === "number" ? target.selectionEnd : target.value.length;
    const before = target.value.slice(0, start);
    const after = target.value.slice(end);
    target.value = before + text + after;
    const nextPos = start + text.length;
    target.focus();
    target.setSelectionRange(nextPos, nextPos);
    target.dispatchEvent(new Event("input", { bubbles: true }));
    target.dispatchEvent(new Event("change", { bubbles: true }));
    return true;
  }

  document.addEventListener("focusin", (event) => rememberActive(event.target));
  document.addEventListener("click", (event) => rememberActive(event.target));

  document.addEventListener("click", (event) => {
    const button = event.target.closest(".tag-chip");
    if (!button) return;
    event.preventDefault();

    let target = state.active;
    if (!isEditable(target)) {
      target = document.querySelector("#target-textbox textarea, #target-textbox input");
    }
    const value = button.dataset.value || "";
    const insertion = (target && target.value && !/\\s$/.test(target.value) ? " " : "") + value;
    insertAtCursor(target, insertion);
  });
})();
</script>
"""


def render_tag_cloud() -> str:
    chips = []
    for tag in VOICE_TAG_PRESETS:
        chips.append(
            f'<button class="tag-chip" data-value="{tag["value"]}" type="button">{tag["label"]} <code>{tag["value"]}</code></button>'
        )
    return '<div class="tag-cloud">' + "".join(chips) + "</div>"


class VoxCPMStudio:
    def __init__(self, config: EngineConfig):
        self.config = config
        self.engine = VoxCPMEngine(config)
        self._managed_trimmed_paths: set[str] = set()

    @staticmethod
    def _inspect_audio_duration(audio_path: Optional[str]) -> float:
        if not audio_path:
            return 0.0
        return float(sf.info(audio_path).duration)

    def update_reference_trim(self, reference_audio: Optional[str]):
        duration = self._inspect_audio_duration(reference_audio)
        if duration <= 0:
            return (
                gr.update(value=0.0),
                gr.update(value=0.0),
                gr.update(value="未检测到参考音频时长。", visible=True),
            )
        return (
            gr.update(value=0.0),
            gr.update(value=round(duration, 2)),
            gr.update(
                value=f"当前参考音频时长：{duration:.2f}s。生成和自动转写都会优先使用你设置的裁剪区间。",
                visible=True,
            ),
        )

    def apply_reference_trim(
        self,
        reference_audio: Optional[str],
        trim_start: float,
        trim_end: float,
    ):
        if not reference_audio:
            raise gr.Error("请先上传参考音频，再应用裁剪。")

        prepared_audio, temp_path = self._prepare_reference_audio(reference_audio, trim_start, trim_end)
        used_duration = self._inspect_audio_duration(prepared_audio)
        if temp_path:
            if reference_audio in self._managed_trimmed_paths and os.path.exists(reference_audio):
                try:
                    os.unlink(reference_audio)
                except OSError:
                    pass
                self._managed_trimmed_paths.discard(reference_audio)
            self._managed_trimmed_paths.add(temp_path)
        status = f"已应用裁剪，当前参考音频时长：{used_duration:.2f}s。后续转写和克隆都会使用这段最新音频。"
        return (
            gr.update(value=prepared_audio),
            gr.update(value=0.0),
            gr.update(value=round(used_duration, 2)),
            gr.update(value=status, visible=True),
        )

    def _prepare_reference_audio(
        self,
        reference_audio: Optional[str],
        trim_start: float,
        trim_end: float,
    ) -> tuple[Optional[str], Optional[str]]:
        if not reference_audio:
            return None, None

        total_duration = self._inspect_audio_duration(reference_audio)
        start = max(0.0, float(trim_start or 0.0))
        end = float(trim_end or 0.0)
        if end <= 0 or end > total_duration:
            end = total_duration

        if start >= end:
            raise gr.Error("参考音频裁剪范围无效：开始时间必须小于结束时间。")

        if start == 0.0 and abs(end - total_duration) < 1e-3:
            return reference_audio, None

        audio, sample_rate = sf.read(reference_audio, always_2d=True)
        start_frame = int(start * sample_rate)
        end_frame = int(end * sample_rate)
        clipped = audio[start_frame:end_frame]
        if clipped.size == 0:
            raise gr.Error("裁剪后的参考音频为空，请重新调整开始和结束时间。")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            temp_path = tmp_file.name
        sf.write(temp_path, clipped, sample_rate)
        return temp_path, temp_path

    @staticmethod
    def _split_long_text(text: str, max_chars: int = 110) -> list[str]:
        cleaned = re.sub(r"\s+", " ", (text or "").strip())
        if not cleaned:
            return []
        sentences = [
            chunk.strip()
            for chunk in re.split(r"(?<=[。！？!?；;：:\n])\s*", cleaned)
            if chunk.strip()
        ]
        if not sentences:
            return [cleaned]

        chunks: list[str] = []
        current = ""
        for sentence in sentences:
            if len(sentence) > max_chars:
                if current:
                    chunks.append(current.strip())
                    current = ""
                for idx in range(0, len(sentence), max_chars):
                    chunks.append(sentence[idx : idx + max_chars].strip())
                continue

            candidate = f"{current} {sentence}".strip() if current else sentence
            if current and len(candidate) > max_chars:
                chunks.append(current.strip())
                current = sentence
            else:
                current = candidate

        if current:
            chunks.append(current.strip())
        return chunks

    def _generate_single(self, request: GenerationRequest) -> tuple[int, np.ndarray]:
        return self.engine.generate(request)

    def _generate_segmented(self, request: GenerationRequest) -> tuple[int, np.ndarray, int]:
        segments = self._split_long_text(request.text)
        if len(segments) <= 1:
            sample_rate, wav = self._generate_single(request)
            return sample_rate, wav, 1

        rendered_segments: list[np.ndarray] = []
        sample_rate: Optional[int] = None
        pause = np.zeros(2400, dtype=np.float32)
        anchor_reference_audio: Optional[str] = request.reference_audio
        temp_anchor_path: Optional[str] = None

        try:
            for index, segment in enumerate(segments):
                segment_request = GenerationRequest(
                    text=segment,
                    control=request.control,
                    prompt_audio=request.prompt_audio if index == 0 else None,
                    reference_audio=anchor_reference_audio,
                    prompt_text=request.prompt_text if index == 0 else None,
                    cfg_value=request.cfg_value,
                    inference_timesteps=request.inference_timesteps,
                    normalize=request.normalize,
                    denoise=request.denoise,
                )
                current_rate, current_wav = self._generate_single(segment_request)
                if sample_rate is None:
                    sample_rate = current_rate

                clip = np.asarray(current_wav, dtype=np.float32)
                rendered_segments.append(clip)

                # Use the first generated segment as a stable voice anchor for later segments,
                # but keep later chunks on the reference-audio path so control text is not read aloud.
                if index == 0:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                        temp_anchor_path = tmp_file.name
                    sf.write(temp_anchor_path, clip, current_rate)
                    anchor_reference_audio = temp_anchor_path

            merged_parts: list[np.ndarray] = []
            for index, clip in enumerate(rendered_segments):
                if index:
                    merged_parts.append(pause)
                merged_parts.append(clip)
            return sample_rate or 24000, np.concatenate(merged_parts), len(rendered_segments)
        finally:
            if temp_anchor_path and os.path.exists(temp_anchor_path):
                try:
                    os.unlink(temp_anchor_path)
                except OSError:
                    pass

    def backend_summary(self) -> dict[str, str]:
        info = self.engine.describe()
        return {
            "backend": info["backend"],
            "device": info["device"],
            "model": info["model_id"],
            "acceleration": "MPS GPU" if info["device"] == "mps" else info["device"].upper(),
        }

    def apply_persona(self, preset_name: str):
        preset = next((item for item in VOICE_PERSONA_PRESETS if item["name"] == preset_name), None)
        if not preset:
            return gr.update(), gr.update(), gr.update()
        return preset["control"], preset["text"], "声音设计"

    def mode_updates(self, mode: str):
        is_design = mode == "声音设计"
        is_ultimate = mode == "极致克隆"
        show_reference_tools = not is_design
        status = {
            "声音设计": "无需参考音频，直接用控制指令描述音色。",
            "可控克隆": "上传参考音频后，仍然可以继续用控制指令微调情绪、语速和说话方式。",
            "极致克隆": "上传参考音频并填写逐字稿，同时仍支持控制指令做轻微风格微调。",
        }[mode]
        return (
            gr.update(visible=show_reference_tools),
            gr.update(visible=show_reference_tools),
            gr.update(visible=show_reference_tools),
            gr.update(visible=show_reference_tools),
            gr.update(visible=show_reference_tools),
            gr.update(visible=is_ultimate),
            gr.update(visible=is_ultimate),
            gr.update(interactive=True),
            status,
        )

    def transcribe_reference(
        self,
        reference_audio: Optional[str],
        trim_start: float,
        trim_end: float,
    ) -> str:
        if not reference_audio:
            raise gr.Error("请先上传参考音频，再执行自动转写。")
        prepared_audio, temp_path = self._prepare_reference_audio(reference_audio, trim_start, trim_end)
        try:
            return self.engine.transcribe(prepared_audio)
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

    def generate(
        self,
        mode: str,
        text: str,
        control: str,
        reference_audio: Optional[str],
        trim_start: float,
        trim_end: float,
        prompt_text: str,
        cfg_value: float,
        inference_timesteps: int,
        normalize: bool,
        denoise: bool,
        split_long_text: bool,
    ):
        if mode != "声音设计" and not reference_audio:
            raise gr.Error("当前模式需要先上传参考音频。")
        if mode == "极致克隆" and not (prompt_text or "").strip():
            raise gr.Error("极致克隆模式需要参考音频对应的文本。")

        prepared_reference_audio, temp_reference_path = self._prepare_reference_audio(
            reference_audio, trim_start, trim_end
        )

        try:
            request = GenerationRequest(
                text=text,
                control=control,
                prompt_audio=prepared_reference_audio if mode == "极致克隆" else None,
                reference_audio=prepared_reference_audio if mode != "声音设计" else None,
                prompt_text=prompt_text if mode == "极致克隆" else None,
                cfg_value=float(cfg_value),
                inference_timesteps=int(inference_timesteps),
                normalize=bool(normalize),
                denoise=bool(denoise),
            )
            started = time.perf_counter()
            segment_count = 1
            if split_long_text:
                sample_rate, wav, segment_count = self._generate_segmented(request)
            else:
                sample_rate, wav = self.engine.generate(request)
            profile = self.engine.last_profile()
            wall_seconds = time.perf_counter() - started
            status = (
                f"已完成 | mode={request.mode} | backend={self.engine.backend.kind} | "
                f"device={self.engine.describe()['device']} | sample_rate={sample_rate} | "
                f"耗时={wall_seconds:.1f}s"
            )
            if prepared_reference_audio and mode != "声音设计":
                used_duration = self._inspect_audio_duration(prepared_reference_audio)
                status += f" | 参考片段={used_duration:.1f}s"
            if split_long_text:
                status += f" | 分段={segment_count}"
            if profile.get("cold_start"):
                status += f" | 首次加载={float(profile.get('load_seconds', 0.0)):.1f}s"
            return status, (sample_rate, wav)
        finally:
            if temp_reference_path and os.path.exists(temp_reference_path):
                try:
                    os.unlink(temp_reference_path)
                except OSError:
                    pass


def build_app(studio: VoxCPMStudio):
    gr.set_static_paths(paths=[Path.cwd().absolute() / "assets"])
    backend = studio.backend_summary()

    with gr.Blocks(fill_width=True, title="VoxCPM2 Mac Studio", head=APP_HEAD) as demo:
        with gr.Column(elem_classes=["app-shell"]):
            gr.HTML(
                f"""
                <section class="hero">
                  <h1>VoxCPM2 Mac Studio</h1>
                  <div class="status-bar">
                    <div class="status-card"><div class="status-label">Backend</div><div class="status-value">{backend["backend"]}</div></div>
                    <div class="status-card"><div class="status-label">Device</div><div class="status-value">{backend["device"]}</div></div>
                    <div class="status-card"><div class="status-label">Acceleration</div><div class="status-value">{backend["acceleration"]}</div></div>
                  </div>
                </section>
                """
            )

            with gr.Row():
                with gr.Column(scale=7, elem_classes=["panel"]):
                    gr.Markdown('<div class="panel-title">创作面板</div>')
                    mode_hint = gr.Markdown(
                        '<div class="mode-hint">无需参考音频，直接用控制指令描述音色。</div>'
                    )

                    mode = gr.Radio(
                        choices=["声音设计", "可控克隆", "极致克隆"],
                        value="声音设计",
                        label="生成模式",
                    )

                    with gr.Row():
                        persona = gr.Dropdown(
                            choices=[item["name"] for item in VOICE_PERSONA_PRESETS],
                            label="预设人设",
                            value=VOICE_PERSONA_PRESETS[0]["name"],
                            scale=4,
                        )
                        persona_apply = gr.Button("套用人设", variant="secondary", scale=1)

                    reference_audio = gr.Audio(
                        sources=["upload", "microphone"],
                        type="filepath",
                        label="参考音频",
                        visible=False,
                    )
                    with gr.Row():
                        trim_start = gr.Number(
                            label="裁剪开始时间（秒）", value=0.0, minimum=0.0, precision=2, visible=False
                        )
                        trim_end = gr.Number(
                            label="裁剪结束时间（秒，0=到末尾）", value=0.0, minimum=0.0, precision=2, visible=False
                        )
                    trim_hint = gr.Markdown(
                        '<div class="mode-hint">上传后可手动填写开始和结束时间，生成与自动转写都会使用裁剪后的参考音频。</div>',
                        visible=False,
                    )
                    prompt_text = gr.Textbox(
                        label="参考音频逐字稿",
                        lines=3,
                        placeholder="极致克隆模式下填写参考音频内容，可点击自动转写后再手动修正。",
                        visible=False,
                    )

                    with gr.Row():
                        apply_trim_btn = gr.Button("✂ 裁剪", variant="secondary", visible=False)
                        transcribe_btn = gr.Button("自动转写参考音频", variant="secondary", visible=False)
                        generate_btn = gr.Button("开始生成", variant="primary", elem_id="generate-btn")

                    control = gr.Textbox(
                        label="控制指令 / 音色描述",
                        lines=3,
                        value=VOICE_PERSONA_PRESETS[0]["control"],
                        placeholder="例如：成熟女声，冷静、贴耳、语速偏慢，结尾带一点气声。",
                        elem_id="control-textbox",
                    )
                    text = gr.Textbox(
                        label="目标文本",
                        lines=7,
                        value=VOICE_PERSONA_PRESETS[0]["text"],
                        placeholder="把标签插入到光标所在位置，而不是强制追加到末尾。",
                        elem_id="target-textbox",
                    )

                    gr.Markdown(
                        '<div class="panel-title" style="margin-top:10px;">声音标签</div>'
                    )
                    gr.Markdown(
                        '<div class="mode-hint">点击任意标签，会直接插入到当前聚焦的输入框光标位置；如果当前没有聚焦输入框，就默认插入到目标文本。</div>'
                    )
                    gr.HTML(render_tag_cloud())

                    gr.Markdown(COOKBOOK_GUIDE_HTML)

                    with gr.Accordion("高级设置", open=False):
                        cfg_value = gr.Slider(
                            minimum=0.8,
                            maximum=4.0,
                            value=2.0,
                            step=0.1,
                            label="CFG",
                            info="MPS 默认建议 1.6 - 2.4，过高会更慢也更容易不稳定。",
                        )
                        inference_timesteps = gr.Slider(
                            minimum=1,
                            maximum=24,
                            value=10,
                            step=1,
                            label="推理步数",
                            info="Apple Silicon 上通常先从 8 - 10 开始，通常是速度和质量的更佳平衡。",
                        )
                        with gr.Row():
                            normalize = gr.Checkbox(label="文本规范化", value=False)
                            denoise = gr.Checkbox(label="参考音频降噪", value=False)
                            split_long_text = gr.Checkbox(
                                label="长文本自动切分后合并",
                                value=False,
                                info="开启后会按句子自动拆段生成，再顺序拼接，适合较长正文。",
                            )

                with gr.Column(scale=5, elem_classes=["panel"]):
                    gr.Markdown('<div class="panel-title">结果与建议</div>')
                    status = gr.Textbox(label="状态", value="等待生成", interactive=False)
                    audio_output = gr.Audio(label="生成结果")
                    gr.Markdown(
                        """
                        <div class="tips">
                        当前默认策略是先把 <code>PyTorch + MPS</code> 跑稳：
                        <br>1. 声音设计：不传参考音频，只写正文和控制指令。
                        <br>2. 可控克隆：上传参考音频后，控制指令仍然生效。
                        <br>3. 极致克隆：上传参考音频 + 逐字稿，同时仍可用控制指令做轻微微调。
                        <br>4. 方言场景优先改写正文，再在控制指令里简短写方言名。
                        <br>5. 如果你只是想追求更高吞吐，先把推理步数压到 6 - 8。
                        <br>6. 第一次生成通常最慢，因为会做模型首轮加载；第二次开始更接近真实速度。
                        <br>7. 参考音频降噪走的是更重的 CPU 链路，不勾选时现在不会提前加载它。
                        <br>8. 长文本如果不稳定，可以开启“长文本自动切分后合并”。
                        </div>
                        """
                    )

            persona_apply.click(
                fn=studio.apply_persona,
                inputs=[persona],
                outputs=[control, text, mode],
            )

            mode.change(
                fn=studio.mode_updates,
                inputs=[mode],
                outputs=[
                    reference_audio,
                    trim_start,
                    trim_end,
                    trim_hint,
                    apply_trim_btn,
                    prompt_text,
                    transcribe_btn,
                    control,
                    mode_hint,
                ],
            )

            reference_audio.change(
                fn=studio.update_reference_trim,
                inputs=[reference_audio],
                outputs=[trim_start, trim_end, trim_hint],
            )

            apply_trim_btn.click(
                fn=studio.apply_reference_trim,
                inputs=[reference_audio, trim_start, trim_end],
                outputs=[reference_audio, trim_start, trim_end, trim_hint],
            )

            transcribe_btn.click(
                fn=studio.transcribe_reference,
                inputs=[reference_audio, trim_start, trim_end],
                outputs=[prompt_text],
            )

            generate_btn.click(
                fn=studio.generate,
                inputs=[
                    mode,
                    text,
                    control,
                    reference_audio,
                    trim_start,
                    trim_end,
                    prompt_text,
                    cfg_value,
                    inference_timesteps,
                    normalize,
                    denoise,
                    split_long_text,
                ],
                outputs=[status, audio_output],
                show_progress=True,
            )

    return demo


def main():
    default_device = "mps" if _is_apple_silicon() else "auto"

    parser = argparse.ArgumentParser(description="VoxCPM2 Mac Studio WebUI")
    parser.add_argument("--model-id", type=str, default=_default_model_id())
    parser.add_argument("--modelscope-model-id", type=str, default=DEFAULT_MODELSCOPE_MODEL_ID)
    parser.add_argument("--device", type=str, default=default_device)
    parser.add_argument("--port", type=int, default=8808)
    parser.add_argument("--server-name", type=str, default="0.0.0.0")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-denoiser", action="store_true")
    parser.add_argument("--no-optimize", action="store_true")
    args = parser.parse_args()

    config = EngineConfig(
        model_id=args.model_id,
        modelscope_model_id=args.modelscope_model_id,
        device=args.device,
        enable_denoiser=not args.no_denoiser,
        optimize=not args.no_optimize,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    studio = VoxCPMStudio(config)
    demo = build_app(studio)
    demo.queue(max_size=12, default_concurrency_limit=1).launch(
        server_name=args.server_name,
        server_port=args.port,
        show_error=True,
        theme=APP_THEME,
        css=APP_CSS,
    )


if __name__ == "__main__":
    main()
