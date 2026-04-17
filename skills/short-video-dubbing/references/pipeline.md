The short-video dubbing pipeline is intentionally staged:

1. `ffmpeg` extracts full audio from the source video.
2. `demucs --two-stems=vocals` separates vocals and background.
3. `whisper.cpp` resamples the vocal stem to mono 16 kHz and runs timestamped transcription. In this project the default is CPU-first for stability with `medium.en`, while keeping the native `whisper.cpp` JSON timestamp output.
4. Word timestamps are first grouped into transcript segments using punctuation, silence gaps, and a max-duration cap.
5. For single-speaker dubbing, adjacent transcript segments are then merged into larger dubbing chunks when the silence gap between them is short. Continuous speech should still respect a chunk-duration ceiling so long videos do not collapse into one giant render job on Mac.
6. Misaki estimates a phoneme budget for each dubbing chunk so translation can stay closer to the original speaking time.
7. The agent reads the whole transcript first, then translates chunk-by-chunk under the rules in `translation_rules.md`.
8. Before any cloning happens, the translated chunks must pass a phoneme-length preflight check. If a chunk is predicted too long, it should be shortened; if it is predicted noticeably too short, it should be expanded before cloning. The lower bound should stay fairly tight so the dubbed speech does not end up obviously under-filled.
9. VoxCPM uses controllable cloning only. For the current single-speaker workflow, all translated chunks share one global reference voice clip built from the earliest effective speech, up to roughly 20 seconds. If the video has less than 20 seconds of speech, the full voiced portion is used.
10. Rendering now happens chunk-by-chunk instead of sentence-by-sentence. Each chunk is synthesized once as one continuous cloned utterance, which is more stable for timbre than cloning many tiny clips, but chunk length should stay bounded to keep thermals and retry cost under control.
11. Retry logic is also chunk-based. If a rendered chunk still looks too long after preflight, translation should be shortened again before another render; if it is only slightly long, time-compression is allowed but must stay within the configured maximum speed-up.
12. `ffmpeg` mixes the dubbed track with the separated background stem and muxes the result back into the original video stream.

Known constraints:

- `demucs` is CPU-heavy on Mac unless you deliberately move it elsewhere.
- `whisper.cpp` gives more stable timestamps on Mac than the previous FunASR path here, but chunking still matters; very long uninterrupted speech should still be split.
- Translation is intentionally agent-driven; this skill no longer depends on an API translator.
- Misaki APIs vary by language module; the implementation falls back to heuristic budgets if G2P fails.
- Very dense multi-speaker scenes are not diarized yet; the current pipeline assumes a single dominant voice track.
