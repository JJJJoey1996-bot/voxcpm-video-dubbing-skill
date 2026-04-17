Translate for dubbing, not subtitles.

Agent rules:

1. Read the full transcript first before translating any segment.
2. Keep the translated tone natural for spoken dubbing.
3. Respect each segment's `duration_ms` and `phoneme_budget`.
4. Shorten or expand wording so the spoken length stays close to the source timing.
5. Keep the start of each segment semantically aligned with the original; timing error should be absorbed at the end, not by delaying the next line.
6. Prefer short spoken clauses over written-bookish sentences.
7. When a source span is clearly one continuous monologue, you may rewrite each sub-segment for smoother spoken flow, and individual words do not need literal preservation as long as the overall meaning of the full passage stays basically right.
8. Return or save strict JSON with:
   `{"segments":[{"index":0,"translated_text":"..."}, ...]}`
9. Do not invent extra segments or change indexes.
10. If a line is too short in available speaking time, aggressively simplify it. Prefer a compact phrase over a faithful long sentence.
11. If a retry prompt marks a line as still too long, shorten it even further than the provided phoneme and character hints.
12. If a very short segment still cannot be made natural within the timing budget, an empty string is allowed.
