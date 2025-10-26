from typing import List, Dict

def segments_to_blocks(
    segments: List[Dict],
    max_secs: float = 60.0,
    max_chars: int = 1200,
    gap_break: float = 1.5
) -> List[Dict]:
    """
    Merge Whisper segments into readable time blocks.
    Each block: {"text": "...", "start": <float>, "end": <float>}
    """
    print(f"[asr_segmerge] merging {len(segments)} segments → blocks")
    blocks, cur, t0, last_end, cur_chars = [], [], None, None, 0

    for seg in segments:
        txt = (seg.get("text") or "").strip()
        if not txt:
            continue
        s, e = float(seg["start"]), float(seg["end"])
        if t0 is None:
            t0 = s

        # boundary: long gap, too long, or too many chars
        gap = 0 if last_end is None else (s - last_end)
        too_long = (e - t0) >= max_secs
        too_big = (cur_chars + len(txt)) >= max_chars
        if gap >= gap_break or too_long or too_big:
            if cur:
                blocks.append({"text": " ".join(cur).strip(), "start": t0, "end": last_end})
            cur, t0, cur_chars = [], s, 0

        cur.append(txt)
        cur_chars += len(txt) + 1
        last_end = e

    if cur:
        blocks.append({"text": " ".join(cur).strip(), "start": t0, "end": last_end})

    print(f"[asr_segmerge] produced {len(blocks)} blocks")
    return blocks
