#!/usr/bin/env python
"""Segment raw episodes into utterance/scene clips for TRACE streaming input.

TRACE expects per-utterance clips. This helper takes a raw video and a
cut table, and emits short MP4 clips plus a JSONL that
``run_streaming_online.py`` consumes directly.

Two cut sources are supported:

* ``--cut-jsonl``  an annotation file describing utterance boundaries with
  the StreamMER schema (``basic_info.timestamp.start/end``, ``speaker``,
  ``dialogue``). Each row's timestamps may be relative to its scene;
  ``--offset-key``/``-1``…… ALWAYS use ``--scene-windows`` to establish the
  absolute in-episode time if the timestamps are scene-relative.

* ``--scene-windows`` the released ``scene_windows.json``. Combined with an
  ``--original-video <episode.mp4>``, it cuts the whole episode into scene
  clips (scene-level granularity). This is the minimum requirement.

Usage examples:

1. Scene-level cut (everything the repo ships runs immediately after):

   python scripts/segment_video.py \
     --original-video S01E01.mp4 --episode-id S01E01 \
     --scene-windows scene_windows.json \
     --out-dir clips --clip-root /abs/path/to/clips

2. Utterance-level cut when annotations are available: the script first cuts
   the scene window (keeps the original scene-relative timebase for the
   utterance timestamps), then for each utterance row it cuts
   ``[start, end]`` from the scene clip into a per-utterance MP4 and emits a
   per-utterance JSONL row with ``video_path/start_time/end_time``.

Outputs: ``<out-dir>/<clip>.mp4`` (or ``<out-dir>/<clip>/<utt>.mp4``) and
``<out-dir>/<stem>.jsonl``.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def _run(cmd):
    print("  $", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def parse_ts(value):
    """'[00:00:00:960]' or '00:00:01.23' or '12.34' (seconds) -> float seconds."""
    s = str(value).strip().lstrip("[").rstrip("]")
    if s.count(":") == 0:
        return float(s)
    parts = s.split(":")
    if len(parts) == 4:  # hh:mm:ss:frames  (frame = 25 fps in the annotation)
        h, m, sec, fr = parts
        return int(h) * 3600 + int(m) * 60 + float(sec) + int(fr) / 25.0
    if len(parts) == 3:  # hh:mm:ss
        h, m, sec = parts
        return int(h) * 3600 + int(m) * 60 + float(sec)
    if len(parts) == 2:  # mm:ss
        m, sec = parts
        return int(m) * 60 + float(sec)
    return float(s)


def sec2ts(x, fps=25.0):
    h = int(x // 3600)
    m = int((x % 3600) // 60)
    s = x % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def main():
    ap = argparse.ArgumentParser(description="Segment raw episodes into TRACE input clips.")
    ap.add_argument("--original-video", required=True, help="raw episode .mp4")
    ap.add_argument("--episode-id", required=True, help="e.g. S01E01")
    ap.add_argument("--scene-windows", required=True, help="released scene_windows.json")
    ap.add_argument("--cut-jsonl", default="", help="optional per-utterance annotation JSONL")
    ap.add_argument("--out-dir", required=True, help="directory to write clips and JSONL")
    ap.add_argument("--clip-root", default="", help="absolute root to store in video_path (default out-dir)")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--no-video", action="store_true", help="audio-only segmentation (no re-encode)")
    args = ap.parse_args()

    scenes = json.load(open(args.scene_windows, encoding="utf-8"))
    by_ep = [s for s in scenes if str(s.get("episode_id")) == args.episode_id]
    if not by_ep:
        print(f"No scene windows for episode {args.episode_id}", file=sys.stderr)
        sys.exit(1)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    clip_root = Path(args.clip_root or str(out))

    rows = []
    for sc in sorted(by_ep, key=lambda s: float(s.get("abs_start", 0.0))):
        cid = str(sc.get("clip_id", ""))
        scene_start = float(sc.get("abs_start", 0.0))
        scene_end = float(sc.get("abs_end", 0.0))
        scene_name = str(sc.get("scene_name", cid)).replace("/", "_")

        # ---- cut the scene window from the raw episode ----
        scene_out = out / f"{scene_name}_scene.mp4"
        vf = "anull" if args.no_video else ""
        ff = ["ffmpeg", "-y", "-ss", str(scene_start), "-to", str(scene_end),
              "-i", args.original_video]
        if vf:
            ff += ["-vf", vf]
        ff += ["-c", "copy", str(scene_out)]
        _run(ff)

        # ---- utterance-level cut if annotations given ----
        if args.cut_jsonl:
            utts = [r for r in json.load(open(args.cut_jsonl, encoding="utf-8"))
                    if str(r.get("scene_id") or r.get("clip_id") or "") == cid or scene_name in str(r.get("clip_id") or "")]
            for i, u in enumerate(utts):
                basic = u.get("basic_info", {})
                ts = basic.get("timestamp", {})
                st = parse_ts(ts.get("start", "0"))
                et = parse_ts(ts.get("end", f"{st + 1:.3f}"))
                # scene-relative -> add scene offset
                abs_st = scene_start + st
                abs_et = scene_start + et
                uids = str(u.get("sample_id") or f"{scene_name}::{i}")
                clip = out / scene_name / f"{i:05d}.mp4"
                clip.parent.mkdir(parents=True, exist_ok=True)
                _run(["ffmpeg", "-y", "-ss", sec2ts(abs_st, args.fps),
                      "-to", sec2ts(abs_et, args.fps),
                      "-i", args.original_video] + (["-vf", "anull"] if args.no_video else [])
                     + ["-c", "copy", str(clip)])
                rows.append({
                    "sample_id": uids,
                    "clip_id": cid or scene_name,
                    "scene_id": cid or scene_name,
                    "stream_id": str(u.get("stream_id") or u.get("speaker") or f"s{cid}::{i}"),
                    "speaker": str(u.get("speaker") or basic.get("speaker") or ""),
                    "dialogue": str(u.get("dialogue") or basic.get("dialogue") or ""),
                    "video_path": str(clip_root / clip.relative_to(out)),
                    "start_time": round(abs_st, 3),
                    "end_time": round(abs_et, 3),
                    "target": str(u.get("target") or ""),
                })
        else:
            rows.append({
                "sample_id": cid or scene_name,
                "clip_id": cid or scene_name,
                "scene_id": cid or scene_name,
                "video_path": str(clip_root / scene_out.relative_to(out)),
                "start_time": round(scene_start, 3),
                "end_time": round(scene_end, 3),
                "target": "",
            })

    jout = out / (Path(args.original_video).stem + ".jsonl")
    with open(jout, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[done] {len(rows)} rows -> {jout}", flush=True)


if __name__ == "__main__":
    main()