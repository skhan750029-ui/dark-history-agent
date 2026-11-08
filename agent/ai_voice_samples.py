#!/usr/bin/env python3
"""Everyday AI Easy: 4 calm, friendly voice samples (Google Chirp 3 HD) to choose the channel voice.
Writes data/ai/voice_samples/<n>_<voice>.mp3 (+ list.txt). Pick one, then set the repository variable
AI_TTS_VOICE to its full name (e.g. en-US-Chirp3-HD-Achird)."""
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ai_shorts as ai  # noqa: E402

VOICES = [("Achird", "male, friendly"), ("Sulafat", "female, warm"),
          ("Umbriel", "male, easy-going"), ("Callirrhoe", "female, easy-going")]
TEXT = [
    "Want to study smarter this week? Here are three tools worth a look.",
    "Number one: NotebookLM, from Google.",
    "You add your own notes or files, and it answers questions based on those sources.",
    "For example, upload your lecture slides and ask for a short summary.",
    "Follow for one new AI tool every day.",
]


def main():
    out = ai.ROOT / "data" / "ai" / "voice_samples"
    out.mkdir(parents=True, exist_ok=True)
    work = ai.OUT / "voice_samples"
    work.mkdir(parents=True, exist_ok=True)
    track, _ = ai.music(work, "samples")
    listing = []
    for n, (name, kind) in enumerate(VOICES, 1):
        voice = f"en-US-Chirp3-HD-{name}"
        ai.VOICE_A = voice
        lines = [{"speaker": "A", "text": t, "seg": i} for i, t in enumerate(TEXT)]
        try:
            wav, total = ai.narrate(lines, work)
        except Exception as e:
            listing.append(f"{n}. {voice} ({kind}) - FAILED: {e}")
            continue
        dst = out / f"{n}_{name}.mp3"
        ins = ["-i", str(wav)]
        fc = "[0:a]loudnorm=I=-15:TP=-1.5,aresample=44100[vo];"
        if track:
            ins += ["-stream_loop", "-1", "-i", str(track)]
            fc += "[1:a]aresample=44100,loudnorm=I=-31:TP=-6[m];[vo][m]amix=inputs=2:duration=first:normalize=0[a]"
        else:
            fc += "[vo]anull[a]"
        subprocess.run(["ffmpeg", "-y", "-v", "error", *ins, "-filter_complex", fc, "-map", "[a]",
                        "-c:a", "libmp3lame", "-b:a", "160k", str(dst)], check=True)
        listing.append(f"{n}. {voice} ({kind}) - {total:.1f}s -> {dst.name}")
    (out / "list.txt").write_text("\n".join(listing) + "\n", encoding="utf-8")
    print("\n".join(listing))


if __name__ == "__main__":
    main()
