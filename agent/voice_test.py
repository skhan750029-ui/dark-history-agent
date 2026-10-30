#!/usr/bin/env python3
"""Makes ~20 second samples of 4 deep, cinematic English Google voices so we can pick one.
Each sample is a slightly slow read with short pauses between sentences, over a quiet dark music bed
and a few scene sound effects - the way it will sound in a video.
Results go to data/voice_samples/ (mp3 files + results.txt)."""
import base64
import json
import os
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "voice_samples"
OUT.mkdir(parents=True, exist_ok=True)
for old in OUT.glob("*.mp3"):
    old.unlink()
KEY = os.getenv("GOOGLE_TTS_KEY", "")
RATE = float(os.getenv("TTS_RATE") or 0.92)
LOG = []

SENTENCES = ["In 1980, America built a secret plane...",
             "A plane that could land inside a football stadium.",
             "Its mission was to rescue fifty-two hostages from the heart of Tehran.",
             "But on the night of its final test... something went terribly wrong.",
             "And for years, nobody was allowed to talk about it."]
TEXT = " ".join(SENTENCES)
MARKUP = " [pause short] ".join(SENTENCES)  # Chirp 3 HD pause tags; plain text is the fallback

# Deep male voices first; the first 4 that work become the samples.
VOICES = ["en-US-Chirp3-HD-Charon", "en-US-Chirp3-HD-Algenib", "en-US-Chirp3-HD-Enceladus",
          "en-US-Chirp3-HD-Iapetus", "en-GB-Chirp3-HD-Charon", "en-US-Chirp3-HD-Fenrir", "en-US-Chirp3-HD-Orus"]


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *map(str, args)], check=True)


def seconds(path):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                 str(path)], capture_output=True, text=True).stdout.strip() or 0)


def speak(voice, path):
    """Returns 'markup' or 'text' (what worked), or raises."""
    for kind, value in (("markup", MARKUP), ("text", TEXT)):
        body = {"input": {kind: value}, "voice": {"languageCode": "-".join(voice.split("-")[:2]), "name": voice},
                "audioConfig": {"audioEncoding": "MP3", "speakingRate": RATE, "sampleRateHertz": 24000}}
        try:
            r = post(f"https://texttospeech.googleapis.com/v1/text:synthesize?key={KEY}", body)
            path.write_bytes(base64.b64decode(r["audioContent"]))
            return kind
        except urllib.error.HTTPError as e:
            if kind == "markup" and e.code == 400:
                continue  # this voice/API version doesn't take pause tags
            raise RuntimeError(f"HTTP {e.code} {e.read()[:200]!r}")


def music(work):
    for name in ["Darkest Child", "Long Note Two", "Ghost Story", "Unseen Horrors"]:
        url = "https://incompetech.com/music/royalty-free/mp3-royaltyfree/" + urllib.parse.quote(name) + ".mp3"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            with urllib.request.urlopen(req, timeout=60) as r:
                b = r.read()
            if len(b) > 300_000:
                p = work / "music.mp3"
                p.write_bytes(b)
                LOG.append(f'MUSIC "{name}" Kevin MacLeod (incompetech.com), CC BY 4.0')
                return p
        except Exception as e:
            LOG.append(f"MUSIC FAIL {name}: {e}")
    return None


def mix(voice_mp3, track, out):
    """Voice up front, music ducked under it, wind ambience, a low hit on 'something went wrong'."""
    d = seconds(voice_mp3) + 1.5
    hit_at = d * 0.62
    inputs = ["-i", voice_mp3,
              "-f", "lavfi", "-t", f"{d:.2f}", "-i",
              "anoisesrc=c=pink:a=0.6:r=44100,bandpass=f=550:w=500,tremolo=f=0.17:d=0.8,lowpass=f=1800",
              "-f", "lavfi", "-i",
              "aevalsrc='0.9*sin(2*PI*42*t)*exp(-2.2*t)+0.5*sin(2*PI*31*t)*exp(-1.5*t)':s=44100:d=3.5"]
    fc = ("[0:a]aresample=44100,highpass=f=70,acompressor=threshold=-18dB:ratio=3:attack=5:release=150,"
          "loudnorm=I=-15:TP=-1.5,apad=pad_dur=1.5,asplit=2[v][sc];"
          "[1:a]volume=0.10,afade=t=in:d=1[wind];"
          f"[2:a]lowpass=f=400,volume=0.7,adelay={int(hit_at * 1000)}|{int(hit_at * 1000)}[hit];")
    mixin = "[v][wind][hit]"
    if track:
        inputs += ["-ss", "8", "-i", track]
        fc += ("[3:a]aresample=44100,loudnorm=I=-24:TP=-3,afade=t=in:d=1.5[m];"
               "[m][sc]sidechaincompress=threshold=0.06:ratio=3:attack=40:release=700[duck];")
        mixin += "[duck]"
    else:
        fc += "[sc]anullsink;"
    n = mixin.count("[")
    fc += f"{mixin}amix=inputs={n}:duration=first:normalize=0,afade=t=out:st={d - 1.2:.2f}:d=1.2,alimiter=limit=0.95"
    ffmpeg(*inputs, "-filter_complex", fc, "-ac", "2", "-b:a", "192k", out)


work = OUT / "_work"
work.mkdir(exist_ok=True)
track = music(work)
made = 0
for voice in VOICES:
    if made == 4:
        break
    raw = work / f"{voice}.mp3"
    try:
        how = speak(voice, raw)
        name = f"{made + 1}_{voice.split('-')[0]}-{voice.split('-')[1]}_{voice.split('-')[-1]}.mp3"
        mix(raw, track, OUT / name)
        made += 1
        LOG.append(f"OK   {name}  voice={voice} rate={RATE} pauses={how}  ({seconds(OUT / name):.1f} s)")
    except Exception as e:
        LOG.append(f"FAIL {voice}: {e}")
for p in work.iterdir():
    p.unlink()
work.rmdir()
(OUT / "results.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
print("\n".join(LOG))
