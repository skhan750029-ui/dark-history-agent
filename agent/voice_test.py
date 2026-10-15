#!/usr/bin/env python3
"""Makes short voice samples with different Google voices so we can pick the best one.
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
KEY = os.getenv("GOOGLE_TTS_KEY", "")
GKEY = os.getenv("GEMINI_API_KEY", "")
LOG = []

TEXT = ("سن انیس سو بائیس... جرمنی کا ایک چھوٹا سا، خاموش فارم۔ "
        "ایک رات، اس گھر کے چھ لوگ... ہمیشہ کے لیے خاموش ہو گئے۔ "
        "لیکن اصل خوف یہ نہیں تھا! اصل خوف یہ تھا کہ قاتل... کئی دن تک اسی گھر میں رہتا رہا۔ "
        "وہ گائے کو چارہ ڈالتا رہا، کھانا کھاتا رہا، اور گاؤں والوں کو کچھ پتا ہی نہیں چلا۔ "
        "سوچیے... آپ کے گھر کی چھت پر، رات کو، کوئی قدموں کی آواز آئے... تو آپ کیا کریں گے؟ "
        "آج کی کہانی... آپ کی نیند اڑا دے گی۔")

STYLE = ("You are a famous Urdu true-crime and dark-history storyteller on YouTube. "
         "Narrate with deep suspense and real emotion: natural, confident, fluent pace (not slow), "
         "dramatic pauses at '...', rising tension, a lower, almost whispering voice for the scary parts, "
         "energy and surprise at '!', and a direct questioning tone when asking the viewer.")


def post(url, body, headers=None):
    h = {"Content-Type": "application/json"}
    h.update(headers or {})
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=h)
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def save(name, audio_bytes, ext):
    raw = OUT / f"{name}.{ext}"
    raw.write_bytes(audio_bytes)
    mp3 = OUT / f"{name}.mp3"
    if ext != "mp3":
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "s16le", "-ar", "24000", "-ac", "1",
                        "-i", raw, "-b:a", "128k", mp3] if ext == "pcm" else
                       ["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-b:a", "128k", mp3], check=True)
        raw.unlink()
    d = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", mp3],
                       capture_output=True, text=True).stdout.strip()
    LOG.append(f"OK   {name}.mp3  ({d} s)")
    return mp3


def cloud(name, voice, lang, model=None, rate=None, prompt=None, version="v1"):
    body = {"input": {"text": TEXT}, "voice": {"languageCode": lang, "name": voice},
            "audioConfig": {"audioEncoding": "MP3"}}
    if model:
        body["voice"]["model_name"] = model
    if prompt:
        body["input"]["prompt"] = prompt
    if rate:
        body["audioConfig"]["speakingRate"] = rate
    try:
        r = post(f"https://texttospeech.googleapis.com/{version}/text:synthesize?key={KEY}", body)
        save(name, base64.b64decode(r["audioContent"]), "mp3")
    except urllib.error.HTTPError as e:
        LOG.append(f"FAIL {name}: HTTP {e.code} {e.read()[:300]!r}")
    except Exception as e:
        LOG.append(f"FAIL {name}: {e}")


def gemini_api(name, model, voice):
    if not GKEY:
        LOG.append(f"SKIP {name}: no GEMINI_API_KEY")
        return
    body = {"contents": [{"parts": [{"text": STYLE + "\n\nRead this:\n" + TEXT}]}],
            "generationConfig": {"responseModalities": ["AUDIO"],
                                 "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}}}}
    try:
        r = post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GKEY}", body)
        data = r["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]
        save(name, base64.b64decode(data), "pcm")
    except urllib.error.HTTPError as e:
        LOG.append(f"FAIL {name}: HTTP {e.code} {e.read()[:300]!r}")
    except Exception as e:
        LOG.append(f"FAIL {name}: {e}")


def music_test():
    for name in ["Darkest Child", "Long Note Two", "Ghost Story"]:
        url = "https://incompetech.com/music/royalty-free/mp3-royaltyfree/" + urllib.parse.quote(name) + ".mp3"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            with urllib.request.urlopen(req, timeout=60) as r:
                b = r.read()
            LOG.append(f"MUSIC {name}: {len(b)} bytes, type {r.headers.get('Content-Type')}")
            if len(b) > 300_000:
                (OUT / "music_test.mp3").write_bytes(b)
                return OUT / "music_test.mp3"
        except Exception as e:
            LOG.append(f"MUSIC FAIL {name}: {e}")
    return None


cloud("1_old_chirp_charon_slow", "ur-IN-Chirp3-HD-Charon", "ur-IN", rate=0.97)
cloud("2_chirp_charon_faster", "ur-IN-Chirp3-HD-Charon", "ur-IN", rate=1.12)
cloud("3_gemini_pro_urdu_charon", "Charon", "ur-PK", model="gemini-2.5-pro-tts", prompt=STYLE)
cloud("4_gemini_flash_urdu_orus", "Orus", "ur-PK", model="gemini-2.5-flash-tts", prompt=STYLE)
cloud("5_gemini_31_urdu_charon", "Charon", "ur-PK", model="gemini-3.1-flash-tts-preview", prompt=STYLE)
cloud("6_gemini_pro_hindi_charon", "Charon", "hi-IN", model="gemini-2.5-pro-tts", prompt=STYLE)
gemini_api("7_gemini_api_charon", "gemini-2.5-flash-preview-tts", "Charon")

track = music_test()
best = next((OUT / f"{n}.mp3" for n in ["3_gemini_pro_urdu_charon", "5_gemini_31_urdu_charon",
                                         "4_gemini_flash_urdu_orus", "2_chirp_charon_faster"]
             if (OUT / f"{n}.mp3").exists()), None)
if track and best:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", best, "-stream_loop", "-1", "-i", track,
                    "-filter_complex",
                    "[0:a]aresample=44100,loudnorm=I=-15:TP=-1.5,asplit=2[v][sc];"
                    "[1:a]aresample=44100,loudnorm=I=-20:TP=-3[m];"
                    "[m][sc]sidechaincompress=threshold=0.08:ratio=2.5:attack=40:release=700[d];"
                    "[v][d]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.95",
                    "-b:a", "160k", OUT / "8_best_voice_with_music.mp3"], check=False)
    LOG.append(f"MIX  8_best_voice_with_music.mp3 from {best.name}")
(OUT / "music_test.mp3").unlink(missing_ok=True)
(OUT / "results.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
print("\n".join(LOG))
