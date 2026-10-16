#!/usr/bin/env python3
"""
Dark History studio (v2) - one run makes and uploads:
  * 1 long documentary (16:9, ~10-12 min, Urdu narration, English title)
  * 2 YouTube Shorts (9:16) cut from the strongest moments of that video

Pipeline: researched story -> Urdu narration (Google Chirp HD voice) -> visuals
(Pexels stock video, AI cinematic images via Cloudflare, archive photos) ->
animated edit (camera moves, light flicker, dust, grain) -> music with ducking
-> thumbnail -> YouTube upload (Shorts scheduled later in the day).

Secrets (GitHub -> Settings -> Secrets -> Actions):
  GROQ_API_KEY or GEMINI_API_KEY   AI writer
  GOOGLE_TTS_KEY                   Google Text-to-Speech API key (Urdu voice)
  PEXELS_API_KEY / PIXABAY_API_KEY free stock videos (optional but recommended)
  CF_ACCOUNT_ID + CF_API_TOKEN     Cloudflare Workers AI images (optional)
  YT_CLIENT_ID / YT_CLIENT_SECRET / YT_REFRESH_TOKEN   YouTube upload
Variables (optional): CHANNEL_NAME, PRIVACY, TTS_VOICE, LONG_MINUTES, SHORTS_COUNT

Usage:
  python agent/studio.py                 # full run
  python agent/studio.py --offline --script samples/x.json --no-upload   # local test
"""
import argparse
import base64
import datetime as dt
import json
import math
import os
import random
import re
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "out"
MUSIC_DIR = ROOT / "assets" / "music"
UA = ("DarkHistoryStudio/2.1 (https://github.com/skhan750029-ui/dark-history-agent; "
      "educational history channel) python-urllib")

CHANNEL = os.getenv("CHANNEL_NAME", "Dark History")
PRIVACY = (os.getenv("PRIVACY") or "public").strip().lower()
LONG_MINUTES = float(os.getenv("LONG_MINUTES") or 11)
SHORTS_COUNT = int(os.getenv("SHORTS_COUNT") or 2)
TTS_VOICES = [v for v in [os.getenv("TTS_VOICE"), "ur-IN-Chirp3-HD-Charon", "ur-IN-Chirp3-HD-Orus",
                          "ur-IN-Chirp3-HD-Fenrir", "ur-IN-Wavenet-B", "ur-IN-Standard-B"] if v]
W, H, FPS = 1920, 1080, 30

FONT_LATIN = ["/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
URDU_FONT_NAME = "Noto Naskh Arabic"


def log(*a):
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}]", *a, flush=True)


def font(size):
    for p in FONT_LATIN:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def run(cmd, timeout=3600):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(map(str, cmd[:8]))}...\n{r.stderr[-2500:]}")
    return r.stdout


def duration(path):
    return float(run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                      "-of", "default=nw=1:nk=1", path]).strip())


def http(url, data=None, headers=None, method=None, timeout=120, raw=False):
    h = {"User-Agent": UA}
    h.update(headers or {})
    if isinstance(data, (dict, list)):
        data = json.dumps(data).encode()
        h.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
        if raw:
            return body, r.headers
        return json.loads(body) if body else {}


def slugify(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:60] or "video"


# =============================================================== 1. STORY (AI writer)

def groq(prompt, json_only=False, search=False, max_tokens=8000):
    key = os.environ["GROQ_API_KEY"]
    models = [m for m in [os.getenv("GROQ_MODEL"), "openai/gpt-oss-120b", "openai/gpt-oss-20b",
                          "moonshotai/kimi-k2-instruct-0905", "llama-3.3-70b-versatile"] if m]
    last = None
    for model in models:
        opts = [True, False] if (search and "gpt-oss" in model) else [False]
        for use_search in opts:
            body = {"model": model, "temperature": 0.8, "max_completion_tokens": max_tokens,
                    "messages": [{"role": "user", "content": prompt}]}
            if use_search:
                body["tools"] = [{"type": "browser_search"}]
            elif json_only:
                body["response_format"] = {"type": "json_object"}
            for attempt in range(4):
                try:
                    r = http("https://api.groq.com/openai/v1/chat/completions", body,
                             headers={"Authorization": f"Bearer {key}"}, timeout=300)
                    txt = r["choices"][0]["message"]["content"] or ""
                    if txt.strip():
                        log(f"AI writer ok ({model}, search={use_search})")
                        return txt
                    break
                except urllib.error.HTTPError as e:
                    last = e
                    log(f"Groq {model} search={use_search}: {e}")
                    if e.code == 429:
                        time.sleep(20 * (attempt + 1))
                        continue
                    break
                except Exception as e:
                    last = e
                    log(f"Groq {model}: {str(e)[:150]}")
                    break
    raise RuntimeError(f"AI writer failed: {last}")


def gemini(prompt, search=False):
    key = os.environ["GEMINI_API_KEY"]
    for model in [m for m in [os.getenv("GEMINI_MODEL"), "gemini-2.5-flash", "gemini-flash-latest"] if m]:
        body = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.8}}
        if search:
            body["tools"] = [{"google_search": {}}]
        try:
            r = http(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",
                     body, timeout=300)
            txt = "".join(p.get("text", "") for p in r["candidates"][0]["content"]["parts"])
            if txt.strip():
                return txt
        except Exception as e:
            log(f"Gemini {model}: {str(e)[:150]}")
    raise RuntimeError("Gemini failed")


def llm(prompt, search=False, json_only=False):
    if os.getenv("GROQ_API_KEY"):
        try:
            return groq(prompt, json_only=json_only, search=search)
        except Exception as e:
            if not os.getenv("GEMINI_API_KEY"):
                raise
            log(f"Groq failed, trying Gemini: {e}")
    if os.getenv("GEMINI_API_KEY"):
        return gemini(prompt, search=search)
    raise RuntimeError("Add GROQ_API_KEY or GEMINI_API_KEY in GitHub secrets")


def extract_json(txt):
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", txt, re.S)
    s = m.group(1) if m else txt[txt.find("{"): txt.rfind("}") + 1]
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        s2 = re.sub(r",\s*([}\]])", r"\1", s)  # trailing commas
        return json.loads(s2)


def ask_json(prompt, search=False, keys=()):
    for attempt in range(3):
        raw = llm(prompt, search=search)
        try:
            obj = extract_json(raw)
        except Exception as e:
            log(f"JSON broken ({e}), repairing")
            fix = ("Return the following content as ONE strictly valid JSON object. Keep every value and "
                   "all wording exactly. Output only JSON.\n\n" + raw)
            try:
                obj = extract_json(llm(fix, json_only=True))
            except Exception as e2:
                log(f"Repair failed: {e2}")
                continue
        if all(k in obj for k in keys):
            return obj
        log(f"JSON missing keys {keys}, retry {attempt + 1}")
    raise RuntimeError("AI writer did not return usable JSON")


def used_topics():
    f = DATA / "topics_done.txt"
    return [l.strip() for l in f.read_text(encoding="utf-8").splitlines() if l.strip()] if f.exists() else []


VISUAL_RULES = """The picture on screen MUST show exactly what the narration of that scene says,
in the correct era and place of the story. For every scene choose the visual:
- "type": "image" (a cinematic AI picture of the exact moment: people in period clothes,
  ships, rooms, letters, events - write it like a film still) - use this for about 75% of scenes,
  "archive" (a real historical photo/painting/map of a named person, place or object) - about 15%,
  "video" (real stock footage) - ONLY about 10%, and ONLY for timeless nature/atmosphere that fits
  the era (stormy sea, waves, fog, rain, fire, candle, desert, forest at night, clouds).
  NEVER video of modern things (cities, lights, cars, concerts, phones, offices, modern people).
  Never two "archive" in a row.
- "prompt": ALWAYS write it, for every scene (also video/archive scenes, as a backup):
  English prompt for the AI picture showing this scene's exact content: subject, era/year, place,
  action, clothing, lighting, camera angle. Photorealistic cinematic film still, moody low-key
  lighting, 35mm. No text, no logos, no gore, no real celebrities' faces in close-up.
- "query": 2-4 simple English words for stock-footage search (e.g. "stormy ocean night").
- "archive": 2-5 English words to search Wikimedia Commons (names, places, years)."""

SFX_RULES = """Sound design for every scene (like a real documentary editor):
- "sfx": the real background sound of what is on screen, 1-3 simple English words, e.g.
  "steam train", "snow wind", "heavy rain", "thunder storm", "ocean waves", "crackling fire",
  "clock ticking", "heartbeat", "footsteps wood", "door creak", "crowd murmur", "horse carriage",
  "church bells", "forest night", "dark drone". Use "none" only when silence is more powerful.
- "hit": true ONLY for a shocking reveal / twist / scary moment (about 1 in 6 scenes), else false."""

MODERN = {"city", "cities", "urban", "skyline", "concert", "neon", "party", "club", "car", "cars",
          "traffic", "phone", "smartphone", "computer", "laptop", "office", "business", "technology",
          "drone", "highway", "skyscraper", "festival", "disco", "led", "stage", "selfie", "airplane",
          "aircraft", "bridge", "nightlife", "lights", "fashion", "model", "woman", "man", "girl", "boy"}


def looks_modern(tags):
    words = set(re.split(r"[^a-z]+", (tags or "").lower()))
    return bool(words & MODERN)


def write_story(minutes):
    done = used_topics()
    avoid = "\n".join(f"- {t}" for t in done[-300:]) or "- (none yet)"
    n_ch = max(4, round(minutes / 2.3))
    outline = ask_json(f"""You are the head writer of "{CHANNEL}", a faceless YouTube channel that tells
true, little-known, dark and mysterious stories from real history - in the style of the best
Hindi/Urdu storytelling documentaries (gripping narrator, suspense, cliffhangers, a final twist).

Pick ONE real, specific, verifiable dark or mysterious historical story (curses, disappearances,
disasters, cover-ups, strange deaths, forgotten tragedies, bizarre experiments, lost places).
It must NOT be any of these already-used topics:
{avoid}
Use web search to verify names, dates, numbers. Never invent facts; mark legends as legends.
Respect victims: no gore for shock, no glorifying violence.

Plan a {minutes:.0f}-minute documentary with {n_ch} chapters. Return ONLY JSON:
{{"topic": "short English name of the story",
  "title": "English YouTube title, max 70 chars, curiosity + truthful, may use one CAPS word",
  "thumbnail_text": "2-4 English words, punchy",
  "setting": "visual setting for all pictures: year/era, place, period details (e.g. 1909, steamship era, stormy Indian Ocean off South Africa, Edwardian clothing)",
  "thumbnail_prompt": "English prompt for a dramatic photorealistic cinematic thumbnail image of the story's key moment, no text",
  "description": "English description: 2 short hook paragraphs + 'Sources:' with 2-4 real sources",
  "tags": ["12-15 English SEO tags"],
  "hashtags": ["#history", "#mystery", "..."],
  "chapters": [{{"title": "short English chapter name", "summary": "3-4 sentences of the verified facts this chapter tells"}}],
  "shorts": [{{"chapter": 0, "title": "English Shorts title max 60 chars"}}, {{"chapter": 2, "title": "..."}}]
}}""", search=True, keys=("topic", "title", "chapters"))
    if outline["topic"].strip().lower() in (t.lower() for t in done):
        raise RuntimeError("repeated topic")
    log(f"Topic: {outline['topic']} | {outline['title']}")

    words_per_ch = int(minutes * 135 / len(outline["chapters"]))
    scenes, previous = [], ""
    for i, ch in enumerate(outline["chapters"]):
        first = i == 0
        last = i == len(outline["chapters"]) - 1
        part = ask_json(f"""You write the Urdu narration for chapter {i + 1} of {len(outline['chapters'])} of a
YouTube documentary "{outline['title']}" (story: {outline['topic']}).
Chapter: {ch['title']} - {ch['summary']}
Story so far (last lines): {previous[-600:] or '(start of the video)'}

Write about {words_per_ch} words of narration in natural spoken Urdu (Urdu script, simple
everyday Urdu like a popular storyteller; English words only for names/places/terms).
Style: suspenseful, emotional, short sentences, questions to the viewer, cliffhanger at the end
of the chapter. {'Start with a powerful 2-sentence hook, then a quick "is video mein..." promise.' if first else 'Continue smoothly, do not greet again.'}
{'End with a chilling final thought, then ask viewers to like, comment and subscribe to ' + CHANNEL + '.' if last else ''}
Write it to be PERFORMED, not read: use "..." for dramatic pauses, "!" for shock, "?" for questions
to the viewer, and words of emotion (achanak, khaufnak, yaqeen nahi aata...).
Split into scenes of 1-2 sentences (8-14 seconds each when spoken).
{VISUAL_RULES}
{SFX_RULES}
Return ONLY JSON: {{"scenes": [{{"ur": "Urdu narration", "type": "video|image|archive",
"query": "...", "prompt": "...", "archive": "...", "sfx": "...", "hit": false}}]}}""", keys=("scenes",))
        for sc in part["scenes"]:
            if sc.get("ur", "").strip():
                sc["chapter"] = i
                scenes.append(sc)
        previous = " ".join(s["ur"] for s in part["scenes"][-3:])
        log(f"Chapter {i + 1}: {len(part['scenes'])} scenes")
        time.sleep(4)
    outline["scenes"] = scenes
    return outline


# =============================================================== 2. VOICE (Google TTS)

GEMINI_TTS = {"on": bool(os.getenv("GEMINI_API_KEY")) and (os.getenv("TTS_ENGINE") or "gemini") == "gemini",
              "fails": 0}
VOICE_STYLE = ("You are a famous Urdu dark-history and true-crime storyteller on YouTube. Narrate this in "
               "natural, fluent Urdu with real emotion and suspense, at a confident natural pace (never slow, "
               "never like reading from paper): dramatic pauses at '...', a lower, almost whispering voice for "
               "scary parts, shock and surprise at '!', and a direct questioning tone when asking the viewer.")


def gemini_tts(text, path):
    """Expressive Gemini voice (AI Studio key). Returns False to fall back to the normal Google voice."""
    key = os.getenv("GEMINI_API_KEY")
    voice = os.getenv("GEMINI_VOICE") or "Charon"
    for model in [os.getenv("GEMINI_TTS_MODEL") or "gemini-2.5-flash-preview-tts", "gemini-2.5-pro-preview-tts"]:
        body = {"contents": [{"parts": [{"text": VOICE_STYLE + "\n\nText:\n" + text}]}],
                "generationConfig": {"responseModalities": ["AUDIO"], "speechConfig": {
                    "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}}}}
        for attempt in range(5):
            try:
                r = http(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",
                         body, timeout=180)
                data = r["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]
                pcm = Path(str(path) + ".pcm")
                pcm.write_bytes(base64.b64decode(data))
                run(["ffmpeg", "-y", "-f", "s16le", "-ar", "24000", "-ac", "1", "-i", pcm,
                     "-af", "silenceremove=start_periods=1:start_threshold=-50dB,areverse,"
                            "silenceremove=start_periods=1:start_threshold=-50dB,areverse",
                     "-b:a", "160k", path])
                pcm.unlink(missing_ok=True)
                GEMINI_TTS["used"] = True
                return True
            except urllib.error.HTTPError as e:
                msg = e.read()[:200]
                if e.code in (429, 500, 503):
                    time.sleep(8 * (attempt + 1))
                    continue
                log(f"Gemini voice {model}: HTTP {e.code} {msg!r}")
                break
            except Exception as e:
                log(f"Gemini voice {model}: {e}")
                time.sleep(3)
    if GEMINI_TTS.get("used"):  # never mix two different voices inside one video
        log("Gemini voice failed mid-video, waiting and retrying once more")
        time.sleep(90)
        if GEMINI_TTS.get("retry_done"):
            raise RuntimeError("Gemini voice stopped working in the middle of the video")
        GEMINI_TTS["retry_done"] = True
        return gemini_tts(text, path)
    GEMINI_TTS["on"] = False
    log("Gemini voice not available - using normal Google voice")
    return False


def tts(text, path, offline=False):
    if offline:
        sec = max(2.5, len(text) / 14)
        run(["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=f=180:r=24000", "-af", "volume=0.05",
             "-t", f"{sec:.2f}", path])
        return
    if GEMINI_TTS["on"] and gemini_tts(text, path):
        return
    key = os.environ["GOOGLE_TTS_KEY"]
    rate = float(os.getenv("TTS_RATE") or 1.08)
    last = None
    for voice in list(TTS_VOICES):
        body = {"input": {"text": text},
                "voice": {"languageCode": "-".join(voice.split("-")[:2]), "name": voice},
                "audioConfig": {"audioEncoding": "MP3", "speakingRate": rate, "sampleRateHertz": 24000}}
        for attempt in range(3):
            try:
                r = http(f"https://texttospeech.googleapis.com/v1/text:synthesize?key={key}", body, timeout=120)
                Path(path).write_bytes(base64.b64decode(r["audioContent"]))
                if voice != TTS_VOICES[0]:  # remember a working voice for the rest of the video
                    TTS_VOICES.remove(voice)
                    TTS_VOICES.insert(0, voice)
                return
            except urllib.error.HTTPError as e:
                last = f"{voice}: {e.code} {e.read()[:300]!r}"
                if e.code in (429, 500, 503):
                    time.sleep(5 * (attempt + 1))
                    continue
                break
            except Exception as e:
                last = f"{voice}: {e}"
                time.sleep(3)
        log(f"TTS voice failed -> {last}")
    raise RuntimeError(f"Google TTS failed: {last}")


# =============================================================== 3. VISUALS

OK_LICENSES = ("public domain", "pd", "cc0", "cc by", "cc-by", "no restrictions")


def commons(query, used):
    q = urllib.parse.quote(f"{query} filetype:bitmap")
    url = ("https://commons.wikimedia.org/w/api.php?action=query&format=json&generator=search"
           f"&gsrsearch={q}&gsrnamespace=6&gsrlimit=12&prop=imageinfo&iiprop=url|size|extmetadata&iiurlwidth=2400")
    pages = None
    for attempt in range(4):
        try:
            time.sleep(1.2)  # be polite to Wikimedia (avoids HTTP 429)
            pages = http(url, timeout=40).get("query", {}).get("pages", {})
            break
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(8 * (attempt + 1))
                continue
            log(f"Commons failed '{query}': {e}")
            return None
        except Exception as e:
            log(f"Commons failed '{query}': {e}")
            return None
    if pages is None:
        return None
    for p in sorted(pages.values(), key=lambda p: p.get("index", 99)):
        ii = (p.get("imageinfo") or [{}])[0]
        if ii.get("width", 0) < 800 or ii.get("height", 0) < 500:
            continue
        meta = ii.get("extmetadata", {})
        lic = meta.get("LicenseShortName", {}).get("value", "")
        if not any(k in lic.lower() for k in OK_LICENSES) or "-nd" in lic.lower() or " nc" in lic.lower():
            continue
        src = ii.get("thumburl") or ii.get("url")
        if not src or src in used or src.lower().endswith((".svg", ".gif", ".tif", ".tiff")):
            continue
        artist = re.sub(r"<[^>]+>", "", meta.get("Artist", {}).get("value", ""))[:60].strip() or "Unknown"
        return src, f"{p['title'].replace('File:', '')} - {artist} ({lic}), Wikimedia Commons"
    return None


def pexels_video(query, used):
    key = os.getenv("PEXELS_API_KEY")
    if not key:
        return None
    url = ("https://api.pexels.com/videos/search?orientation=landscape&size=medium&per_page=15&query="
           + urllib.parse.quote(query))
    try:
        vids = http(url, headers={"Authorization": key}, timeout=40).get("videos", [])
    except Exception as e:
        log(f"Pexels failed '{query}': {e}")
        return None
    for v in vids:
        if v["id"] in used or v.get("duration", 0) < 4:
            continue
        files = [f for f in v.get("video_files", []) if f.get("width") and 1280 <= f["width"] <= 2560
                 and f.get("file_type") == "video/mp4"]
        if not files:
            continue
        f = sorted(files, key=lambda f: f["width"])[0]
        return v["id"], f["link"], f"Video by {v.get('user', {}).get('name', 'Pexels')} on Pexels"
    return None


def pixabay_video(query, used):
    key = os.getenv("PIXABAY_API_KEY")
    if not key:
        return None
    url = (f"https://pixabay.com/api/videos/?key={key}&safesearch=true&per_page=20&q="
           + urllib.parse.quote(query[:100]))
    try:
        hits = http(url, timeout=40).get("hits", [])
    except Exception as e:
        log(f"Pixabay failed '{query}': {e}")
        return None
    for v in hits:
        vid = f"pb{v['id']}"
        if vid in used or v.get("duration", 0) < 4 or looks_modern(v.get("tags")):
            continue
        files = v.get("videos", {})
        for size in ("medium", "large", "small"):
            f = files.get(size) or {}
            if f.get("url") and 1200 <= (f.get("width") or 0) <= 2600:
                return vid, f["url"], f"Video by {v.get('user', 'Pixabay')} on Pixabay"
    return None


def pixabay_image(query, used):
    key = os.getenv("PIXABAY_API_KEY")
    if not key:
        return None
    url = (f"https://pixabay.com/api/?key={key}&safesearch=true&per_page=20&orientation=horizontal"
           f"&min_width=1600&q=" + urllib.parse.quote(query[:100]))
    try:
        hits = http(url, timeout=40).get("hits", [])
    except Exception as e:
        log(f"Pixabay image failed '{query}': {e}")
        return None
    for h in hits:
        src = h.get("largeImageURL")
        if src and src not in used and not looks_modern(h.get("tags")):
            return src, f"Image by {h.get('user', 'Pixabay')} on Pixabay"
    return None


def stock_video(query, used):
    q2 = " ".join(query.split()[:2])
    return (pexels_video(query, used) or pixabay_video(query, used)
            or pexels_video(q2, used) or pixabay_video(q2, used))


SETTING = ""


def ai_image(prompt, path, seed):
    acc, tok = os.getenv("CF_ACCOUNT_ID"), os.getenv("CF_API_TOKEN")
    if not (acc and tok):
        return False
    full = (prompt + (f". Setting: {SETTING}" if SETTING else "") + ", photorealistic cinematic film still, dramatic moody lighting, 35mm film, "
            "high detail, muted colors, 16:9 composition")[:2000]
    for model in [os.getenv("CF_IMAGE_MODEL") or "@cf/black-forest-labs/flux-1-schnell"]:
        for attempt in range(3):
            try:
                r = http(f"https://api.cloudflare.com/client/v4/accounts/{acc}/ai/run/{model}",
                         {"prompt": full, "steps": 6},
                         headers={"Authorization": f"Bearer {tok}"}, timeout=120)
                img = r.get("result", {}).get("image")
                if img:
                    Path(path).write_bytes(base64.b64decode(img))
                    Image.open(path).convert("RGB").save(path, quality=95)
                    return True
            except urllib.error.HTTPError as e:
                log(f"AI image HTTP {e.code}: {e.read()[:200]!r}")
                if e.code == 429:
                    time.sleep(10 * (attempt + 1))
                    continue
                return False
            except Exception as e:
                log(f"AI image failed: {e}")
                time.sleep(3)
    return False


def placeholder(path, label, seed):
    rnd = random.Random(seed)
    img = Image.new("RGB", (1600, 900), (14, 12, 11))
    d = ImageDraw.Draw(img)
    for _ in range(45):
        x, y, r = rnd.randint(0, 1600), rnd.randint(0, 900), rnd.randint(80, 420)
        c = rnd.randint(25, 75)
        d.ellipse([x - r, y - r, x + r, y + r], fill=(c + 18, c + 4, c - 8))
    img = img.filter(ImageFilter.GaussianBlur(55))
    d = ImageDraw.Draw(img)
    for i, line in enumerate(textwrap.wrap(label.upper(), 26)[:3]):
        d.text((800, 420 + i * 70), line, font=font(56), fill=(210, 200, 180), anchor="mm")
    img.save(path, quality=92)


def download(url, path):
    body, _ = http(url, raw=True, timeout=180)
    Path(path).write_bytes(body)


def gather_visuals(scenes, work, offline):
    used_img, used_vid, credits = set(), set(), []
    ai_ok = 0
    for i, sc in enumerate(scenes):
        want = sc.get("type", "image")
        order = {"video": ["video", "image", "archive"], "image": ["image", "archive", "video"],
                 "archive": ["archive", "image", "video"]}.get(want, ["image", "archive", "video"])
        if not sc.get("prompt"):
            sc["prompt"] = sc.get("query") or sc.get("archive") or ""
        sc["asset"], sc["kind"] = None, None
        if not offline:
            for kind in order:
                try:
                    if kind == "video" and sc.get("query"):
                        got = stock_video(sc["query"], used_vid)
                        if got:
                            p = work / f"v_{i:03d}.mp4"
                            download(got[1], p)
                            used_vid.add(got[0])
                            credits.append(got[2])
                            sc["asset"], sc["kind"] = p, "video"
                    elif kind == "image" and sc.get("prompt"):
                        p = work / f"i_{i:03d}.jpg"
                        if ai_image(sc["prompt"], p, 1000 + i):
                            sc["asset"], sc["kind"] = p, "image"
                            ai_ok += 1
                        elif sc.get("query"):
                            got = pixabay_image(sc["query"], used_img) or pixabay_image(" ".join(sc["query"].split()[:2]), used_img)
                            if got:
                                download(got[0], p)
                                Image.open(p).convert("RGB").save(p, quality=94)
                                used_img.add(got[0])
                                credits.append(got[1])
                                sc["asset"], sc["kind"] = p, "image"
                    elif kind == "archive":
                        q = sc.get("archive") or sc.get("query", "")
                        got = commons(q, used_img) if q else None
                        if got:
                            p = work / f"a_{i:03d}.jpg"
                            download(got[0], p)
                            Image.open(p).convert("RGB").save(p, quality=94)
                            used_img.add(got[0])
                            credits.append(got[1])
                            sc["asset"], sc["kind"] = p, "image"
                except Exception as e:
                    log(f"Scene {i} {kind} failed: {e}")
                if sc["asset"]:
                    break
        if not sc["asset"] and offline:
            p = work / f"p_{i:03d}.jpg"
            placeholder(p, sc.get("query") or sc.get("archive") or "history", i)
            sc["asset"], sc["kind"] = p, "image"
        if i % 10 == 0:
            log(f"Visuals {i + 1}/{len(scenes)}")
    real = [s for s in scenes if s.get("asset")]
    log(f"Real visuals found for {len(real)}/{len(scenes)} scenes (AI pictures: {ai_ok})")
    if not offline and len(real) < max(5, 0.5 * len(scenes)):
        raise RuntimeError(f"Only {len(real)} real visuals found - refusing to make a low-quality video. "
                           "Add PIXABAY_API_KEY / CF_ACCOUNT_ID + CF_API_TOKEN secrets.")
    for i, sc in enumerate(scenes):  # fill gaps by re-using real footage (different crop/motion)
        if not sc.get("asset"):
            src = real[(i * 7) % len(real)]
            sc["asset"], sc["kind"] = src["asset"], src["kind"]
    return list(dict.fromkeys(credits))


# =============================================================== 4. EDIT

def grade_img(img):
    img = ImageOps.autocontrast(img.convert("RGB"), cutoff=1)
    img = ImageEnhance.Color(img).enhance(0.75)
    img = ImageEnhance.Contrast(img).enhance(1.08)
    tint = Image.new("RGB", img.size, (40, 52, 70))  # cool cinematic shadows
    return Image.blend(img, tint, 0.08)


def still_frame(src, out, w, h):
    img = grade_img(Image.open(src))
    ia, ta = img.width / img.height, w / h
    if max(ia, ta) / min(ia, ta) < 1.5:
        canvas = ImageOps.fit(img, (w * 2, h * 2), Image.LANCZOS, centering=(0.5, 0.42))
    else:  # very different aspect (e.g. portrait archive photo): blurred backdrop
        bg = ImageOps.fit(img, (w * 2, h * 2), Image.LANCZOS)
        bg = ImageEnhance.Brightness(bg.filter(ImageFilter.GaussianBlur(w // 25))).enhance(0.4)
        fg = ImageOps.contain(img, (int(w * 2 * 0.94), int(h * 2 * 0.94)), Image.LANCZOS)
        bg.paste(fg, ((w * 2 - fg.width) // 2, (h * 2 - fg.height) // 2))
        canvas = bg
    canvas = canvas.filter(ImageFilter.UnsharpMask(radius=2, percent=60, threshold=2))
    canvas.save(out, quality=94)


def animate_still(frame, dur, idx, out, w=W, h=H):
    """Ken Burns move + optional light flicker, so stills feel alive."""
    n = max(2, int(math.ceil(dur * FPS)))
    z1 = 1.14
    mode = idx % 6
    if mode == 0:
        z, x, y = f"1+{(z1 - 1) / n:.6f}*on", "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    elif mode == 1:
        z, x, y = f"{z1}-{(z1 - 1) / n:.6f}*on", "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    elif mode == 2:
        z, x, y = f"{z1}", f"(iw-iw/zoom)*on/{n}", "ih/2-(ih/zoom/2)"
    elif mode == 3:
        z, x, y = f"{z1}", f"(iw-iw/zoom)*(1-on/{n})", "ih/2-(ih/zoom/2)"
    elif mode == 4:
        z, x, y = f"1+{(z1 - 1) / n:.6f}*on", f"(iw-iw/zoom)*0.3", f"(ih-ih/zoom)*0.3"
    else:
        z, x, y = f"{z1}", "iw/2-(iw/zoom/2)", f"(ih-ih/zoom)*(1-on/{n})"
    vf = f"zoompan=z='{z}':x='{x}':y='{y}':d={n}:s={w}x{h}:fps={FPS}"
    if idx % 3 == 1:  # candle/lantern style light flicker
        vf += ",eq=brightness='0.035*sin(2*PI*t*1.3)+0.02*sin(2*PI*t*3.7)':eval=frame"
    vf += ",format=yuv420p"
    run(["ffmpeg", "-y", "-loop", "1", "-i", frame, "-vf", vf, "-frames:v", n,
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "19", "-an", out])


def video_clip(src, dur, out, w=W, h=H):
    vf = (f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},fps={FPS},"
          "eq=saturation=0.78:contrast=1.06:brightness=-0.02,format=yuv420p")
    run(["ffmpeg", "-y", "-stream_loop", "-1", "-i", src, "-t", f"{dur:.3f}", "-vf", vf, "-an",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", out])


def make_dust(path, seconds=8):
    """Floating dust specks layer (screen-blended over the whole video)."""
    run(["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=black:s={W}x{H}:r={FPS}:d={seconds}",
         "-vf", "noise=c0s=60:allf=t,lutyuv=y='if(gt(val,247),255,0)':u=128:v=128,"
                "boxblur=2:1,scroll=v=-0.0006:h=0.0002,format=yuv420p",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "24", path])


def make_score(total, path, seed):
    """Original dark cinematic bed (no copyright): minor-key pads, swells and low hits."""
    rnd = random.Random(seed)
    root = rnd.choice([110.0, 116.54, 123.47, 130.81, 98.0])
    m3, p5, oc = root * 1.189, root * 1.498, root * 2
    pad = (f"0.18*sin(2*PI*{root}*t)+0.14*sin(2*PI*{m3 * 2:.2f}*t)*(0.6+0.4*sin(2*PI*0.05*t))"
           f"+0.12*sin(2*PI*{p5 * 2:.2f}*t)*(0.6+0.4*sin(2*PI*0.037*t+1))"
           f"+0.07*sin(2*PI*{oc * 2:.2f}*t)*(0.5+0.5*sin(2*PI*0.11*t))"
           f"+0.05*sin(2*PI*{m3 * 4:.2f}*t)*(0.5+0.5*sin(2*PI*0.07*t+2))")
    hits = f"0.5*sin(2*PI*48*t)*exp(-3*mod(t,{rnd.choice([7.5, 8.0, 9.6])}))"
    d = total + 2
    run(["ffmpeg", "-y",
         "-f", "lavfi", "-i", f"aevalsrc='{pad}':s=44100:d={d:.2f}",
         "-f", "lavfi", "-i", f"aevalsrc='{hits}':s=44100:d={d:.2f}",
         "-f", "lavfi", "-i", f"anoisesrc=color=pink:amplitude=0.06:d={d:.2f}:r=44100",
         "-filter_complex",
         "[0]aecho=0.8:0.7:420|780:0.35|0.25,lowpass=f=3200[p];[1]lowpass=f=160[h];"
         "[2]bandpass=f=900:w=600,volume=0.5[n];[p][h][n]amix=inputs=3:normalize=0,"
         f"afade=t=in:d=3,afade=t=out:st={max(0, total - 4):.2f}:d=4,loudnorm=I=-24:TP=-3,aresample=44100[a]",
         "-map", "[a]", "-ac", "2", "-c:a", "pcm_s16le", path])


MACLEOD = ["Darkest Child", "Long Note Two", "Long Note Four", "Ghost Story", "Unseen Horrors",
           "Lightless Dawn", "Gathering Darkness", "Echoes of Time v2", "Dark Fog", "Ossuary 6 - Air",
           "Spacial Winds", "Static Motion", "Dark Walk", "Mistake the Getaway"]
MUSIC_CREDIT = []


def fetch_music(work, seed):
    """Download one dark royalty-free track (Kevin MacLeod, incompetech.com, CC BY 4.0)."""
    names = MACLEOD[:]
    random.Random(seed).shuffle(names)
    for name in names:
        url = ("https://incompetech.com/music/royalty-free/mp3-royaltyfree/"
               + urllib.parse.quote(name) + ".mp3")
        try:
            body, hd = http(url, raw=True, timeout=90,
                            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            if len(body) > 300_000:
                p = work / "track.mp3"
                p.write_bytes(body)
                MUSIC_CREDIT.append(f'Music: "{name}" Kevin MacLeod (incompetech.com) - '
                                    "Licensed under Creative Commons: By Attribution 4.0 License")
                return p
        except Exception as e:
            log(f"Music download failed '{name}': {e}")
    return None


def pick_music(total, work, seed):
    tracks = sorted(MUSIC_DIR.glob("*.mp3")) + sorted(MUSIC_DIR.glob("*.wav")) if MUSIC_DIR.exists() else []
    out = work / "music.wav"
    t = random.Random(seed).choice(tracks) if tracks else fetch_music(work, seed)
    if t:
        log(f"Music: {t.name} {MUSIC_CREDIT[-1:] }")
        run(["ffmpeg", "-y", "-stream_loop", "-1", "-i", t, "-t", f"{total + 1:.2f}",
             "-af", f"loudnorm=I=-24:TP=-3,aresample=44100,"
                    f"afade=t=in:d=2,afade=t=out:st={max(0, total - 4):.2f}:d=4",
             "-ac", "2", "-ar", "44100", out])
    else:
        log("Music: generated score (add tracks to assets/music for real music)")
        make_score(total, out, seed)
    return out


# ---------------------------------------------------------------- sound effects
SYNTH = {
    "wind": "anoisesrc=c=pink:a=0.6:r=44100,bandpass=f=550:w=500,tremolo=f=0.17:d=0.8,lowpass=f=1800,volume=1.6",
    "rain": "anoisesrc=c=white:a=0.35:r=44100,highpass=f=900,lowpass=f=7000,volume=0.8",
    "sea": "anoisesrc=c=brown:a=0.8:r=44100,lowpass=f=1100,tremolo=f=0.11:d=0.9,volume=1.4",
    "fire": "aevalsrc='0.25*(random(0)-0.5)*lt(random(1),0.02)+0.04*(random(2)-0.5)':s=44100,highpass=f=300,lowpass=f=6000",
    "heartbeat": "aevalsrc='0.9*sin(2*PI*48*t)*exp(-22*mod(t,0.95))+0.7*sin(2*PI*44*t)*exp(-22*mod(t-0.28,0.95))*gte(mod(t,0.95),0.28)':s=44100,lowpass=f=200",
    "clock": "aevalsrc='0.5*sin(2*PI*2400*t)*exp(-400*mod(t,1))+0.35*sin(2*PI*1900*t)*exp(-400*mod(t-0.5,1))*gte(mod(t,1),0.5)':s=44100",
    "train": "aevalsrc='0.6*sin(2*PI*70*t)*exp(-30*mod(t,0.62))+0.5*sin(2*PI*90*t)*exp(-30*mod(t-0.16,0.62))*gte(mod(t,0.62),0.16)+0.06*(random(0)-0.5)':s=44100,lowpass=f=900",
    "drone": "aevalsrc='0.35*sin(2*PI*55*t)+0.25*sin(2*PI*82.4*t)*(0.6+0.4*sin(2*PI*0.07*t))+0.12*sin(2*PI*110.3*t)*(0.5+0.5*sin(2*PI*0.05*t+1))':s=44100,aecho=0.8:0.7:500:0.4",
    "crowd": "anoisesrc=c=pink:a=0.5:r=44100,bandpass=f=700:w=900,tremolo=f=3.1:d=0.3,volume=0.9",
    "forest": "anoisesrc=c=pink:a=0.3:r=44100,bandpass=f=400:w=300,volume=0.7",
}
SFX_MAP = [("train", "train"), ("locomotive", "train"), ("rail", "train"), ("snow", "wind"), ("wind", "wind"),
           ("blizzard", "wind"), ("storm", "rain"), ("rain", "rain"), ("thunder", "rain"), ("sea", "sea"),
           ("ocean", "sea"), ("wave", "sea"), ("ship", "sea"), ("river", "sea"), ("fire", "fire"),
           ("burn", "fire"), ("candle", "fire"), ("heart", "heartbeat"), ("clock", "clock"), ("tick", "clock"),
           ("crowd", "crowd"), ("people", "crowd"), ("market", "crowd"), ("forest", "forest"),
           ("night", "forest"), ("drone", "drone"), ("dark", "drone"), ("tension", "drone")]
SFX_CACHE = {}


def freesound(query, out):
    key = os.getenv("FREESOUND_API_KEY")
    if not key:
        return False
    try:
        url = ("https://freesound.org/apiv2/search/text/?page_size=8&fields=id,duration,previews,license&query="
               + urllib.parse.quote(query) + "&filter=" + urllib.parse.quote('license:"Creative Commons 0" duration:[4 TO 90]')
               + "&token=" + key)
        for r in http(url, timeout=40).get("results", []):
            prev = (r.get("previews") or {}).get("preview-hq-mp3")
            if prev:
                download(prev, out)
                return True
    except Exception as e:
        log(f"Freesound failed '{query}': {e}")
    return False


def sfx_source(name, work):
    """Returns a looping-capable audio file for this sound name (cached), or None."""
    name = (name or "").strip().lower()
    if not name or name == "none":
        return None
    if name in SFX_CACHE:
        return SFX_CACHE[name]
    out = None
    p = work / f"fs_{slugify(name)}.mp3"
    if freesound(name, p):
        out = p
    else:
        kind = next((k for w, k in SFX_MAP if w in name), "drone")
        p = work / f"syn_{kind}.wav"
        if not p.exists():
            run(["ffmpeg", "-y", "-f", "lavfi", "-i", SYNTH[kind], "-t", "20", "-ac", "2", "-ar", "44100", p])
        out = p
    SFX_CACHE[name] = out
    return out


def make_hit(path):
    run(["ffmpeg", "-y", "-f", "lavfi", "-i",
         "aevalsrc='0.9*sin(2*PI*42*t)*exp(-2.2*t)+0.5*sin(2*PI*31*t)*exp(-1.5*t)+0.25*(random(0)-0.5)*exp(-6*t)':s=44100:d=3.5",
         "-af", "lowpass=f=400,aecho=0.8:0.8:90|170:0.4|0.3", "-ac", "2", path])


def make_whoosh(path):
    run(["ffmpeg", "-y", "-f", "lavfi", "-i", "anoisesrc=c=pink:a=0.9:r=44100:d=1.2",
         "-af", "bandpass=f=1400:w=1800,volume='sin(PI*t/1.2)^3':eval=frame,highpass=f=250", "-ac", "2", path])


def build_sfx(scenes, durs, work):
    """One background-sound track aligned with the narration: ambience per scene + hits + whooshes."""
    hit, whoosh = work / "hit.wav", work / "whoosh.wav"
    make_hit(hit)
    make_whoosh(whoosh)
    parts = []
    for i, (sc, du) in enumerate(zip(scenes, durs)):
        out = work / f"sfx_{i:03d}.wav"
        src = sfx_source(sc.get("sfx"), work)
        inputs, fc, mix = [], [], []
        if src:
            inputs += ["-stream_loop", "-1", "-i", src]
            fc.append(f"[0:a]atrim=0:{du:.3f},asetpts=PTS-STARTPTS,aresample=44100,"
                      f"loudnorm=I=-26:TP=-6,afade=t=in:d=0.6,afade=t=out:st={max(0, du - 0.6):.3f}:d=0.6[amb]")
            mix.append("[amb]")
        accents = []
        if sc.get("hit") in (True, "true", "True", 1):
            accents.append((hit, 0.9))
        if i > 0 and scenes[i - 1]["chapter"] != sc["chapter"]:
            accents.append((whoosh, 0.6))
        for k, (a, vol) in enumerate(accents):
            idx = len(inputs) // 2 if not src else 1 + k
            inputs += ["-i", a]
            fc.append(f"[{idx}:a]volume={vol}[x{k}]")
            mix.append(f"[x{k}]")
        if not mix:
            run(["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-t", f"{du:.3f}", out])
        else:
            graph = ";".join(fc) + ";" + "".join(mix) + (f"amix=inputs={len(mix)}:duration=longest:normalize=0,"
                                                         f"apad,atrim=0:{du:.3f}[o]")
            run(["ffmpeg", "-y", *inputs, "-filter_complex", graph, "-map", "[o]", "-ac", "2", "-ar", "44100", out])
        parts.append(out)
    (work / "sfx.txt").write_text("".join(f"file '{p.name}'\n" for p in parts))
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", work / "sfx.txt", "-c:a", "pcm_s16le", work / "sfx.wav"])
    used = sorted({str(sc.get("sfx")) for sc in scenes if sc.get("sfx") and sc.get("sfx") != "none"})
    log(f"Sound effects: {', '.join(used)[:300]}")
    return work / "sfx.wav"


def ass_time(t):
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"


def caption_file(items, path, w, h, size, margin):
    """items: list of (start, end, text). Urdu right-to-left handled by libass."""
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Ur,{URDU_FONT_NAME},{size},&H00FFFFFF,&H0000D7FF,&H00000000,&H90000000,-1,0,0,0,100,100,0,0,1,{max(3, size // 14)},2,2,70,70,{margin},1
Style: Top,DejaVu Sans,{int(size * 0.9)},&H0000D7FF,&H00FFFFFF,&H00000000,&H90000000,-1,0,0,0,100,100,1,0,1,{max(3, size // 12)},2,8,60,60,{int(h * 0.09)},1
"""
    lines = [f"Dialogue: 0,{ass_time(s)},{ass_time(e)},{st},,0,0,0,,{t}" for s, e, t, st in items]
    path.write_text(head + "\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
                    + "\n".join(lines) + "\n", encoding="utf-8")


def chunk_captions(scenes, starts, durs, t0=0.0, t1=1e9, words=4):
    items = []
    for sc, st, du in zip(scenes, starts, durs):
        if st + du <= t0 or st >= t1:
            continue
        ws = sc["ur"].split()
        chunks = [ws[i:i + words] for i in range(0, len(ws), words)]
        speak = max(0.5, du - 0.4)
        tot = sum(len(" ".join(c)) + 2 for c in chunks)
        t = st
        for c in chunks:
            d = speak * (len(" ".join(c)) + 2) / tot
            if t >= t0 and t < t1:
                items.append((t - t0, min(t + d, t1) - t0, " ".join(c), "Ur"))
            t += d
    return items


def render_long(story, work, offline):
    scenes = story["scenes"]
    log(f"Voice for {len(scenes)} scenes...")
    auds, durs = [], []
    for i, sc in enumerate(scenes):
        raw = work / f"vo_{i:03d}.mp3"
        tts(sc["ur"], raw, offline)
        wav = work / f"vo_{i:03d}.wav"
        pad = 0.7 if (i + 1 < len(scenes) and scenes[i + 1]["chapter"] != sc["chapter"]) else 0.12
        run(["ffmpeg", "-y", "-i", raw, "-af", f"apad=pad_dur={pad},aresample=44100", "-ac", "2", wav])
        auds.append(wav)
        durs.append(duration(wav))
    total = sum(durs)
    log(f"Narration: {total / 60:.1f} min")

    log("Visuals...")
    credits = gather_visuals(scenes, work, offline)

    log("Scene clips...")
    clips = []
    for i, (sc, du) in enumerate(zip(scenes, durs)):
        clip = work / f"c_{i:03d}.mp4"
        if sc["kind"] == "video":
            video_clip(sc["asset"], du, clip)
        else:
            fr = work / f"f_{i:03d}.jpg"
            still_frame(sc["asset"], fr, W, H)
            animate_still(fr, du, i, clip)
        clips.append(clip)
        if i % 10 == 0:
            log(f"Clip {i + 1}/{len(scenes)}")
    starts, t = [], 0.0
    for d in durs:
        starts.append(t)
        t += d

    (work / "clips.txt").write_text("".join(f"file '{c.name}'\n" for c in clips))
    (work / "auds.txt").write_text("".join(f"file '{a.name}'\n" for a in auds))
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", work / "clips.txt", "-c", "copy", work / "visual.mp4"])
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", work / "auds.txt", "-c:a", "pcm_s16le", work / "voice.wav"])
    music = pick_music(total, work, story["title"])
    sfx = build_sfx(scenes, durs, work)
    dust = work / "dust.mp4"
    make_dust(dust)

    log("Final long render...")
    final = work / "long.mp4"
    fc = (f"[0:v]setpts=PTS-STARTPTS[base];[1:v]setpts=PTS-STARTPTS[d];"
          f"[base][d]blend=c0_expr='min(255,A+0.4*B)':c1_expr='A':c2_expr='A',"
          f"vignette=PI/5,noise=alls=7:allf=t+u,fade=t=in:d=1,fade=t=out:st={max(0, total - 1.2):.2f}:d=1.2[v];"
          "[2:a]highpass=f=70,acompressor=threshold=-18dB:ratio=3:attack=5:release=150,"
          "loudnorm=I=-15:TP=-1.5:LRA=9,aresample=44100,asplit=3[vo][sc][sc2];"
          "[3:a]volume=0.8[mu];[mu][sc]sidechaincompress=threshold=0.06:ratio=3:attack=40:release=700[duck];"
          "[4:a]volume=1.0[fx];[fx][sc2]sidechaincompress=threshold=0.08:ratio=2:attack=40:release=500[fxd];"
          "[vo][duck][fxd]amix=inputs=3:duration=longest:normalize=0,alimiter=limit=0.95[a]")
    run(["ffmpeg", "-y", "-i", work / "visual.mp4", "-stream_loop", "-1", "-i", dust, "-i", work / "voice.wav",
         "-i", music, "-i", sfx, "-filter_complex", fc, "-map", "[v]", "-map", "[a]",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", "-r", FPS, "-t", f"{total:.2f}", "-movflags", "+faststart", final])
    return final, credits, starts, durs, total


def render_short(story, long_video, work, starts, durs, total, which, title):
    scenes = story["scenes"]
    ch = which.get("chapter", 0)
    idxs = [i for i, s in enumerate(scenes) if s["chapter"] == ch] or [0]
    s0 = starts[idxs[0]]
    end = s0
    for i in range(idxs[0], len(scenes)):
        if starts[i] + durs[i] - s0 > 57:
            break
        end = starts[i] + durs[i]
    if end - s0 < 25:  # chapter too short: extend in time
        end = min(total, s0 + 55)
    dur = end - s0
    tag = f"short{ch}"
    caps = chunk_captions(scenes, starts, durs, s0, end, words=3)
    caps.insert(0, (0, 3.2, title.upper(), "Top"))
    caps.append((max(0, dur - 2.6), dur, "FULL STORY ON THE CHANNEL", "Top"))
    ass = work / f"{tag}.ass"
    caption_file(caps, ass, 1080, 1920, 96, 430)
    out = work / f"{tag}.mp4"
    subs = str(ass).replace("\\", "/").replace(":", "\\:")
    vf = (f"scale=-2:1920,crop=1080:1920,eq=contrast=1.05,subtitles='{subs}',"
          f"fade=t=out:st={max(0, dur - 0.5):.2f}:d=0.5,format=yuv420p")
    run(["ffmpeg", "-y", "-ss", f"{s0:.2f}", "-i", long_video, "-t", f"{dur:.2f}", "-vf", vf,
         "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-c:a", "aac", "-b:a", "160k",
         "-af", f"afade=t=out:st={max(0, dur - 0.6):.2f}:d=0.6", "-movflags", "+faststart", out])
    log(f"Short ready: {out.name} ({dur:.1f}s)")
    return out


# =============================================================== 5. THUMBNAIL

def thumbnail(story, work, offline):
    Wt, Ht = 1280, 720
    base_path = work / "thumb_base.jpg"
    if offline or not ai_image(story.get("thumbnail_prompt", story["title"]), base_path, 7):
        cand = [s["asset"] for s in story["scenes"] if s.get("kind") == "image"]
        base_path = cand[0] if cand else None
    if base_path:
        base = ImageOps.fit(Image.open(base_path).convert("RGB"), (Wt, Ht), Image.LANCZOS, centering=(0.6, 0.45))
        base = ImageEnhance.Contrast(ImageEnhance.Color(base).enhance(1.1)).enhance(1.2)
    else:
        base = Image.new("RGB", (Wt, Ht), (15, 12, 10))
    grad = Image.new("L", (Wt, Ht))
    gd = ImageDraw.Draw(grad)
    for x in range(Wt):
        gd.line([(x, 0), (x, Ht)], fill=int(225 * max(0, 1 - x / (Wt * 0.7))))
    base = Image.composite(Image.new("RGB", (Wt, Ht), (0, 0, 0)), base, grad)
    d = ImageDraw.Draw(base)
    words = (story.get("thumbnail_text") or story["title"]).upper().split()[:5]
    lines, cur = [], ""
    for wd in words:
        if len(cur + " " + wd) > 11 and cur:
            lines.append(cur)
            cur = wd
        else:
            cur = (cur + " " + wd).strip()
    lines.append(cur)
    size = 132 if len(lines) <= 2 else 106
    y = Ht // 2 - len(lines) * (size + 12) // 2
    for i, line in enumerate(lines):
        d.text((54, y), line, font=font(size), fill=(255, 214, 0) if i == len(lines) - 1 else (255, 255, 255),
               stroke_width=9, stroke_fill=(0, 0, 0))
        y += size + 12
    d.rectangle([54, y + 6, 290, y + 22], fill=(205, 18, 30))
    out = work / "thumb.jpg"
    base.save(out, quality=92)
    return out


# =============================================================== 6. YOUTUBE

def yt_token():
    r = http("https://oauth2.googleapis.com/token", urllib.parse.urlencode({
        "client_id": os.environ["YT_CLIENT_ID"], "client_secret": os.environ["YT_CLIENT_SECRET"],
        "refresh_token": os.environ["YT_REFRESH_TOKEN"], "grant_type": "refresh_token"}).encode(),
        headers={"Content-Type": "application/x-www-form-urlencoded"})
    return r["access_token"]


def yt_upload(tok, video, title, desc, tags, publish_at=None, thumb=None):
    status = {"privacyStatus": PRIVACY, "selfDeclaredMadeForKids": False, "embeddable": True,
              "containsSyntheticMedia": True}  # AI voice + AI images: honest disclosure
    if publish_at and PRIVACY == "public":
        status["privacyStatus"] = "private"
        status["publishAt"] = publish_at
    meta = {"snippet": {"title": title[:100], "description": desc[:4900], "tags": tags[:15],
                        "categoryId": "27", "defaultLanguage": "en", "defaultAudioLanguage": "ur"},
            "status": status}
    size = Path(video).stat().st_size
    _, hdr = http("https://www.googleapis.com/upload/youtube/v3/videos?uploadType=resumable&part=snippet,status",
                  meta, headers={"Authorization": f"Bearer {tok}", "X-Upload-Content-Type": "video/mp4",
                                 "X-Upload-Content-Length": str(size)}, raw=True)
    req = urllib.request.Request(hdr["Location"], data=Path(video).read_bytes(), method="PUT",
                                 headers={"Authorization": f"Bearer {tok}", "Content-Type": "video/mp4",
                                          "User-Agent": UA})
    with urllib.request.urlopen(req, timeout=3600) as r:
        res = json.loads(r.read())
    vid = res["id"]
    log(f"Uploaded https://youtu.be/{vid} ({res['status'].get('privacyStatus')}"
        f"{', publishes ' + publish_at if publish_at else ''})")
    if thumb:
        try:
            http(f"https://www.googleapis.com/upload/youtube/v3/thumbnails/set?videoId={vid}&uploadType=media",
                 Path(thumb).read_bytes(), headers={"Authorization": f"Bearer {tok}", "Content-Type": "image/jpeg"})
            log("Thumbnail set")
        except Exception as e:
            log(f"Thumbnail not set (verify channel at youtube.com/verify): {e}")
    return vid, res["status"].get("privacyStatus")


def chapter_lines(story, starts):
    seen, out = set(), []
    for sc, st in zip(story["scenes"], starts):
        ch = sc["chapter"]
        if ch in seen:
            continue
        seen.add(ch)
        name = (story.get("chapters") or [{}] * (ch + 1))[ch].get("title", f"Part {ch + 1}") if ch < len(story.get("chapters") or []) else f"Part {ch + 1}"
        t = 0 if not out else int(st)
        out.append(f"{t // 60}:{t % 60:02d} {name}")
    return "Chapters:\n" + "\n".join(out) + "\n\n" if len(out) >= 3 else ""


def description(story, credits, extra=""):
    tags = " ".join(story.get("hashtags", [])[:4])
    cred = "\n".join(f"- {c}" for c in credits[:30])
    return (f"{story['description'].strip()}\n\n{extra}"
            f"Subscribe to {CHANNEL} for the stories history tried to forget.\n\n"
            f"Narration: Urdu. Some scenes are AI-generated illustrations of historical events.\n"
            f"Footage & images:\n{cred}\n" + "".join(c + "\n" for c in MUSIC_CREDIT) + f"\n{tags}")


# =============================================================== MAIN

def preview(files, out):
    try:
        ims = []
        for f in files:
            im = Image.open(f).convert("RGB")
            ims.append(im.resize((int(im.width * 400 / im.height), 400)))
        sheet = Image.new("RGB", (sum(i.width for i in ims) + 10 * (len(ims) + 1), 420), (20, 20, 20))
        x = 10
        for im in ims:
            sheet.paste(im, (x, 10))
            x += im.width + 10
        out.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(out, quality=80)
    except Exception as e:
        log(f"Preview skipped: {e}")


def grab(video, t, out):
    run(["ffmpeg", "-y", "-ss", f"{t:.2f}", "-i", video, "-frames:v", "1", out])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--script")
    a = ap.parse_args()
    DATA.mkdir(exist_ok=True)

    story = json.loads(Path(a.script).read_text(encoding="utf-8")) if a.script else None
    if not story:
        for attempt in range(2):
            try:
                story = write_story(LONG_MINUTES)
                break
            except Exception as e:
                log(f"Story attempt {attempt + 1} failed: {e}")
        if not story:
            raise RuntimeError("Could not write a story")

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M")
    work = OUT / f"{stamp}-{slugify(story['topic'])}"
    work.mkdir(parents=True, exist_ok=True)
    (work / "story.json").write_text(json.dumps(story, indent=1, ensure_ascii=False, default=str), encoding="utf-8")

    global SETTING
    SETTING = (story.get("setting") or "").strip()[:300]
    long_v, credits, starts, durs, total = render_long(story, work, a.offline)
    thumb = thumbnail(story, work, a.offline)
    shorts = []
    for k, sh in enumerate((story.get("shorts") or [{"chapter": 0}])[:SHORTS_COUNT]):
        title = sh.get("title") or story["title"]
        shorts.append((render_short(story, long_v, work, starts, durs, total, sh, title), title))

    frames = [grab(long_v, total * f, work / f"pv{n}.jpg") for n, f in enumerate((0.05, 0.3, 0.55, 0.8))]
    frames += [grab(s[0], 4, work / f"pvs{n}.jpg") for n, s in enumerate(shorts)] + [thumb]
    preview(frames, DATA / "previews" / "latest.jpg")
    log(f"Long video ready: {total / 60:.1f} min, {len(shorts)} shorts")

    results = []
    if not (a.offline or a.no_upload):
        tok = yt_token()
        vid, st = yt_upload(tok, long_v, story["title"], description(story, credits, chapter_lines(story, starts)),
                            story.get("tags", []), thumb=thumb)
        results.append(("long", story["title"], st, vid))
        now = dt.datetime.now(dt.timezone.utc)
        for k, (sv, stitle) in enumerate(shorts):
            when = (now + dt.timedelta(hours=4 + 5 * k)).replace(microsecond=0).isoformat().replace("+00:00", "Z")
            t = stitle if "#shorts" in stitle.lower() else f"{stitle} #shorts"
            extra = f"Full documentary: https://youtu.be/{vid}\n\n"
            try:
                svid, sst = yt_upload(tok, sv, t, description(story, credits, extra),
                                      story.get("tags", []), publish_at=when)
                results.append(("short", t, sst, svid))
            except Exception as e:
                log(f"Short upload failed: {e}")

    if not a.offline:
        with open(DATA / "topics_done.txt", "a", encoding="utf-8") as f:
            f.write(story["topic"].strip() + "\n")
        with open(DATA / "upload_log.csv", "a", encoding="utf-8") as f:
            for kind, t, st, vid in results:
                f.write(f"{dt.datetime.now(dt.timezone.utc).isoformat()},{kind},\"{t}\",{st},https://youtu.be/{vid}\n")
    keep = {"long.mp4", "thumb.jpg", "story.json"} | {s[0].name for s in shorts}
    for p in work.iterdir():
        if p.name not in keep:
            p.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FAILED: {e}")
        sys.exit(1)
