#!/usr/bin/env python3
"""
Dark History - faceless video agent.

One run = one video, start to finish:
  topic -> researched script -> narration -> archive images -> edit
  (Ken Burns, film grain, captions, ambient score) -> thumbnail
  -> title/description/tags -> YouTube upload -> log.

Usage:
  python agent/make_video.py --format short
  python agent/make_video.py --format long
  python agent/make_video.py --format short --offline   # local test, no internet
  python agent/make_video.py --format short --no-upload # make video, skip upload

Settings come from environment variables (GitHub Secrets):
  GEMINI_API_KEY      AI writer (free: aistudio.google.com/apikey)
  GROQ_API_KEY        or this AI writer instead (free: console.groq.com/keys)
  YT_CLIENT_ID        required for upload
  YT_CLIENT_SECRET    required for upload
  YT_REFRESH_TOKEN    required for upload
  PEXELS_API_KEY      optional  extra photo source
  ELEVENLABS_API_KEY  optional  more human voice (paid)
  ELEVENLABS_VOICE_ID optional
  CHANNEL_NAME        optional  default "Dark History"
  PRIVACY             optional  public | unlisted | private (default public)
  VOICE               optional  edge-tts voice (default en-US-ChristopherNeural)
"""
import argparse
import asyncio
import datetime as dt
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import textwrap
import time
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "out"
UA = "DarkHistoryAgent/1.0 (faceless educational channel; contact via GitHub)"

CHANNEL = os.getenv("CHANNEL_NAME", "Dark History")
PRIVACY = os.getenv("PRIVACY", "public").strip().lower() or "public"
VOICE = os.getenv("VOICE", "en-US-ChristopherNeural")

FORMATS = {
    "short": {"w": 1080, "h": 1920, "words": (115, 140), "max_sec": 58},
    "long": {"w": 1920, "h": 1080, "words": (750, 900), "max_sec": 900},
}

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "C:/Windows/Fonts/impact.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
]


def log(*a):
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}]", *a, flush=True)


def font(size):
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(map(str, cmd[:6]))}...\n{r.stderr[-2000:]}")
    return r.stdout


def duration(path):
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "default=nw=1:nk=1", str(path)])
    return float(out.strip())


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


# ---------------------------------------------------------------- 1. SCRIPT

def used_topics():
    f = DATA / "topics_done.txt"
    return [l.strip() for l in f.read_text(encoding="utf-8").splitlines() if l.strip()] if f.exists() else []


def groq(prompt):
    """Free alternative to Gemini (console.groq.com). groq/compound can search the web."""
    key = os.environ["GROQ_API_KEY"]
    models = [m for m in [os.getenv("GROQ_MODEL"), "groq/compound", "openai/gpt-oss-120b",
                          "llama-3.3-70b-versatile"] if m]
    last = None
    for model in models:
        try:
            r = http("https://api.groq.com/openai/v1/chat/completions",
                     {"model": model, "temperature": 0.9,
                      "messages": [{"role": "user", "content": prompt}]},
                     headers={"Authorization": f"Bearer {key}"}, timeout=240)
            txt = r["choices"][0]["message"]["content"] or ""
            if txt.strip():
                log(f"Groq ok ({model})")
                return txt
        except Exception as e:
            last = e
            log(f"Groq {model} failed: {str(e)[:150]}")
            time.sleep(3)
    raise RuntimeError(f"Groq failed: {last}")


def llm(prompt):
    if os.getenv("GEMINI_API_KEY"):
        try:
            return gemini(prompt)
        except Exception as e:
            if not os.getenv("GROQ_API_KEY"):
                raise
            log(f"Gemini unavailable, switching to Groq: {e}")
    if os.getenv("GROQ_API_KEY"):
        return groq(prompt)
    raise RuntimeError("Add a GEMINI_API_KEY or GROQ_API_KEY secret in GitHub")


def gemini(prompt, use_search=True):
    key = os.environ["GEMINI_API_KEY"]
    models = [m for m in [os.getenv("GEMINI_MODEL"), "gemini-2.5-flash",
                          "gemini-flash-latest", "gemini-2.0-flash"] if m]
    last = None
    for model in models:
        for search in ([True, False] if use_search else [False]):
            body = {"contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.9}}
            if search:
                body["tools"] = [{"google_search": {}}]
            url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
                   f"{model}:generateContent?key={key}")
            try:
                r = http(url, body, timeout=180)
                parts = r["candidates"][0]["content"]["parts"]
                txt = "".join(p.get("text", "") for p in parts)
                if txt.strip():
                    log(f"Gemini ok ({model}, search={search})")
                    return txt
            except Exception as e:  # try next option
                last = e
                log(f"Gemini {model} search={search} failed: {str(e)[:150]}")
                time.sleep(3)
    raise RuntimeError(f"Gemini failed: {last}")


def extract_json(txt):
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", txt, re.S)
    s = m.group(1) if m else txt[txt.find("{"): txt.rfind("}") + 1]
    return json.loads(s)


def write_script(fmt):
    lo, hi = FORMATS[fmt]["words"]
    done = used_topics()
    avoid = "\n".join(f"- {t}" for t in done[-300:]) or "- (none yet)"
    n_scenes = "8-10" if fmt == "short" else "40-50"
    kind = ("a YouTube Short (vertical, under 60 seconds)" if fmt == "short"
            else "a 5-6 minute YouTube documentary")
    prompt = f"""You are the head writer of "{CHANNEL}", a faceless YouTube channel that tells
true, little-known, disturbing stories from real history in the style of a gripping
documentary narrator. Write {kind}.

STEP 1 - Pick ONE real, specific, verifiable historical event, person or practice that is
dark, strange or shocking (plagues, disasters, cruel punishments, strange deaths, cover-ups,
forgotten tragedies, bizarre medicine, cursed places...). It must NOT be any of these
already-used topics:
{avoid}
Use web search to verify every name, date and number. Never invent facts. If a detail is
disputed, say "according to records" or leave it out. Be respectful to victims; no gore for
its own sake, no glorifying violence.

STEP 2 - Write the narration: {lo}-{hi} words total, English, spoken style, short punchy
sentences, present-tense storytelling where it helps.
- First sentence = a hook that creates instant curiosity (no "Hello", no channel intro).
- Build suspense, reveal details step by step, end with a chilling twist or a thought that
  lingers.
- Final line: a natural call to action, e.g. "Follow {CHANNEL} for the stories history
  tried to forget."
- Split into {n_scenes} scenes. Each scene = 1-3 sentences.
- For every scene give an "image_query": 2-6 words to search Wikimedia Commons for a REAL
  archival photo, painting, engraving, map or document matching that moment (names, places,
  years work best, e.g. "Joseph Bazalgette", "Thames 1858 cartoon Punch", "cholera London
  map 1854"). Also give "mood_query": 2-3 generic words for a fallback atmospheric photo
  (e.g. "foggy river night").

STEP 3 - Packaging:
- "title": max 65 characters, curiosity-driven, truthful, no clickbait lies, no ALL CAPS
  sentence (one capitalised word is ok).
- "thumbnail_text": 2-4 punchy words for the thumbnail, e.g. "LONDON COULDN'T BREATHE".
- "description": 3 short paragraphs: a hook, what the story covers, and 2-4 "Sources:"
  lines naming real books/sites you verified with. No links you are unsure exist.
- "tags": 12-15 SEO tags.
- "hashtags": 3-4 hashtags.
- "topic": the short name of the event (used to avoid repeats).

Return ONLY JSON inside a ```json code block with exactly this shape:
{{"topic": "...", "title": "...", "thumbnail_text": "...", "description": "...",
  "tags": ["..."], "hashtags": ["#history"],
  "scenes": [{{"text": "...", "image_query": "...", "mood_query": "..."}}]}}"""
    for attempt in range(3):
        try:
            s = extract_json(llm(prompt))
            words = sum(len(sc["text"].split()) for sc in s["scenes"])
            log(f"Script: '{s['title']}' ({words} words, {len(s['scenes'])} scenes)")
            if s["topic"].strip().lower() in (t.lower() for t in done):
                raise ValueError("repeated topic")
            if words < lo * 0.7:
                raise ValueError(f"too short: {words} words")
            return s
        except Exception as e:
            log(f"Script attempt {attempt + 1} failed: {e}")
    raise RuntimeError("Could not get a valid script from the AI writer")


# ---------------------------------------------------------------- 2. VOICE

async def _edge(text, path):
    import edge_tts
    c = edge_tts.Communicate(text, VOICE, rate="-4%", pitch="-3Hz")
    await c.save(str(path))


def tts(text, path, offline=False):
    if offline:  # silent placeholder of a realistic length
        sec = max(2.0, len(text.split()) / 2.4)
        run(["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
             "-t", f"{sec:.2f}", "-c:a", "libmp3lame", str(path)])
        return
    key = os.getenv("ELEVENLABS_API_KEY")
    if key:
        try:
            vid = os.getenv("ELEVENLABS_VOICE_ID", "onwK4e9ZLuTAKqWW03F9")  # "Daniel"
            body, _ = http(f"https://api.elevenlabs.io/v1/text-to-speech/{vid}",
                           {"text": text, "model_id": "eleven_multilingual_v2",
                            "voice_settings": {"stability": 0.45, "similarity_boost": 0.8,
                                               "style": 0.35}},
                           headers={"xi-api-key": key, "Accept": "audio/mpeg"}, raw=True)
            path.write_bytes(body)
            return
        except Exception as e:
            log(f"ElevenLabs failed, using free voice: {e}")
    for i in range(3):
        try:
            asyncio.run(_edge(text, path))
            if path.exists() and path.stat().st_size > 1000:
                return
        except Exception as e:
            log(f"edge-tts retry {i + 1}: {e}")
            time.sleep(4)
    raise RuntimeError("Voice generation failed")


# ---------------------------------------------------------------- 3. IMAGES

OK_LICENSES = ("public domain", "pd", "cc0", "cc by", "cc-by", "no restrictions")


def strip_html(s):
    return re.sub(r"<[^>]+>", "", s or "").strip()


def search_commons(query, used):
    q = urllib.parse.quote(f"{query} filetype:bitmap")
    url = ("https://commons.wikimedia.org/w/api.php?action=query&format=json"
           f"&generator=search&gsrsearch={q}&gsrnamespace=6&gsrlimit=12"
           "&prop=imageinfo&iiprop=url|size|extmetadata&iiurlwidth=2400")
    try:
        pages = http(url, timeout=40).get("query", {}).get("pages", {})
    except Exception as e:
        log(f"Commons search failed '{query}': {e}")
        return None
    for p in sorted(pages.values(), key=lambda p: p.get("index", 99)):
        ii = (p.get("imageinfo") or [{}])[0]
        if ii.get("width", 0) < 700 or ii.get("height", 0) < 450:
            continue
        meta = ii.get("extmetadata", {})
        lic = meta.get("LicenseShortName", {}).get("value", "").lower()
        if not any(k in lic for k in OK_LICENSES) or "nc" in lic.split() or "-nd" in lic:
            continue
        src = ii.get("thumburl") or ii.get("url")
        if not src or src in used or src.lower().endswith((".svg", ".gif", ".tif", ".tiff")):
            continue
        artist = strip_html(meta.get("Artist", {}).get("value", ""))[:80] or "Unknown"
        credit = f"{p['title'].replace('File:', '')} - {artist} ({meta.get('LicenseShortName', {}).get('value', '')}), Wikimedia Commons"
        return src, credit
    return None


def search_pexels(query, used, orientation):
    key = os.getenv("PEXELS_API_KEY")
    if not key:
        return None
    url = ("https://api.pexels.com/v1/search?per_page=15&query="
           f"{urllib.parse.quote(query)}&orientation={orientation}")
    try:
        for ph in http(url, headers={"Authorization": key}, timeout=40).get("photos", []):
            src = ph["src"]["large2x"]
            if src not in used:
                return src, f"Photo by {ph['photographer']} on Pexels"
    except Exception as e:
        log(f"Pexels failed '{query}': {e}")
    return None


def placeholder_image(path, label, w, h, seed):
    rnd = random.Random(seed)
    img = Image.new("RGB", (w, h), (12, 10, 9))
    d = ImageDraw.Draw(img)
    for _ in range(40):
        x, y, r = rnd.randint(0, w), rnd.randint(0, h), rnd.randint(80, 500)
        c = rnd.randint(25, 70)
        d.ellipse([x - r, y - r, x + r, y + r], fill=(c + 15, c, c - 10))
    img = img.filter(ImageFilter.GaussianBlur(60))
    d = ImageDraw.Draw(img)
    f = font(int(w * 0.05))
    for i, line in enumerate(textwrap.wrap(label.upper(), 18)):
        d.text((w // 2, h // 2 + i * int(w * 0.07)), line, font=f, fill=(200, 190, 170), anchor="mm")
    img.save(path, quality=92)


def get_images(scenes, work, fmt, offline):
    used, credits, files = set(), [], []
    orient = "portrait" if fmt == "short" else "landscape"
    for i, sc in enumerate(scenes):
        path = work / f"img_{i:02d}.jpg"
        found = None
        if not offline:
            for q in [sc.get("image_query", ""), " ".join(sc.get("image_query", "").split()[:3]),
                      sc.get("mood_query", "")]:
                if q.strip():
                    found = search_commons(q, used)
                    if found:
                        break
            if not found:
                found = search_pexels(sc.get("mood_query") or sc.get("image_query", "history"), used, orient)
        if found:
            try:
                body, _ = http(found[0], raw=True, timeout=60)
                path.write_bytes(body)
                Image.open(path).convert("RGB").save(path, quality=93)
                used.add(found[0])
                if found[1] not in credits:
                    credits.append(found[1])
                files.append(path)
                continue
            except Exception as e:
                log(f"Download failed: {e}")
        # reuse an earlier real image with a different crop rather than a blank card
        if files and not offline:
            files.append(files[(i * 7) % len(files)])
        else:
            placeholder_image(path, sc.get("image_query", "history"), 1600, 1200, i)
            files.append(path)
    return files, credits


# ---------------------------------------------------------------- 4. EDIT

def grade(img):
    """Old-archive documentary look: muted colour, warm shadows, contrast."""
    img = ImageOps.autocontrast(img.convert("RGB"), cutoff=1)
    img = ImageEnhance.Color(img).enhance(0.55)
    img = ImageEnhance.Contrast(img).enhance(1.12)
    img = ImageEnhance.Brightness(img).enhance(0.92)
    tint = Image.new("RGB", img.size, (60, 42, 25))
    return Image.blend(img, tint, 0.10)


def compose_frame(src, w, h, out_path, scale=2):
    """Fill the frame. Very different aspect -> blurred backdrop + full image."""
    W, H = w * scale, h * scale
    img = grade(Image.open(src))
    ia, ta = img.width / img.height, W / H
    if max(ia, ta) / min(ia, ta) < 1.45:
        canvas = ImageOps.fit(img, (W, H), Image.LANCZOS, centering=(0.5, 0.4))
    else:
        bg = ImageOps.fit(img, (W, H), Image.LANCZOS)
        bg = ImageEnhance.Brightness(bg.filter(ImageFilter.GaussianBlur(W // 40))).enhance(0.45)
        fg = ImageOps.contain(img, (int(W * 0.96), int(H * 0.96)), Image.LANCZOS)
        bg.paste(fg, ((W - fg.width) // 2, (H - fg.height) // 2))
        canvas = bg
    canvas.save(out_path, quality=94)


def scene_clip(frame, dur, w, h, idx, out_path):
    fps = 30
    n = max(1, int(math.ceil(dur * fps)))
    zmax = 1.12
    step = (zmax - 1) / n
    mode = idx % 4
    if mode == 0:    # slow push in
        z = f"min(1+{step:.6f}*on,{zmax})"
        x, y = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    elif mode == 1:  # pull out
        z = f"max({zmax}-{step:.6f}*on,1)"
        x, y = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    elif mode == 2:  # pan right
        z = f"{zmax}"
        x, y = f"(iw-iw/zoom)*on/{n}", "ih/2-(ih/zoom/2)"
    else:            # pan down
        z = f"{zmax}"
        x, y = "iw/2-(iw/zoom/2)", f"(ih-ih/zoom)*on/{n}"
    vf = (f"zoompan=z='{z}':x='{x}':y='{y}':d={n}:s={w}x{h}:fps={fps},"
          f"format=yuv420p")
    run(["ffmpeg", "-y", "-loop", "1", "-i", str(frame), "-vf", vf, "-frames:v", str(n),
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-an", str(out_path)])


def ass_time(t):
    h, m = int(t // 3600), int(t % 3600 // 60)
    return f"{h}:{m:02d}:{t % 60:05.2f}"


def build_captions(scenes, starts, durs, fmt, w, h, path):
    if fmt == "short":
        size, words_per, margin, align, outline = 82, 3, int(h * 0.30), 2, 6
    else:
        size, words_per, margin, align, outline = 54, 7, 70, 2, 4
    fname = "DejaVu Sans Condensed"
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{fname},{size},&H00FFFFFF,&H0000D7FF,&H00000000,&H80000000,-1,0,0,0,100,100,1,0,1,{outline},2,{align},60,60,{margin},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    for sc, st, du in zip(scenes, starts, durs):
        words = sc["text"].split()
        chunks = [words[i:i + words_per] for i in range(0, len(words), words_per)]
        speak = max(0.5, du - 0.35)  # scene audio ends with a short pause
        total = sum(len(" ".join(c)) + 3 for c in chunks)
        t = st
        for c in chunks:
            d = speak * (len(" ".join(c)) + 3) / total
            txt = " ".join(c)
            if fmt == "short":
                txt = txt.upper()
                # highlight the longest word in gold, like popular Shorts captions
                key = max(c, key=len)
                txt = txt.replace(key.upper(), r"{\c&H0000D7FF&}" + key.upper() + r"{\c&H00FFFFFF&}", 1)
                txt = r"{\fad(60,0)}" + txt
            lines.append(f"Dialogue: 0,{ass_time(t)},{ass_time(t + d)},Cap,,0,0,0,,{txt}")
            t += d
    path.write_text(head + "\n".join(lines) + "\n", encoding="utf-8")


def make_score(total, path, seed):
    """Original dark ambient bed (no copyright): low drones + filtered noise."""
    rnd = random.Random(seed)
    base = rnd.choice([46.25, 49.0, 51.91, 55.0, 58.27])
    fifth = base * 1.5
    expr = (f"0.30*sin(2*PI*{base}*t)*(0.75+0.25*sin(2*PI*0.07*t))"
            f"+0.18*sin(2*PI*{fifth}*t+0.5*sin(2*PI*0.05*t))"
            f"+0.08*sin(2*PI*{base * 2:.2f}*t)*(0.5+0.5*sin(2*PI*0.11*t))")
    run(["ffmpeg", "-y",
         "-f", "lavfi", "-i", f"aevalsrc='{expr}':s=44100:d={total + 1:.2f}",
         "-f", "lavfi", "-i", f"anoisesrc=color=brown:amplitude=0.25:d={total + 1:.2f}:r=44100",
         "-filter_complex",
         "[1]lowpass=f=400,volume=0.6[n];[0][n]amix=inputs=2:normalize=0,"
         f"afade=t=in:d=2,afade=t=out:st={max(0, total - 2.5):.2f}:d=2.5,volume=0.9[a]",
         "-map", "[a]", "-ac", "2", "-c:a", "pcm_s16le", str(path)])


def render(script, fmt, work, offline):
    cfg = FORMATS[fmt]
    w, h = cfg["w"], cfg["h"]
    scenes = script["scenes"]

    log("Voice...")
    auds, durs = [], []
    for i, sc in enumerate(scenes):
        raw = work / f"vo_{i:02d}.mp3"
        tts(sc["text"], raw, offline)
        pad = work / f"vo_{i:02d}.wav"
        run(["ffmpeg", "-y", "-i", str(raw), "-af", "apad=pad_dur=0.35,aresample=44100",
             "-ac", "2", str(pad)])
        auds.append(pad)
        durs.append(duration(pad))

    total = sum(durs)
    tempo = 1.0
    if total > cfg["max_sec"]:
        tempo = min(1.18, total / cfg["max_sec"])
        log(f"Narration {total:.1f}s too long, speeding up x{tempo:.2f}")
        for i, a in enumerate(auds):
            fast = work / f"vo_{i:02d}_f.wav"
            run(["ffmpeg", "-y", "-i", str(a), "-af", f"atempo={tempo:.3f}", str(fast)])
            auds[i] = fast
            durs[i] = duration(fast)
        total = sum(durs)

    log("Images...")
    imgs, credits = get_images(scenes, work, fmt, offline)

    log("Scenes...")
    clips = []
    for i, (img, du) in enumerate(zip(imgs, durs)):
        frame = work / f"frame_{i:02d}.jpg"
        compose_frame(img, w, h, frame)
        clip = work / f"clip_{i:02d}.mp4"
        scene_clip(frame, du, w, h, i, clip)
        clips.append(clip)

    starts, t = [], 0.0
    for d in durs:
        starts.append(t)
        t += d

    (work / "clips.txt").write_text("".join(f"file '{c.name}'\n" for c in clips))
    (work / "auds.txt").write_text("".join(f"file '{a.name}'\n" for a in auds))
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(work / "clips.txt"),
         "-c", "copy", str(work / "visual.mp4")])
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(work / "auds.txt"),
         "-c:a", "pcm_s16le", str(work / "voice.wav")])

    build_captions(scenes, starts, durs, fmt, w, h, work / "subs.ass")
    make_score(total, work / "score.wav", script["title"])

    log("Final render...")
    final = work / "final.mp4"
    subs = str(work / "subs.ass").replace("\\", "/").replace(":", "\\:")
    vf = (f"[0:v]eq=gamma=0.97,vignette=PI/4.5,noise=alls=9:allf=t+u,"
          f"subtitles='{subs}',fade=t=in:d=0.4,fade=t=out:st={max(0, total - 0.6):.2f}:d=0.6[v];"
          "[1:a]highpass=f=70,acompressor=threshold=-18dB:ratio=3:attack=5:release=120,"
          "loudnorm=I=-15:TP=-1.5:LRA=9,aresample=44100[vo];"
          "[2:a]volume=0.13[bg];[vo][bg]amix=inputs=2:duration=longest:normalize=0[a]")
    run(["ffmpeg", "-y", "-i", str(work / "visual.mp4"), "-i", str(work / "voice.wav"),
         "-i", str(work / "score.wav"), "-filter_complex", vf, "-map", "[v]", "-map", "[a]",
         "-c:v", "libx264", "-preset", "medium" if fmt == "short" else "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", "-r", "30", "-t", f"{total:.2f}",
         "-movflags", "+faststart", str(final)])
    return final, imgs, credits, total


# ---------------------------------------------------------------- 5. THUMBNAIL

def thumbnail(script, imgs, work):
    W, H = 1280, 720
    src = imgs[0] if imgs else None
    if src:
        base = ImageOps.fit(grade(Image.open(src)), (W, H), Image.LANCZOS, centering=(0.6, 0.4))
        base = ImageEnhance.Contrast(base).enhance(1.25)
    else:
        base = Image.new("RGB", (W, H), (15, 12, 10))
    # dark gradient on the left for text
    grad = Image.new("L", (W, H))
    gd = ImageDraw.Draw(grad)
    for x in range(W):
        gd.line([(x, 0), (x, H)], fill=int(235 * max(0, 1 - x / (W * 0.75))))
    base = Image.composite(Image.new("RGB", (W, H), (0, 0, 0)), base, grad)
    vig = Image.new("L", (W, H), 0)
    ImageDraw.Draw(vig).ellipse([-W * 0.25, -H * 0.35, W * 1.25, H * 1.35], fill=255)
    vig = vig.filter(ImageFilter.GaussianBlur(120))
    base = Image.composite(base, Image.new("RGB", (W, H), (0, 0, 0)), vig)

    d = ImageDraw.Draw(base)
    words = (script.get("thumbnail_text") or script["title"]).upper().split()[:5]
    lines, cur = [], ""
    for wd in words:
        if len(cur + " " + wd) > 11 and cur:
            lines.append(cur)
            cur = wd
        else:
            cur = (cur + " " + wd).strip()
    lines.append(cur)
    size = 128 if len(lines) <= 2 else 104
    f = font(size)
    y = H // 2 - (len(lines) * (size + 10)) // 2
    for i, line in enumerate(lines):
        color = (255, 214, 0) if i == len(lines) - 1 else (255, 255, 255)
        d.text((56, y), line, font=f, fill=color, stroke_width=8, stroke_fill=(0, 0, 0))
        y += size + 10
    # red accent bar + small channel tag
    d.rectangle([56, y + 8, 56 + 220, y + 22], fill=(200, 16, 30))
    out = work / "thumb.jpg"
    base.save(out, quality=92)
    return out


# ---------------------------------------------------------------- 6. YOUTUBE

def yt_token():
    r = http("https://oauth2.googleapis.com/token", urllib.parse.urlencode({
        "client_id": os.environ["YT_CLIENT_ID"],
        "client_secret": os.environ["YT_CLIENT_SECRET"],
        "refresh_token": os.environ["YT_REFRESH_TOKEN"],
        "grant_type": "refresh_token"}).encode(),
        headers={"Content-Type": "application/x-www-form-urlencoded"})
    return r["access_token"]


def build_description(script, credits, fmt):
    tags = " ".join(script.get("hashtags", []))
    if fmt == "short" and "#shorts" not in tags.lower():
        tags = "#shorts " + tags
    cred = "\n".join(f"- {c}" for c in credits[:25])
    desc = (f"{script['description'].strip()}\n\n"
            f"Subscribe to {CHANNEL} for the stories history tried to forget.\n\n"
            f"Image credits:\n{cred}\n\n{tags}")
    return desc[:4900]


def upload(video, thumb, script, credits, fmt):
    tok = yt_token()
    title = script["title"][:95]
    if fmt == "short" and "#shorts" not in title.lower() and len(title) < 90:
        title += " #shorts"
    meta = {"snippet": {"title": title,
                        "description": build_description(script, credits, fmt),
                        "tags": script.get("tags", [])[:15],
                        "categoryId": "27",  # Education
                        "defaultLanguage": "en", "defaultAudioLanguage": "en"},
            "status": {"privacyStatus": PRIVACY, "selfDeclaredMadeForKids": False,
                       "embeddable": True}}
    if os.getenv("DISCLOSE_SYNTHETIC", "").lower() in ("1", "true", "yes"):
        meta["status"]["containsSyntheticMedia"] = True
    size = video.stat().st_size
    _, hdr = http("https://www.googleapis.com/upload/youtube/v3/videos"
                  "?uploadType=resumable&part=snippet,status", meta,
                  headers={"Authorization": f"Bearer {tok}",
                           "X-Upload-Content-Type": "video/mp4",
                           "X-Upload-Content-Length": str(size)}, raw=True)
    loc = hdr["Location"]
    req = urllib.request.Request(loc, data=video.read_bytes(), method="PUT",
                                 headers={"Authorization": f"Bearer {tok}",
                                          "Content-Type": "video/mp4", "User-Agent": UA})
    with urllib.request.urlopen(req, timeout=1800) as r:
        res = json.loads(r.read())
    vid = res["id"]
    log(f"Uploaded: https://youtu.be/{vid} (status: {res['status']['privacyStatus']})")
    if fmt == "long" and thumb:
        try:
            http(f"https://www.googleapis.com/upload/youtube/v3/thumbnails/set?videoId={vid}"
                 "&uploadType=media", thumb.read_bytes(),
                 headers={"Authorization": f"Bearer {tok}", "Content-Type": "image/jpeg"})
            log("Thumbnail set")
        except Exception as e:
            log(f"Thumbnail not set (verify the channel by phone to allow custom thumbnails): {e}")
    return vid, res["status"]["privacyStatus"]


# ---------------------------------------------------------------- MAIN

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", choices=["short", "long"], default="short")
    ap.add_argument("--offline", action="store_true", help="test render without internet")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--script", help="use an existing script.json instead of writing one")
    a = ap.parse_args()

    DATA.mkdir(exist_ok=True)
    if a.script:
        script = json.loads(Path(a.script).read_text(encoding="utf-8"))
    else:
        script = write_script(a.format)

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M")
    work = OUT / f"{stamp}-{a.format}-{slugify(script['topic'])}"
    work.mkdir(parents=True, exist_ok=True)
    (work / "script.json").write_text(json.dumps(script, indent=2, ensure_ascii=False), encoding="utf-8")

    final, imgs, credits, total = render(script, a.format, work, a.offline)
    thumb = thumbnail(script, imgs, work)
    (work / "description.txt").write_text(build_description(script, credits, a.format), encoding="utf-8")
    log(f"Video ready: {final} ({total:.1f}s)")

    vid, status = "", "not uploaded"
    if not (a.offline or a.no_upload):
        vid, status = upload(final, thumb, script, credits, a.format)

    if not a.offline:
        with open(DATA / "topics_done.txt", "a", encoding="utf-8") as f:
            f.write(script["topic"].strip() + "\n")
        with open(DATA / "upload_log.csv", "a", encoding="utf-8") as f:
            f.write(f"{dt.datetime.now(dt.timezone.utc).isoformat()},{a.format},\"{script['title']}\","
                    f"{status},{'https://youtu.be/' + vid if vid else ''}\n")
    # keep only final outputs to save space
    for p in work.iterdir():
        if p.name not in ("final.mp4", "thumb.jpg", "script.json", "description.txt", "subs.ass"):
            p.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FAILED: {e}")
        sys.exit(1)
