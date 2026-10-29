#!/usr/bin/env python3
"""
Dark History studio (v2) - one run makes and uploads:
  * 1 long documentary (16:9, ~10-12 min, Urdu narration, English title)
  * 2 YouTube Shorts (9:16) cut from the strongest moments of that video

Pipeline: trending topic (YouTube + Wikipedia + anniversaries) -> story checked against Wikipedia
(3-4 free AI requests) -> narration (Google Chirp HD voice, spoken in long passages) -> real video
clips (Pexels, Pixabay, Wikimedia Commons film; archive photos only for named people/places) ->
edit (dust, grain, vignette) -> music with ducking -> thumbnail -> YouTube upload (Shorts later).

Secrets (GitHub -> Settings -> Secrets -> Actions) - all free tiers:
  GROQ_API_KEY                     AI writer (free). Fallbacks: GEMINI_API_KEY (free-tier project),
                                   OPENROUTER_API_KEY (":free" models), CF_ACCOUNT_ID + CF_API_TOKEN
  GOOGLE_TTS_KEY                   Google Text-to-Speech API key (voice)
  PEXELS_API_KEY / PIXABAY_API_KEY free stock videos (needed for video clips)
  YT_API_KEY                       YouTube Data API key for trending topics (optional)
  YT_CLIENT_ID / YT_CLIENT_SECRET / YT_REFRESH_TOKEN   YouTube upload
Variables (optional): CHANNEL_NAME, PRIVACY, TTS_VOICE, TTS_RATE, LONG_MINUTES, SHORTS_COUNT

Usage:
  python agent/studio.py                 # full run
  python agent/studio.py --offline --script samples/x.json --no-upload   # local test
"""
import argparse
import base64
import datetime as dt
import html
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
LANG = (os.getenv("NARRATION_LANG") or "en").strip().lower()   # "en" = English narration, "ur" = Urdu
TTS_VOICES = [v for v in [os.getenv("TTS_VOICE")] + (
    ["en-US-Chirp3-HD-Charon", "en-US-Chirp3-HD-Orus", "en-US-Chirp3-HD-Fenrir", "en-US-Studio-Q"] if LANG == "en" else
    ["ur-IN-Chirp3-HD-Charon", "ur-IN-Chirp3-HD-Orus", "ur-IN-Chirp3-HD-Fenrir", "ur-IN-Wavenet-B"]) if v]
W, H, FPS = 1920, 1080, 30

FONT_LATIN = ["/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
URDU_FONT_NAME = "DejaVu Sans" if LANG == "en" else "Noto Naskh Arabic"


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

# Free AI writers only, tried in this order: Groq free tier -> Gemini free tier -> OpenRouter ":free"
# models -> Cloudflare Workers AI (free daily allowance). Model names are read from each provider at run
# time, so a retired model can never 404 the whole run again. A model that answers 403/404 is dropped,
# a provider that answers 401/402 (bad key / billing) is switched off, and a 429 waits exactly as long as
# the API asks before retrying (or moves on when that wait is longer than LLM_MAX_WAIT).

MAX_WAIT = float(os.getenv("LLM_MAX_WAIT") or 90)
LLM_STATE = {"dead": set(), "models": None, "budget": {}, "last": 0.0, "calls": 0, "ok": None}

GROQ_PREFER = ["openai/gpt-oss-120b", "moonshotai/kimi-k2", "qwen/qwen3-32b", "llama-4-maverick",
               "openai/gpt-oss-20b", "llama-4-scout", "llama-3.3-70b"]
OPENROUTER_PREFER = ["deepseek/deepseek-chat", "deepseek/deepseek-v3", "openai/gpt-oss-120b",
                     "meta-llama/llama-3.3-70b", "qwen/qwen3-235b", "moonshotai/kimi-k2", "google/gemma-3-27b",
                     "mistralai/mistral-small"]
CF_MODELS = ["@cf/meta/llama-3.3-70b-instruct-fp8-fast", "@cf/meta/llama-4-scout-17b-16e-instruct",
             "@cf/mistralai/mistral-small-3.1-24b-instruct"]
SKIP_MODEL = re.compile(r"whisper|tts|guard|safeguard|compound|embed|vision|image|audio|live|orpheus|playai|"
                        r"allam|instant|\b[1-9]b\b|-[1-9]b|nano|robotics|computer-use", re.I)


def parse_secs(v):
    """'20', '7.66s', '2m59.5s', '1h2m3s', '350ms' -> seconds (None if unreadable)."""
    v = str(v or "").strip()
    try:
        return float(v)
    except ValueError:
        pass
    if re.fullmatch(r"[\d.]+ms", v):
        return float(v[:-2]) / 1000
    parts = re.findall(r"([\d.]+)(h|m|s)", v)
    return sum(float(n) * {"h": 3600, "m": 60, "s": 1}[u] for n, u in parts) if parts else None


def retry_after(e, body):
    """How long a 429 answer asks us to wait (Retry-After, Groq reset headers, Gemini retryDelay)."""
    hdr = e.headers or {}
    for k in ("retry-after", "x-ratelimit-reset-tokens", "x-ratelimit-reset-requests"):
        s = parse_secs(hdr.get(k))
        if s is not None:
            return s
    m = re.search(rb'"retryDelay":\s*"([\d.]+)s"', body) or re.search(rb"try again in ([\dhms.]+)", body)
    return parse_secs(m.group(1).decode()) if m else None


def rank(ids, prefer):
    score = lambda m: next((i for i, p in enumerate(prefer) if p in m), len(prefer))
    return sorted((m for m in ids if not SKIP_MODEL.search(m)), key=score)


def provider_models():
    """[(provider, model)] in the order we try them, looked up once per run."""
    if LLM_STATE["models"] is not None:
        return LLM_STATE["models"]
    out = []
    if os.getenv("GROQ_API_KEY"):
        try:
            data = http("https://api.groq.com/openai/v1/models", timeout=30,
                        headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"})["data"]
            ids = [m["id"] for m in data if m.get("active", True)]
        except Exception as e:
            log(f"Groq model list failed ({e}) - using defaults")
            ids = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
        ids = rank(ids, GROQ_PREFER)[:5]
        if os.getenv("GROQ_MODEL"):
            ids = [os.getenv("GROQ_MODEL")] + [m for m in ids if m != os.getenv("GROQ_MODEL")]
        out += [("groq", m) for m in ids]
    if os.getenv("GEMINI_API_KEY"):
        try:
            data = http("https://generativelanguage.googleapis.com/v1beta/models?pageSize=200&key="
                        + os.environ["GEMINI_API_KEY"], timeout=30).get("models", [])
            ids = [m["name"] for m in data if "generateContent" in m.get("supportedGenerationMethods", [])
                   and "flash" in m["name"] and not SKIP_MODEL.search(m["name"])]

            def ver(n):
                v = re.search(r"gemini-(\d+(?:\.\d+)?)", n)
                return (-float(v.group(1)) if v else 0, "lite" in n, "preview" in n or "exp" in n)
            ids = sorted(ids, key=ver)[:4]
        except urllib.error.HTTPError as e:
            log(f"Gemini key not usable (HTTP {e.code}) - skipping Gemini")
            ids = []
        except Exception as e:
            log(f"Gemini model list failed: {e}")
            ids = []
        out += [("gemini", m) for m in ids]
    if os.getenv("OPENROUTER_API_KEY"):
        try:
            data = http("https://openrouter.ai/api/v1/models", timeout=30)["data"]
            free = [m["id"] for m in data if m["id"].endswith(":free")
                    and str(m.get("pricing", {}).get("prompt")) in ("0", "0.0")
                    and str(m.get("pricing", {}).get("completion")) in ("0", "0.0")]
            out += [("openrouter", m) for m in rank(free, OPENROUTER_PREFER)[:4]]
        except Exception as e:
            log(f"OpenRouter model list failed: {e}")
    if os.getenv("CF_ACCOUNT_ID") and os.getenv("CF_API_TOKEN"):
        out += [("cloudflare", m) for m in CF_MODELS]
    log("AI writers: " + ", ".join(f"{p}:{m.split('/')[-1]}" for p, m in out))
    LLM_STATE["models"] = out
    return out


def chat_call(provider, model, prompt, max_tokens, json_mode):
    msgs = [{"role": "user", "content": prompt}]
    if provider == "groq":
        body = {"model": model, "temperature": 0.8, "max_completion_tokens": max_tokens, "messages": msgs}
        if "gpt-oss" in model:
            body["reasoning_effort"] = "low"  # thinking tokens count against the free per-minute budget
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        raw, hdr = http("https://api.groq.com/openai/v1/chat/completions", body, raw=True, timeout=300,
                        headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"})
        rem = hdr.get("x-ratelimit-remaining-tokens")
        if rem and rem.isdigit():
            reset = parse_secs(hdr.get("x-ratelimit-reset-tokens")) or 0
            LLM_STATE["budget"][model] = (int(rem), time.time() + reset)
        return json.loads(raw)["choices"][0]["message"]["content"] or ""
    if provider == "gemini":
        body = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.8}}
        if json_mode:
            body["generationConfig"]["responseMimeType"] = "application/json"
        r = http(f"https://generativelanguage.googleapis.com/v1beta/{model}:generateContent?key="
                 + os.environ["GEMINI_API_KEY"], body, timeout=300)
        return "".join(p.get("text", "") for p in r["candidates"][0]["content"]["parts"])
    if provider == "openrouter":
        body = {"model": model, "temperature": 0.8, "max_tokens": max_tokens, "messages": msgs}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        r = http("https://openrouter.ai/api/v1/chat/completions", body, timeout=300,
                 headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}", "X-Title": CHANNEL})
        return r["choices"][0]["message"]["content"] or ""
    r = http(f"https://api.cloudflare.com/client/v4/accounts/{os.environ['CF_ACCOUNT_ID']}/ai/run/{model}",
             {"messages": msgs, "max_tokens": max_tokens, "temperature": 0.8}, timeout=300,
             headers={"Authorization": f"Bearer {os.environ['CF_API_TOKEN']}"})
    out = r.get("result", {}).get("response")
    return json.dumps(out) if isinstance(out, (dict, list)) else (out or "")


def pace(model, prompt, max_tokens):
    """Space requests out and wait for Groq's per-minute token budget before sending, not after a 429."""
    gap = time.time() - LLM_STATE["last"]
    if gap < 3:
        time.sleep(3 - gap)
    rem, reset_at = LLM_STATE["budget"].get(model, (None, 0))
    if rem is not None and rem < len(prompt) / 3.5 + max_tokens and reset_at > time.time():
        w = min(MAX_WAIT, reset_at - time.time() + 1)
        log(f"Waiting {w:.0f}s for {model} token budget")
        time.sleep(w)


def llm(prompt, json_mode=True, max_tokens=4000):
    tried = []
    for provider, model in provider_models():
        if provider in LLM_STATE["dead"] or model in LLM_STATE["dead"]:
            continue
        jm = json_mode
        for attempt in range(4):
            pace(model, prompt, max_tokens)
            try:
                txt = chat_call(provider, model, prompt, max_tokens, jm)
                LLM_STATE["last"] = time.time()
                LLM_STATE["calls"] += 1
                if txt.strip():
                    if LLM_STATE["ok"] != model:
                        log(f"AI writer ok: {provider} {model}")
                    LLM_STATE["ok"] = model
                    return txt
                log(f"{provider} {model}: empty answer")
                break
            except urllib.error.HTTPError as e:
                LLM_STATE["last"] = time.time()
                body = e.read()[:800]
                tried.append(f"{model.split('/')[-1]}:{e.code}")
                if e.code == 429:
                    wait = retry_after(e, body)
                    if wait is None:
                        wait = 15 * (attempt + 1)
                    if wait <= MAX_WAIT and attempt < 3:
                        log(f"{provider} {model}: 429 rate limit, waiting {wait:.0f}s then retrying")
                        time.sleep(wait + 1)
                        continue
                    log(f"{provider} {model}: rate limit needs {wait:.0f}s - trying the next free model")
                    break
                if e.code in (400, 422) and jm:  # some free models don't support JSON mode
                    jm = False
                    continue
                if e.code in (401, 402):
                    LLM_STATE["dead"].add(provider)
                    log(f"{provider}: HTTP {e.code} {body[:180]!r} - switched off for this run")
                    break
                if e.code >= 500 and attempt < 2:
                    time.sleep(5 * (attempt + 1))
                    continue
                if e.code in (403, 404):
                    LLM_STATE["dead"].add(model)
                log(f"{provider} {model}: HTTP {e.code} {body[:180]!r}")
                break
            except Exception as e:
                LLM_STATE["last"] = time.time()
                tried.append(f"{model.split('/')[-1]}:err")
                log(f"{provider} {model}: {str(e)[:150]}")
                break
    if not provider_models():
        raise RuntimeError("Add GROQ_API_KEY (free) in GitHub secrets")
    raise RuntimeError(f"All free AI writers failed: {', '.join(tried[-10:])}")


def extract_json(txt):
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", txt, re.S)
    s = m.group(1) if m else txt[txt.find("{"): txt.rfind("}") + 1]
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        s2 = re.sub(r",\s*([}\]])", r"\1", s)  # trailing commas
        return json.loads(s2)


def ask_json(prompt, keys=(), max_tokens=4000):
    for attempt in range(2):
        raw = llm(prompt, json_mode=True, max_tokens=max_tokens)
        try:
            obj = extract_json(raw)
        except Exception as e:
            log(f"JSON broken ({e}), asking once more")
            continue
        if all(k in obj for k in keys):
            return obj
        log(f"JSON missing keys {keys}, retry {attempt + 1}")
    raise RuntimeError("AI writer did not return usable JSON")


# ---------------------------------------------------------------- trending topics (free public data)

DARK = re.compile(r"disaster|massacre|murder|killing|killer|myster|disappear|vanish|curse|ghost|haunt|shipwreck|"
                  r"sinking|sank|wreck|explosion|plague|epidemic|famine|execut|assassinat|expedition|cannibal|"
                  r"cult\b|tragedy|catastroph|collapse|poison|witch|hoax|unsolved|unexplained|kidnap|siege|mutiny|"
                  r"abandoned|experiment|crash|flood|eruption|stampede|genocide|atrocit|lost |prison|asylum|"
                  r"tomb|mummy|pirate|serial|hanged|burned|buried|torture|inquisition", re.I)
NOT_STORY = re.compile(r"footballer|cricketer|singer|rapper|actor|actress|\bfilm\b|television|tv series|season|"
                       r"album|song|video game|election|politician|company|tournament|league|\bcup\b|\bband\b|"
                       r"wrestler|youtuber|player|musician|episode|novel|comic|franchise|streamer", re.I)
OLD = re.compile(r"\b(1[0-9]{3}|200[0-5])\b|\bBC\b|century|ancient|medieval|historic", re.I)


def yt_trending(limit=15):
    """Dark-history videos with the most views in the last 3 weeks = topics already working in the niche."""
    key = os.getenv("YT_API_KEY") or os.getenv("GOOGLE_TTS_KEY")
    if not key:
        return []
    after = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=21)).strftime("%Y-%m-%dT%H:%M:%SZ")
    found = {}
    for q in ["dark history", "history mystery documentary", "disturbing history"]:
        url = "https://www.googleapis.com/youtube/v3/search?" + urllib.parse.urlencode({
            "part": "snippet", "type": "video", "order": "viewCount", "publishedAfter": after, "q": q,
            "maxResults": 15, "relevanceLanguage": "en", "key": key})
        try:
            for it in http(url, timeout=30).get("items", []):
                found.setdefault(it["id"]["videoId"], html.unescape(it["snippet"]["title"]))
        except urllib.error.HTTPError as e:
            log(f"YouTube trend search off (HTTP {e.code}) - add a YT_API_KEY secret with YouTube Data API enabled")
            break
        except Exception as e:
            log(f"YouTube trend search failed: {e}")
            break
    if not found:
        return []
    views = {}
    try:
        url = "https://www.googleapis.com/youtube/v3/videos?" + urllib.parse.urlencode(
            {"part": "statistics", "id": ",".join(list(found)[:50]), "key": key})
        views = {v["id"]: int(v["statistics"].get("viewCount", 0)) for v in http(url, timeout=30).get("items", [])}
    except Exception as e:
        log(f"YouTube view counts failed: {e}")
    rows = sorted(found.items(), key=lambda kv: -views.get(kv[0], 0))
    return [(t, views.get(v, 0)) for v, t in rows[:limit]]


def wiki_trending(limit=12):
    """History articles people are reading most on Wikipedia in the last 2 days."""
    views = {}
    for back in (1, 2):
        d = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=back)
        try:
            arts = http(f"https://wikimedia.org/api/rest_v1/metrics/pageviews/top/en.wikipedia/all-access/"
                        f"{d:%Y/%m/%d}", timeout=30)["items"][0]["articles"]
        except Exception as e:
            log(f"Wikipedia top views {d:%Y-%m-%d} failed: {e}")
            continue
        for a in arts[:500]:
            t = a["article"].replace("_", " ")
            if ":" not in t and t != "Main Page":
                views[t] = views.get(t, 0) + a["views"]
    desc, names = {}, list(views)
    for k in range(0, len(names), 50):
        q = urllib.parse.urlencode({"action": "query", "format": "json", "prop": "description",
                                    "titles": "|".join(names[k:k + 50])})
        try:
            time.sleep(0.3)
            for p in http("https://en.wikipedia.org/w/api.php?" + q, timeout=30)["query"]["pages"].values():
                desc[p["title"]] = p.get("description", "")
        except Exception as e:
            log(f"Wikipedia descriptions failed: {e}")
    hits = [(t, desc.get(t, ""), v) for t, v in views.items()
            if DARK.search(f"{t} {desc.get(t, '')}") and not NOT_STORY.search(desc.get(t, ""))
            and OLD.search(f"{t} {desc.get(t, '')}")]
    return sorted(hits, key=lambda h: -h[2])[:limit]


def on_this_day(days=4, limit=10):
    """Dark events whose anniversary falls in the next few days (people search them on the date)."""
    out = []
    for k in range(days):
        d = dt.date.today() + dt.timedelta(days=k)
        try:
            evs = http(f"https://en.wikipedia.org/api/rest_v1/feed/onthisday/events/{d:%m}/{d:%d}",
                       timeout=30).get("events", [])
        except Exception as e:
            log(f"On-this-day {d} failed: {e}")
            continue
        for ev in evs:
            if ev.get("year", 3000) > 1995 or not DARK.search(ev.get("text", "")):
                continue
            pages = ev.get("pages") or [{}]
            pg = next((p for p in pages if DARK.search(p.get("description", "") + p.get("title", ""))), pages[0])
            title = (pg.get("normalizedtitle") or pg.get("title", "")).replace("_", " ")
            out.append((title, f"{d:%d %b} {ev['year']}: {ev['text']}"))
    return out[:limit]


def trend_signals():
    yt, wk, od = yt_trending(), wiki_trending(), on_this_day()
    log(f"Trend signals: {len(yt)} YouTube, {len(wk)} Wikipedia, {len(od)} anniversaries")
    lines = []
    if yt:
        lines.append("Dark-history videos with the most views on YouTube in the last 3 weeks (proven demand - "
                     "never copy them, pick the same KIND of story or a fresh one they did not cover):")
        lines += [f"- {t} ({v:,} views)" for t, v in yt]
    if wk:
        lines.append("Real history articles people are reading most on Wikipedia right now:")
        lines += [f"- {t} - {d} ({v:,} views in 2 days)" for t, d, v in wk]
    if od:
        lines.append("Dark anniversaries in the next days (a video on the anniversary gets extra search traffic):")
        lines += [f"- {t} - {d}" for t, d in od]
    return "\n".join(lines)


def wiki_extract(title, chars=4500):
    """Plain-text Wikipedia article for the chosen story - the facts the narration must follow."""
    if not title:
        return ""
    try:
        q = urllib.parse.urlencode({"action": "query", "format": "json", "list": "search", "srsearch": title,
                                    "srlimit": 1})
        hits = http("https://en.wikipedia.org/w/api.php?" + q, timeout=30)["query"]["search"]
        if not hits:
            return ""
        q = urllib.parse.urlencode({"action": "query", "format": "json", "prop": "extracts", "explaintext": 1,
                                    "redirects": 1, "titles": hits[0]["title"]})
        txt = next(iter(http("https://en.wikipedia.org/w/api.php?" + q, timeout=30)["query"]["pages"].values()
                        )).get("extract", "")
    except Exception as e:
        log(f"Wikipedia facts failed: {e}")
        return ""
    cut = re.search(r"\n==\s*(See also|References|Notes|Further reading|External links|In popular culture)\s*==", txt)
    txt = re.sub(r"\n{2,}", "\n", txt[:cut.start()] if cut else txt)
    log(f"Facts source: Wikipedia '{hits[0]['title']}' ({len(txt)} chars)")
    return txt[:chars]


def used_topics():
    f = DATA / "topics_done.txt"
    return [l.strip() for l in f.read_text(encoding="utf-8").splitlines() if l.strip()] if f.exists() else []


VISUAL_RULES = """The picture on screen MUST show what the narration of that scene says, in the era and
place of the story. This is a VIDEO documentary: almost every scene is a real moving video clip.
For every scene choose the visual:
- "type": "video" (real footage) - about 80% of scenes. Think like a documentary editor using stock
  footage: atmosphere, details and re-enactment shots that fit the era - stormy sea, old sailing ship,
  fog over water, snow forest, candle flame, old letter, hands writing with a quill, old book pages,
  lantern in the dark, rain on a window, burning fire, empty dark corridor, old map, ruins, cemetery,
  clock ticking, silhouette walking in fog, horse carriage, steam train, medieval castle.
  NEVER modern things (cities, cars, phones, offices, neon, concerts, modern people's faces).
  "archive" (a real historical photo/painting/map of a named person, place or object) - about 15%,
  only when the narration names a real person, ship, building or document. Never two "archive" in a row.
  "image" (AI picture) - at most 5%, only when nothing filmable fits.
- "query": 2-4 simple English words for stock-footage search of a clip that exists on Pexels/Pixabay
  (e.g. "stormy ocean night", "candle flame dark", "old ship sailing").
- "query2": a second, more general 2-3 word search in case the first finds nothing (e.g. "dark ocean").
- "film": ONLY if real film footage of this exact event/era exists (after 1895: newsreels, war,
  disasters, ships) 2-4 English words to search Wikimedia Commons videos, e.g. "Titanic 1912 film",
  else "".
- "archive": 2-5 English words to search Wikimedia Commons photos (names, places, years).
- "prompt": English prompt for an AI picture of the scene (backup only): subject, era, place, lighting.
  Photorealistic cinematic film still, moody, 35mm. No text, no gore, no real faces in close-up."""

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


def is_used(topic, done):
    key = slugify(re.sub(r"\b(the|disappearance|mystery|tragedy|disaster)\b", "", topic.lower()))
    return any(t.strip().lower() == topic.strip().lower() or
               (len(key) >= 5 and (key in slugify(t) or slugify(t) in key)) for t in done)


def write_story(minutes):
    """3-4 AI requests in total: 1 outline (picks a trending topic) + the narration 2 chapters at a time."""
    done = used_topics()
    avoid = "\n".join(f"- {t}" for t in done[-300:]) or "- (none yet)"
    n_ch = max(4, round(minutes / 2.3))
    trends = trend_signals() or "(no trend data today - pick a story with strong search interest)"
    outline = ask_json(f"""You are the head writer of "{CHANNEL}", a faceless YouTube channel that tells
true, little-known, dark and mysterious stories from real history (gripping narrator, suspense,
cliffhangers, a final twist). Today is {dt.date.today():%d %B %Y}.

TRENDING SIGNALS (free public data collected today):
{trends}

Pick ONE real, specific, verifiable dark or mysterious historical story that is most likely to get
views NOW: best is a story from the Wikipedia or anniversary lists above, or a story of the same kind
as the YouTube videos that are getting views. Curses, disappearances, disasters, cover-ups, strange
deaths, forgotten tragedies, bizarre experiments, lost places.
It must NOT be any of these already-used topics:
{avoid}
Never invent facts; mark legends as legends. Respect victims: no gore for shock, no glorifying violence.
Skip stories about living people and stories tied to wars or political conflicts that are still going on.

Plan a {minutes:.0f}-minute documentary with {n_ch} chapters. Return ONLY JSON:
{{"topic": "short English name of the story",
  "wikipedia": "exact title of the English Wikipedia article about this story",
  "trend_reason": "which trending signal this matches and why it will get views",
  "title": "English YouTube title, max 70 chars, curiosity + truthful, may use one CAPS word",
  "thumbnail_text": "2-4 English words, punchy",
  "setting": "visual setting for all pictures: year/era, place, period details",
  "thumbnail_prompt": "English prompt for a dramatic photorealistic cinematic thumbnail image of the story's key moment, no text",
  "description": "English description: 2 short hook paragraphs + 'Sources:' with 2-4 real sources",
  "tags": ["12-15 English SEO tags"],
  "hashtags": ["#history", "#mystery", "..."],
  "chapters": [{{"title": "short English chapter name", "summary": "3-4 sentences of the facts this chapter tells"}}],
  "shorts": [{{"chapter": 0, "title": "English Shorts title max 60 chars"}}, {{"chapter": 2, "title": "..."}}]
}}""", keys=("topic", "title", "chapters"), max_tokens=3000)
    if is_used(outline["topic"], done):
        raise RuntimeError(f"repeated topic: {outline['topic']}")
    log(f"Topic: {outline['topic']} | {outline['title']} | why: {outline.get('trend_reason', '')[:160]}")
    facts = wiki_extract(outline.get("wikipedia") or outline["topic"])

    LANG_RULE = ("simple, clear spoken English like a gripping documentary narrator (short everyday "
                 "words, easy for non-native listeners; no complicated vocabulary)." if LANG == "en" else
                 "natural spoken Urdu (Urdu script, simple everyday Urdu like a popular storyteller; "
                 "English words only for names/places/terms).")
    chapters = outline["chapters"]
    words_per_ch = int(minutes * (150 if LANG == "en" else 135) / len(chapters))
    scenes, previous = [], ""
    for b0 in range(0, len(chapters), 2):
        batch = list(range(b0, min(b0 + 2, len(chapters))))
        first, last = batch[0] == 0, batch[-1] == len(chapters) - 1
        plan = "\n".join(f"{n + 1}. {chapters[i]['title']} - {chapters[i]['summary']}"
                         for n, i in enumerate(batch))
        part = ask_json(f"""You write the narration of chapters {batch[0] + 1}-{batch[-1] + 1} of {len(chapters)} of the
YouTube documentary "{outline['title']}" (story: {outline['topic']}).
Chapters to write now:
{plan}
Story so far (last lines): {previous[-500:] or '(start of the video)'}
FACTS (Wikipedia - follow these; if the chapter plan disagrees with them, the facts win):
{facts or '(none - only use facts you are sure of)'}

For EACH chapter write about {words_per_ch} words of narration in {LANG_RULE}
Write it the way a great narrator TALKS, not like an essay: varied sentence length, contractions
(it's, didn't), natural connectors (but, and then, so), a few direct questions to the viewer,
"..." only for real dramatic pauses (max one per scene), "!" rarely. No headings, no lists.
End each chapter on a small cliffhanger. {'Chapter 1 starts with a powerful 2-sentence hook, then a quick "In this video..." promise.' if first else 'Continue smoothly, do not greet again.'}
{'The last chapter ends with a chilling final thought, then asks viewers to like, comment and subscribe to ' + CHANNEL + '.' if last else ''}
Split each chapter into scenes of 1-2 sentences (8-14 seconds each when spoken).
{VISUAL_RULES}
{SFX_RULES}
Return ONLY JSON with one entry per chapter, in order:
{{"chapters": [{{"scenes": [{{"ur": "narration", "type": "video|archive|image", "query": "...", "query2": "...",
"film": "", "archive": "...", "prompt": "...", "sfx": "...", "hit": false}}]}}]}}""",
                        keys=("chapters",), max_tokens=min(6000, 1200 + int(len(batch) * words_per_ch * 4.5)))
        got = part["chapters"]
        if len(got) < len(batch):
            raise RuntimeError(f"AI wrote {len(got)} of {len(batch)} chapters")
        for i, ch in zip(batch, got):
            for sc in ch.get("scenes", []):
                if str(sc.get("ur", "")).strip():
                    sc["chapter"] = i
                    scenes.append(sc)
            log(f"Chapter {i + 1}: {len(ch.get('scenes', []))} scenes")
        previous = " ".join(s["ur"] for s in scenes[-3:])
    outline["scenes"] = scenes
    log(f"Story written with {LLM_STATE['calls']} AI requests")
    return outline


# =============================================================== 2. VOICE (Google TTS)
# The narration is spoken in long passages (a whole chapter part per request) instead of one request per
# sentence: the voice keeps its natural flow and intonation, and a video needs ~8 requests instead of ~80.
# Scene timings are then found by snapping each sentence boundary to the nearest real pause in the audio.

GEMINI_TTS = {"on": bool(os.getenv("GEMINI_API_KEY")) and (os.getenv("TTS_ENGINE") or "gemini") == "gemini"}
TTS_CHUNK_BYTES = int(os.getenv("TTS_CHUNK_BYTES") or 2800)
VOICE_STYLE = (f"You are a famous {'English' if LANG == 'en' else 'Urdu'} dark-history storyteller on YouTube. "
               "Narrate this like you are telling it to a friend late at night: natural, warm and fluent, with "
               "real emotion and suspense, a confident relaxed pace (never slow, never like reading from paper), "
               "short pauses at '...', a lower, quieter voice for the scary parts and a direct questioning tone "
               "when asking the viewer.")


def speakable(text):
    """Small clean-ups that stop TTS from sounding robotic (ALL-CAPS spelled out, stacked marks)."""
    t = re.sub(r"\b([A-Z]{4,})\b", lambda m: m.group(1).capitalize(), text)
    t = re.sub(r"\.{4,}", "...", t)
    t = re.sub(r"([!?]){2,}", r"\1", t)
    return re.sub(r"\s+", " ", t).strip()


def gemini_tts(text, path):
    key = os.environ["GEMINI_API_KEY"]
    voice = os.getenv("GEMINI_VOICE") or "Charon"
    last = None
    for model in [os.getenv("GEMINI_TTS_MODEL") or "gemini-2.5-flash-preview-tts"]:
        body = {"contents": [{"parts": [{"text": VOICE_STYLE + "\n\nText:\n" + text}]}],
                "generationConfig": {"responseModalities": ["AUDIO"], "speechConfig": {
                    "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}}}}
        for attempt in range(4):
            try:
                r = http(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",
                         body, timeout=300)
                data = r["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]
                pcm = Path(str(path) + ".pcm")
                pcm.write_bytes(base64.b64decode(data))
                run(["ffmpeg", "-y", "-f", "s16le", "-ar", "24000", "-ac", "1", "-i", pcm,
                     "-af", "silenceremove=start_periods=1:start_threshold=-50dB,areverse,"
                            "silenceremove=start_periods=1:start_threshold=-50dB,areverse",
                     "-b:a", "192k", path])
                pcm.unlink(missing_ok=True)
                return
            except urllib.error.HTTPError as e:
                body_txt = e.read()[:300]
                last = f"HTTP {e.code} {body_txt!r}"
                if e.code in (401, 402, 403, 404):
                    GEMINI_TTS["on"] = False  # billing/key problem: don't keep asking
                    break
                if e.code in (429, 500, 503):
                    wait = retry_after(e, body_txt) or 10 * (attempt + 1)
                    if wait > MAX_WAIT:
                        break
                    time.sleep(wait + 1)
                    continue
                break
    raise RuntimeError(f"Gemini voice: {last}")


def google_tts(text, path):
    key = os.environ["GOOGLE_TTS_KEY"]
    rate = float(os.getenv("TTS_RATE") or 1.05)
    last = None
    for voice in list(TTS_VOICES):
        body = {"input": {"text": text},
                "voice": {"languageCode": "-".join(voice.split("-")[:2]), "name": voice},
                "audioConfig": {"audioEncoding": "MP3", "speakingRate": rate, "sampleRateHertz": 24000}}
        for attempt in range(4):
            try:
                r = http(f"https://texttospeech.googleapis.com/v1/text:synthesize?key={key}", body, timeout=180)
                Path(path).write_bytes(base64.b64decode(r["audioContent"]))
                if voice != TTS_VOICES[0]:  # remember a working voice for the rest of the video
                    TTS_VOICES.remove(voice)
                    TTS_VOICES.insert(0, voice)
                return
            except urllib.error.HTTPError as e:
                body_txt = e.read()[:300]
                last = f"{voice}: {e.code} {body_txt!r}"
                if e.code in (429, 500, 503):
                    time.sleep(min(MAX_WAIT, retry_after(e, body_txt) or 6 * (attempt + 1)))
                    continue
                break
            except Exception as e:
                last = f"{voice}: {e}"
                time.sleep(3)
        log(f"TTS voice failed -> {last}")
    raise RuntimeError(f"Google TTS failed: {last}")


def tts_chunks(scenes):
    """Groups scenes into passages: same chapter and under the TTS request size limit."""
    chunks, cur, size = [], [], 0
    for i, sc in enumerate(scenes):
        b = len(sc["ur"].encode()) + 1
        if cur and (scenes[cur[-1]]["chapter"] != sc["chapter"] or size + b > TTS_CHUNK_BYTES):
            chunks.append(cur)
            cur, size = [], 0
        cur.append(i)
        size += b
    return chunks + [cur] if cur else chunks


def pauses(path):
    """Middle points of the pauses in a narration file."""
    r = subprocess.run(["ffmpeg", "-i", str(path), "-af", "silencedetect=noise=-35dB:d=0.14", "-f", "null", "-"],
                       capture_output=True, text=True)
    st = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", r.stderr)]
    en = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", r.stderr)]
    return [(s + e) / 2 for s, e in zip(st, en)]


def split_points(texts, total, mids):
    """End time of each scene inside one continuous passage: expected spot from the text length,
    snapped to the nearest real pause so a cut never lands in the middle of a word."""
    lens = [len(t) + 2 for t in texts]
    ends, cum, prev = [], 0, 0.0
    for n in lens[:-1]:
        cum += n
        guess = total * cum / sum(lens)
        near = [m for m in mids if prev + 0.8 < m < total - 0.8]
        best = min(near, key=lambda m: abs(m - guess), default=None)
        t = best if best is not None and abs(best - guess) < 2.5 else max(guess, prev + 0.8)
        ends.append(min(t, total - 0.5))
        prev = ends[-1]
    return ends + [total]


def narrate(scenes, work, offline):
    """Speaks the whole story; writes work/voice.wav and returns the duration of every scene."""
    engines = ["offline"] if offline else (["gemini"] if GEMINI_TTS["on"] else []) + ["google"]
    chunks = tts_chunks(scenes)
    for engine in engines:
        try:
            durs, wavs = [0.0] * len(scenes), []
            for k, idx in enumerate(chunks):
                text = " ".join(speakable(scenes[i]["ur"]) for i in idx)
                raw = work / f"vo_{k:03d}.mp3"
                if engine == "offline":
                    run(["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=f=180:r=24000", "-af", "volume=0.05",
                         "-t", f"{max(2.5, len(text) / 14):.2f}", raw])
                elif engine == "gemini":
                    gemini_tts(text, raw)
                else:
                    google_tts(text, raw)
                nxt = chunks[k + 1][0] if k + 1 < len(chunks) else None
                pad = 0.8 if nxt is not None and scenes[nxt]["chapter"] != scenes[idx[0]]["chapter"] else 0.25
                wav = work / f"vo_{k:03d}.wav"
                run(["ffmpeg", "-y", "-i", raw, "-af", f"apad=pad_dur={pad},aresample=44100", "-ac", "2", wav])
                total = duration(wav)
                ends = split_points([scenes[i]["ur"] for i in idx], total - pad, pauses(raw))
                ends[-1] = total
                prev = 0.0
                for i, e in zip(idx, ends):
                    durs[i], prev = e - prev, e
                wavs.append(wav)
            (work / "auds.txt").write_text("".join(f"file '{w.name}'\n" for w in wavs))
            run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", work / "auds.txt", "-c:a", "pcm_s16le",
                 work / "voice.wav"])
            log(f"Voice: {engine}, {len(chunks)} requests for {len(scenes)} scenes")
            return durs
        except Exception as e:  # never mix two voices: the next engine redoes the whole narration
            log(f"Voice engine {engine} failed: {str(e)[:300]}")
    raise RuntimeError("No voice engine worked (check GOOGLE_TTS_KEY)")


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


VID_CACHE, VID_OFF = {}, set()  # one search per query (results reused), sources off after a 429


def pexels_search(query):
    key = os.getenv("PEXELS_API_KEY")
    if not key or "pexels" in VID_OFF:
        return []
    url = ("https://api.pexels.com/videos/search?orientation=landscape&size=medium&per_page=30&query="
           + urllib.parse.quote(query))
    try:
        vids = http(url, headers={"Authorization": key}, timeout=40).get("videos", [])
    except urllib.error.HTTPError as e:
        log(f"Pexels failed '{query}': {e}")
        if e.code == 429:
            VID_OFF.add("pexels")
        return []
    except Exception as e:
        log(f"Pexels failed '{query}': {e}")
        return []
    out = []
    for v in vids:
        slug = (v.get("url") or "").rstrip("/").split("/")[-1]  # e.g. waves-crashing-on-rocks-1234
        if v.get("duration", 0) < 4 or looks_modern(slug.replace("-", " ")):
            continue
        files = [f for f in v.get("video_files", []) if f.get("width") and 1280 <= f["width"] <= 2560
                 and f.get("file_type") == "video/mp4"]
        if files:
            f = sorted(files, key=lambda f: f["width"])[0]
            out.append((f"px{v['id']}", f["link"], f"Video by {v.get('user', {}).get('name', 'Pexels')} on Pexels"))
    return out


def pixabay_search(query):
    key = os.getenv("PIXABAY_API_KEY")
    if not key or "pixabay" in VID_OFF:
        return []
    url = (f"https://pixabay.com/api/videos/?key={key}&safesearch=true&per_page=30&q="
           + urllib.parse.quote(query[:100]))
    try:
        hits = http(url, timeout=40).get("hits", [])
    except urllib.error.HTTPError as e:
        log(f"Pixabay failed '{query}': {e}")
        if e.code == 429:
            VID_OFF.add("pixabay")
        return []
    except Exception as e:
        log(f"Pixabay failed '{query}': {e}")
        return []
    out = []
    for v in hits:
        if v.get("duration", 0) < 4 or looks_modern(v.get("tags")):
            continue
        files = v.get("videos", {})
        for size in ("medium", "large", "small"):
            f = files.get(size) or {}
            if f.get("url") and 1200 <= (f.get("width") or 0) <= 2600:
                out.append((f"pb{v['id']}", f["url"], f"Video by {v.get('user', 'Pixabay')} on Pixabay"))
                break
    return out


def stock_video(queries, used, alt=0):
    """First unused clip for the queries; Pexels and Pixabay take turns so neither hits its limit."""
    sources = [("px", pexels_search), ("pb", pixabay_search)]
    if alt % 2:
        sources.reverse()
    for q in [q for q in queries if q]:
        for tag, fn in sources:
            if (tag, q) not in VID_CACHE:
                VID_CACHE[(tag, q)] = fn(q)
            for c in VID_CACHE[(tag, q)]:
                if c[0] not in used:
                    return c
    return None


def commons_video(query, used, dur, out):
    """Real historical film (newsreels etc.) from Wikimedia Commons, cut straight from the file."""
    q = urllib.parse.quote(f"{query} filetype:video")
    url = ("https://commons.wikimedia.org/w/api.php?action=query&format=json&generator=search"
           f"&gsrsearch={q}&gsrnamespace=6&gsrlimit=8&prop=imageinfo&iiprop=url|size|extmetadata")
    try:
        time.sleep(1.2)
        pages = http(url, timeout=40).get("query", {}).get("pages", {})
    except Exception as e:
        log(f"Commons video failed '{query}': {e}")
        return None
    for p in sorted(pages.values(), key=lambda p: p.get("index", 99)):
        ii = (p.get("imageinfo") or [{}])[0]
        meta = ii.get("extmetadata", {})
        lic = meta.get("LicenseShortName", {}).get("value", "")
        src = ii.get("url")
        if (not src or src in used or ii.get("width", 0) < 400
                or not any(k in lic.lower() for k in OK_LICENSES) or "-nd" in lic.lower() or " nc" in lic.lower()):
            continue
        length = float(ii.get("duration") or 0)
        start = length * 0.12 if length > dur * 2 else 0  # skip title cards at the start of old films
        try:
            run(["ffmpeg", "-y", "-user_agent", UA, "-ss", f"{start:.1f}", "-i", src, "-t", f"{dur + 1:.2f}",
                 "-an", "-vf", f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H}",
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", out], timeout=300)
            if duration(out) < min(3, dur):
                continue
        except Exception as e:
            log(f"Commons film skip {p.get('title')}: {str(e)[:120]}")
            continue
        artist = re.sub(r"<[^>]+>", "", meta.get("Artist", {}).get("value", ""))[:60].strip() or "Unknown"
        return src, f"{p['title'].replace('File:', '')} - {artist} ({lic}), Wikimedia Commons"
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


SETTING = ""
AI_OFF = {"off": False}


def ai_image(prompt, path, seed):
    acc, tok = os.getenv("CF_ACCOUNT_ID"), os.getenv("CF_API_TOKEN")
    if not (acc and tok) or AI_OFF["off"]:
        return False
    full = (prompt + (f". Setting: {SETTING}" if SETTING else "") + ", photorealistic cinematic film still, dramatic moody lighting, 35mm film, "
            "high detail, muted colors, 16:9 composition")[:2000]
    for model in [os.getenv("CF_IMAGE_MODEL") or "@cf/black-forest-labs/flux-1-schnell"]:
        for attempt in range(3):
            try:
                r = http(f"https://api.cloudflare.com/client/v4/accounts/{acc}/ai/run/{model}",
                         {"prompt": full, "steps": 4},
                         headers={"Authorization": f"Bearer {tok}"}, timeout=120)
                img = r.get("result", {}).get("image")
                if img:
                    Path(path).write_bytes(base64.b64decode(img))
                    Image.open(path).convert("RGB").save(path, quality=95)
                    return True
            except urllib.error.HTTPError as e:
                msg = e.read()[:200]
                log(f"AI image HTTP {e.code}: {msg!r}")
                if b"daily free allocation" in msg:
                    AI_OFF["off"] = True
                    log("Cloudflare daily free limit reached - no more AI pictures today")
                    return False
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


def gather_visuals(scenes, work, offline, durs):
    """Real video clips for almost every scene; archive photos only for named people/places,
    AI pictures only as a last resort."""
    used_img, used_vid, credits = set(), set(), []
    stats = {"film": 0, "video": 0, "archive": 0, "image": 0}
    for i, sc in enumerate(scenes):
        want = sc.get("type", "video")
        order = {"video": ["film", "video", "archive", "image"], "archive": ["archive", "film", "video", "image"],
                 "image": ["video", "image", "archive"]}.get(want, ["film", "video", "archive", "image"])
        if i > 0 and scenes[i - 1].get("kind") == "image" and "video" in order:  # never two stills in a row
            order = ["film", "video"] + [k for k in order if k not in ("film", "video")]
        if not sc.get("prompt"):
            sc["prompt"] = sc.get("query") or sc.get("archive") or ""
        sc["asset"], sc["kind"] = None, None
        if not offline:
            for kind in order:
                try:
                    if kind == "film" and sc.get("film"):
                        p = work / f"h_{i:03d}.mp4"
                        got = commons_video(sc["film"], used_vid, durs[i], p)
                        if got:
                            used_vid.add(got[0])
                            credits.append(got[1])
                            sc["asset"], sc["kind"] = p, "video"
                    elif kind == "video":
                        q = sc.get("query") or ""
                        got = stock_video([q, sc.get("query2"), " ".join(q.split()[:2])], used_vid, i)
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
                    stats[kind] += 1
                    break
        if not sc["asset"] and offline:
            p = work / f"p_{i:03d}.jpg"
            placeholder(p, sc.get("query") or sc.get("archive") or "history", i)
            sc["asset"], sc["kind"] = p, "image"
        if i % 10 == 0:
            log(f"Visuals {i + 1}/{len(scenes)}")
    real = [s for s in scenes if s.get("asset")]
    log(f"Visuals for {len(real)}/{len(scenes)} scenes - historical film {stats['film']}, stock video "
        f"{stats['video']}, archive photos {stats['archive']}, AI pictures {stats['image']}")
    if not offline and len(real) < max(5, 0.5 * len(scenes)):
        raise RuntimeError(f"Only {len(real)} real visuals found - refusing to make a low-quality video. "
                           "Add PEXELS_API_KEY / PIXABAY_API_KEY secrets.")
    vids = [s for s in real if s["kind"] == "video"] or real
    for i, sc in enumerate(scenes):  # fill gaps by re-using real footage (different part of the clip)
        if not sc.get("asset"):
            src = vids[(i * 7) % len(vids)]
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


def video_clip(src, dur, out, w=W, h=H, idx=0):
    """Fits a real clip to the scene: a short clip plays in gentle slow motion instead of visibly looping,
    a long one starts at a different point each time it is used."""
    try:
        have = duration(src)
    except Exception:
        have = dur
    slow, start = "", 0.0
    if 0 < have < dur <= have * 1.8:
        slow = f"setpts={dur / have:.4f}*PTS,"
    elif have > dur + 1:
        start = (have - dur) * ((idx * 0.37) % 1)
    vf = (f"{slow}scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},fps={FPS},"
          "eq=saturation=0.78:contrast=1.06:brightness=-0.02,format=yuv420p")
    run(["ffmpeg", "-y", "-ss", f"{start:.2f}", "-stream_loop", "-1", "-i", src, "-t", f"{dur:.3f}", "-vf", vf,
         "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", out])


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


SFX_CREDITS = []


def commons_audio(query, out):
    """Real recorded sound effect from Wikimedia Commons (free licence), converted to wav."""
    q = urllib.parse.quote(f"{query} sound filetype:audio")
    url = ("https://commons.wikimedia.org/w/api.php?action=query&format=json&generator=search"
           f"&gsrsearch={q}&gsrnamespace=6&gsrlimit=10&prop=imageinfo&iiprop=url|size|extmetadata|mime")
    try:
        time.sleep(1.2)
        pages = http(url, timeout=40).get("query", {}).get("pages", {})
    except Exception as e:
        log(f"Commons audio failed '{query}': {e}")
        return False
    for pg in sorted(pages.values(), key=lambda p: p.get("index", 99)):
        ii = (pg.get("imageinfo") or [{}])[0]
        meta = ii.get("extmetadata", {})
        lic = meta.get("LicenseShortName", {}).get("value", "")
        if not any(k in lic.lower() for k in OK_LICENSES) or "-nd" in lic.lower() or " nc" in lic.lower():
            continue
        if not ii.get("url") or not (20_000 < ii.get("size", 0) < 25_000_000):
            continue
        if "midi" in (ii.get("mime") or ""):
            continue
        title = pg.get("title", "").lower()
        if re.search(r"xc\d{4,}", title) or not any(w in title for w in query.lower().split() if len(w) > 3):
            continue  # skip bird-song recordings and unrelated files
        try:
            raw = out.with_suffix(".src")
            download(ii["url"], raw)
            run(["ffmpeg", "-y", "-i", raw, "-t", "60", "-ac", "2", "-ar", "44100", out])
            raw.unlink(missing_ok=True)
            if duration(out) < 2.5:
                continue
            artist = re.sub(r"<[^>]+>", "", meta.get("Artist", {}).get("value", ""))[:50].strip() or "Unknown"
            SFX_CREDITS.append(f"Sound: {pg['title'].replace('File:', '')} - {artist} ({lic}), Wikimedia Commons")
            return True
        except Exception as e:
            log(f"Commons audio skip {pg.get('title')}: {e}")
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
    w = work / f"cm_{slugify(name)}.wav"
    if freesound(name, p):
        out = p
    elif commons_audio(name, w) or commons_audio(" ".join(name.split()[-1:]), w):
        out = w
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
    durs = narrate(scenes, work, offline)
    total = sum(durs)
    log(f"Narration: {total / 60:.1f} min")

    log("Visuals...")
    credits = gather_visuals(scenes, work, offline, durs)

    log("Scene clips...")
    clips = []
    for i, (sc, du) in enumerate(zip(scenes, durs)):
        clip = work / f"c_{i:03d}.mp4"
        if sc["kind"] == "video":
            video_clip(sc["asset"], du, clip, idx=i)
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
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", work / "clips.txt", "-c", "copy", work / "visual.mp4"])
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
        vids = [s["asset"] for s in story["scenes"] if s.get("kind") == "video"]
        if not base_path and vids:  # mostly-video story: use a frame of an early clip
            try:
                base_path = grab(vids[min(2, len(vids) - 1)], 1.0, work / "thumb_base.jpg")
            except Exception as e:
                log(f"Thumbnail frame failed: {e}")
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
            f"Narration: {'English' if LANG == 'en' else 'Urdu'}. Some scenes are AI-generated illustrations of historical events.\n"
            f"Footage & images:\n{cred}\n" + "".join(c + "\n" for c in MUSIC_CREDIT + SFX_CREDITS[:15]) + f"\n{tags}")


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

    if results:  # a --no-upload test run does not use up the topic
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
