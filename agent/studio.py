#!/usr/bin/env python3
"""
Dark History studio (v2) - one run makes and uploads:
  * 1 long documentary (16:9, ~10-12 min, Urdu narration, English title)
  * 2 YouTube Shorts (9:16), each its own complete true story on a different topic (5 rotating styles:
    on this day / vanished / strange accident / the night the... / hidden truth), not cut from the video

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
SHORT_MIN, SHORT_MAX = 35.0, 44.5  # Shorts length in seconds
LANG = (os.getenv("NARRATION_LANG") or "en").strip().lower()   # "en" = English narration, "ur" = Urdu
TTS_VOICES = [v for v in [os.getenv("TTS_VOICE")] + (
    ["en-US-Chirp3-HD-Enceladus", "en-US-Chirp3-HD-Charon", "en-US-Chirp3-HD-Iapetus", "en-US-Studio-Q"] if LANG == "en" else
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

DARK = re.compile(r"disaster|murder|killing|killer|myster|disappear|vanish|curse|ghost|haunt|shipwreck|"
                  r"sinking|sank|wreck|explosion|plague|epidemic|famine|execut|assassinat|expedition|cannibal|"
                  r"cult\b|tragedy|catastroph|collapse|poison|witch|hoax|unsolved|unexplained|kidnap|siege|mutiny|"
                  r"abandoned|experiment|crash|flood|eruption|stampede|lost |prison|asylum|"
                  r"tomb|mummy|pirate|serial|hanged|burned|buried|torture|inquisition", re.I)
NOT_STORY = re.compile(r"footballer|cricketer|singer|rapper|actor|actress|\bfilm\b|television|tv series|season|"
                       r"album|song|video game|election|politician|company|tournament|league|\bcup\b|\bband\b|"
                       r"wrestler|youtuber|player|musician|episode|novel|comic|franchise|streamer", re.I)
OLD = re.compile(r"\b(1[0-9]{3}|200[0-5])\b|\bBC\b|century|ancient|medieval|historic", re.I)
# Topics tied to conflicts that are still alive today (or mass atrocities): never offered to the writer.
SENSITIVE = re.compile(r"israel|palestin|gaza|west bank|zionis|hamas|hezbollah|nakba|ukrain|crimea|chechn|kashmir|"
                       r"partition of india|armenian|kurd|rohingya|uyghur|tibet|taiwan|korean war|troubles|\bIRA\b|"
                       r"holocaust|genocide|ethnic cleansing|pogrom|terror|jihad|isis|taliban|al-qaeda|sectarian|"
                       r"apartheid|lynch|slave|"
                       # racial, religious or political violence: hurts ad revenue and isn't the channel's style
                       r"massacre|atrocit|race riot|racial|racist|\bmob\b|ku klux|\bkkk\b|klan|white supremac|"
                       r"neo-?nazi|civil rights|segregat|\briots?\b|uprising|insurrection|coup\b|revolution|"
                       r"mass shooting|shooting|gunman|school shooting|hostage|bombing campaign|"
                       r"religious|sectarian|church burning|mosque|synagogue|temple attack|"
                       r"election|politic|communis|fascis|militia|paramilitary|death squad", re.I)


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
            and OLD.search(f"{t} {desc.get(t, '')}") and not SENSITIVE.search(f"{t} {desc.get(t, '')}")]
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
            if SENSITIVE.search(" ".join([ev.get("text", ""), title] + [p.get("description", "") for p in pages])):
                continue
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
    LAST_WIKI["title"] = hits[0]["title"]
    log(f"Facts source: Wikipedia '{hits[0]['title']}' ({len(txt)} chars)")
    return txt[:chars]


# =============================================================== TOPIC CHOICE (views first)
# ~70% famous evergreen topics people search for every day (agent/dark_topics.json), ~30% trending ones.
# Before a topic is used, YouTube is asked how many views other channels' videos on it get: only topics
# with a proven record (hundreds of thousands of views) are taken. A topic never repeats within 6 months.

DARK_TOPICS_FILE = Path(__file__).resolve().parent / "dark_topics.json"
REPEAT_DAYS = int(os.getenv("TOPIC_REPEAT_DAYS") or 183)
EVERGREEN_SHARE = float(os.getenv("EVERGREEN_SHARE") or 0.7)
MIN_TOP_VIEWS = int(os.getenv("MIN_TOP_VIEWS") or 500_000)   # best video on the topic
MIN_100K_VIDEOS = int(os.getenv("MIN_100K_VIDEOS") or 3)     # and several videos over 100k views
YT_CHECKS = {"left": int(os.getenv("YT_VIEW_CHECKS") or 4), "off": False}  # 100 quota units per new check
GENERIC = {"mystery", "curse", "history", "death", "what", "really", "true", "story", "lost", "disappearance",
           "murder", "murders", "disaster", "sinking", "event", "incident", "the", "ghost", "ship", "treasure"}
LAST_WIKI = {"title": ""}


def topic_log():
    f = DATA / "topics_log.csv"
    if not f.exists():
        return []
    import csv
    with open(f, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def used_topics():
    """Old topics (topics_done.txt, from before the 6-month rule) + every topic of the last 6 months."""
    f = DATA / "topics_done.txt"
    old = [l.strip() for l in f.read_text(encoding="utf-8").splitlines() if l.strip()] if f.exists() else []
    since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=REPEAT_DAYS)
    recent = []
    for r in topic_log():
        try:
            if dt.datetime.fromisoformat(r["date"]) >= since:
                recent += [r.get("topic", ""), r.get("wikipedia", "")]
        except (KeyError, ValueError):
            continue
    return old + [t for t in recent if t]


def pick_mode(extra_kinds=()):
    """'evergreen' until ~70% of the last 10 videos are famous topics, otherwise 'trending'."""
    forced = (os.getenv("TOPIC_MODE") or "").strip().lower()
    if forced in ("evergreen", "trending"):
        return forced
    kinds = ([r.get("kind") for r in topic_log()] + list(extra_kinds))[-10:]
    if not kinds:
        return "evergreen"
    share = kinds.count("evergreen") / len(kinds)
    return "evergreen" if share < EVERGREEN_SHARE else "trending"


def view_cache():
    f = DATA / "topic_views.csv"
    out = {}
    if f.exists():
        import csv
        with open(f, encoding="utf-8", newline="") as fh:
            for r in csv.reader(fh):
                if len(r) >= 5:
                    out[r[0].lower()] = r  # later lines win
    return out


def topic_word(query):
    """The word a video title must contain to count as a video about this topic (usually the name)."""
    words = [w for w in re.findall(r"[a-z0-9']+", query.lower()) if len(w) >= 4 and w not in GENERIC]
    return words[0] if words else query.lower().split()[0]


def yt_demand(query):
    """Views other channels get on this topic: {"top", "n100k", "titles"} or None when YouTube can't be asked.
    Results are kept 60 days in data/topic_views.csv so the daily quota is hardly touched."""
    hit = view_cache().get(query.lower())
    if hit:
        try:
            if (dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(hit[1])).days < 60:
                return {"top": int(hit[2]), "n100k": int(hit[3]), "titles": [t for t in hit[4].split(" | ") if t]}
        except ValueError:
            pass
    key = os.getenv("YT_API_KEY") or os.getenv("GOOGLE_TTS_KEY")
    if not key or YT_CHECKS["off"] or YT_CHECKS["left"] <= 0:
        return None
    YT_CHECKS["left"] -= 1
    try:
        url = "https://www.googleapis.com/youtube/v3/search?" + urllib.parse.urlencode({
            "part": "snippet", "type": "video", "order": "viewCount", "q": query, "maxResults": 20,
            "relevanceLanguage": "en", "key": key})
        items = http(url, timeout=30).get("items", [])
        want = topic_word(query)
        titles = {it["id"]["videoId"]: html.unescape(it["snippet"]["title"]) for it in items}
        # real videos about the topic only: no fan edits / Shorts compilations riding on the name
        titles = {v: t for v, t in titles.items() if want in t.lower()
                  and not re.search(r"#shorts|#edit|\bedits?\b|#viral|#fyp|sleeping sun|you are my enemy", t, re.I)}
        views = {}
        if titles:
            url = "https://www.googleapis.com/youtube/v3/videos?" + urllib.parse.urlencode(
                {"part": "statistics", "id": ",".join(titles), "key": key})
            views = {v["id"]: int(v["statistics"].get("viewCount", 0)) for v in http(url, timeout=30).get("items", [])}
    except urllib.error.HTTPError as e:
        YT_CHECKS["off"] = True
        log(f"YouTube views check off (HTTP {e.code}) - add a YT_API_KEY secret with YouTube Data API v3 enabled")
        return None
    except Exception as e:
        log(f"YouTube views check failed for '{query}': {e}")
        return None
    ranked = sorted(titles, key=lambda v: -views.get(v, 0))
    d = {"top": views.get(ranked[0], 0) if ranked else 0, "n100k": sum(1 for v in ranked if views.get(v, 0) >= 100_000),
         "titles": [titles[v] for v in ranked[:5]]}
    import csv
    with open(DATA / "topic_views.csv", "a", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerow([query, dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), d["top"],
                                 d["n100k"], " | ".join(t.replace("|", "/") for t in d["titles"])])
    log(f"YouTube views for '{query}': best {d['top']:,}, {d['n100k']} videos over 100k")
    return d


def proven(d):
    return bool(d) and d["top"] >= MIN_TOP_VIEWS and d["n100k"] >= MIN_100K_VIDEOS


def series_list():
    return json.loads(DARK_TOPICS_FILE.read_text(encoding="utf-8")).get("series") or {}


def evergreen_pick(done, extra_series=()):
    """A famous topic from the series used least recently (series rotate), with a proven views record."""
    items = json.loads(DARK_TOPICS_FILE.read_text(encoding="utf-8"))["topics"]
    fresh = [dict(t, curated=True) for t in items if not is_used(t["name"], done)
             and not is_used(t["wikipedia"], done) and t["name"] not in REJECTED]
    if not fresh:
        raise RuntimeError("every evergreen topic was used in the last 6 months - add more to dark_topics.json")
    random.Random(dt.date.today().toordinal() + len(REJECTED)).shuffle(fresh)
    recent = [r.get("series", "") for r in topic_log()][-8:] + list(extra_series)
    last_at = lambda s: max((i for i, x in enumerate(recent) if x == s), default=-1)
    fresh.sort(key=lambda t: (recent.count(t.get("series", "")), last_at(t.get("series", ""))))
    best = None
    for t in fresh[:12]:
        d = yt_demand(t["query"])
        if d is None:
            if YT_CHECKS["off"] or not (os.getenv("YT_API_KEY") or os.getenv("GOOGLE_TTS_KEY")):
                log(f"Views check unavailable - using the well-known topic {t['name']}")
                return {**t, "kind": "evergreen", "demand": None}
            continue  # out of checks for today: only cached topics are compared
        if proven(d):
            return {**t, "kind": "evergreen", "demand": d}
        if not best or d["top"] > best["demand"]["top"]:
            best = {**t, "kind": "evergreen", "demand": d}
    if best:
        log(f"No checked evergreen topic reached the views bar - taking the strongest: {best['name']}")
        return best
    return {**fresh[0], "kind": "evergreen", "demand": None}


def trending_pick(done):
    """The AI proposes 4 stories from today's trend signals; the first with a proven views record wins."""
    avoid = "\n".join(f"- {t}" for t in done[-300:] + REJECTED) or "- (none yet)"
    series = series_list()
    series_lines = "\n".join(f"- {k}: {v}" for k, v in series.items())
    got = ask_json(f"""You choose topics for "{CHANNEL}", a YouTube channel of true dark and mysterious history.
TRENDING SIGNALS (free public data collected today):
{trend_signals() or '(no trend data today)'}

Propose 4 real, specific, verifiable dark-history stories that match these signals AND that many people
already search for on YouTube (a known name, event or place - not an obscure local incident). Our
audience loves: ship disasters, plane crashes, gangster and crime nights, city blackouts and fires,
secrets of past wars. Each story must fit one of our series:
{series_lines}
Not any of:
{avoid}
Skip living people, wars or political conflicts that are still going on, massacres of civilians, race
riots, mass shootings and any racial, religious or political violence.
Return ONLY JSON: {{"topics": [{{"name": "short famous name of the story (2-4 words)",
"wikipedia": "exact English Wikipedia article title", "query": "what people type on YouTube to find it",
"series": "one series name from the list", "why": "which signal it matches"}}]}}""", keys=("topics",), max_tokens=1200)
    for c in got.get("topics") or []:
        if not isinstance(c, dict):
            continue
        c = {k: text(c.get(k)) for k in ("name", "wikipedia", "query", "series", "why")}
        if c["series"] not in series:
            c["series"] = "Unsolved Mysteries"
        c["name"] = c["name"] or c["wikipedia"]
        if not c["name"] or is_used(c["name"], done) or SENSITIVE.search(f"{c['name']} {c['wikipedia']}"):
            continue
        d = yt_demand(c["query"] or c["name"])
        if d is None and not YT_CHECKS["off"] and YT_CHECKS["left"] > 0:
            continue
        if d is None or proven(d):
            return {**c, "wikipedia": c["wikipedia"] or c["name"], "kind": "trending", "demand": d}
        log(f"Trending topic '{c['name']}' skipped: not enough views on YouTube (best {d['top']:,})")
    return None


def promised_topic(done):
    """The topic the last uploaded video promised ("Next time on Dark History: ..."), if still unused."""
    f = DATA / "next_topic.json"
    try:
        p = json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return None
    if p.get("name") and not is_used(p["name"], done) and p["name"] not in REJECTED:
        log(f"Topic promised by the last video: {p['name']}")
        return p
    return None


def choose_topic(done, extra_kinds=(), extra_series=(), use_promise=True):
    pick = promised_topic(done) if use_promise else None
    if not pick:
        mode = pick_mode(extra_kinds)
        if mode == "trending":
            try:
                pick = trending_pick(done)
            except Exception as e:
                log(f"Trending topic search failed: {e}")
            if not pick:
                log("No trending topic with a proven views record today - using a famous evergreen topic")
        pick = pick or evergreen_pick(done, extra_series)
    pick.setdefault("series", "Unsolved Mysteries")
    d = pick.get("demand")
    log(f"Topic ({pick['kind']}, series {pick['series']}): {pick['name']}"
        + (f" - best YouTube video {d['top']:,} views, {d['n100k']} over 100k" if d else " - views not checked"))
    return pick


def fit_title(title, name, angle=""):
    """A complete title of at most 60 characters that starts with the famous name (asks the AI to shorten a
    long one instead of cutting it mid-phrase)."""
    title = re.sub(r"\s+", " ", title).strip()
    for _ in range(2):
        if len(title) <= 60:
            return title
        try:
            got = ask_json(f"""Shorten this YouTube title to at most 55 characters. Keep it complete, truthful and
curious; it must start with "{name}: ". Angle of the video: {angle or '(see title)'}
Title: {title}
Return ONLY JSON: {{"title": "..."}}""", keys=("title",), max_tokens=300)
            new = text(got.get("title"))
            if new.lower().startswith(name.lower()):
                title = new
        except Exception as e:
            log(f"Title shortening failed: {e}")
            break
    if len(title) > 60:  # last resort: end at a natural break, never in the middle of a phrase
        head = title[:60]
        cut = max(head.rfind("?"), head.rfind("!"), head.rfind(". "))  # a whole sentence first
        if cut <= len(name) + 5:
            cut = max(head.rfind(" - "), head.rfind(","), head.rfind(":"))
        title = head[:cut + 1].rstrip(" ,-:") if cut > len(name) + 5 else name
    return title


def short_title(title, name):
    """A Shorts title: a secret / question ('The Night the Mob Came'), never starting with the famous name
    (that goes in the description and hashtags), at most 60 characters, never cut mid-word."""
    t = re.sub(r"\s+", " ", title or "").strip().strip('"')
    t = re.sub(rf"^\W*{re.escape(name)}\s*(?:[:\-–—|]\s*|'s\s+)?", "", t, flags=re.I).strip()
    t = re.sub(r"#\w+", "", t).strip()
    if len(t) > 60:
        t = t[:60].rsplit(" ", 1)[0].rstrip(" ,-:")
    return t[:1].upper() + t[1:] if t else ""


STOP = {"the", "and", "that", "this", "with", "from", "they", "were", "their", "which", "when", "what", "into",
        "after", "about", "there", "would", "could", "have", "been", "over", "only", "then", "than", "them",
        "chapter", "story", "night", "video", "while", "where", "these", "those", "some", "more", "most"}


def fact_paragraphs(full):
    """Article paragraphs (section headings '== X ==' are kept with the paragraph after them)."""
    out, head = [], ""
    for p in (full or "").split("\n"):
        p = p.strip()
        if not p:
            continue
        if re.fullmatch(r"=+\s*[^=]+?\s*=+", p):
            head = p.strip("= ").strip()
            continue
        out.append((head, p))
    return out


def fact_digest(full, limit=7000):
    """The whole article in short: the lead, then the start of every section (for planning the chapters)."""
    paras = fact_paragraphs(full)
    out, seen = [], set()
    for head, p in paras:
        if not head:
            out.append(p)
        elif head not in seen:
            seen.add(head)
            out.append(f"[{head}] {p[:450]}")
    txt = ""
    for p in out:
        if len(txt) + len(p) + 1 > limit:
            break
        txt += p + "\n"
    return txt.strip() or (full or "")[:limit]


def relevant_facts(full, about, limit=7000):
    """The article's lead + the paragraphs that share the most key words with `about` (a chapter plan,
    the narration or a title), in the article's own order."""
    paras = fact_paragraphs(full)
    if not paras:
        return (full or "")[:limit]
    words = {w for w in re.findall(r"[a-z0-9]{4,}", (about or "").lower()) if w not in STOP}
    lead = [i for i, (h, _) in enumerate(paras) if not h][:4]
    score = {i: len(words & set(re.findall(r"[a-z0-9]{4,}", p.lower()))) for i, (_, p) in enumerate(paras)}
    take, size = set(lead), sum(len(paras[i][1]) for i in lead)
    for i in sorted(score, key=lambda i: -score[i]):
        if i in take or score[i] == 0:
            continue
        if size + len(paras[i][1]) > limit:
            continue
        take.add(i)
        size += len(paras[i][1])
    return "\n".join((f"[{paras[i][0]}] " if paras[i][0] else "") + paras[i][1] for i in sorted(take))[:limit + 2000]


def verify_title(title, full, name, short=False):
    """The title must say only what the facts back up (with words copied from the article as proof).
    If not, the AI rewrites it and it is checked again; a title that cannot be backed up stops the upload."""
    rule = ("a secret / question title like 'The Night the ...' or 'What Killed ...', max 50 characters, NOT "
            f"starting with '{name}'" if short else f"it must start with '{name}: ', max 60 characters, Title Case")
    for attempt in range(3):
        facts = relevant_facts(full, title, 7000)
        try:
            got = ask_json(f"""Check this YouTube title against the SOURCE FACTS of a history channel.
TITLE: {title}
The title is NOT supported if it states or clearly implies something the facts do not say - a cause, a
result, a number, a "secret", a blame - or presents a theory as proven. A question is fine if the facts
discuss that question.
SOURCE FACTS (Wikipedia):
{facts}
Return ONLY JSON: {{"supported": true or false, "problem": "short reason if not supported",
"quote": "8-25 words copied EXACTLY from the SOURCE FACTS that back up the title",
"better_title": "only if not supported: a gripping, completely true title - {rule}"}}""",
                           keys=("supported",), max_tokens=800)
        except Exception as e:
            log(f"Title check failed ({e})")
            got = {}
        ok = got.get("supported") is True and quote_found(text(got.get("quote")), full)
        if ok:
            if attempt:
                log(f"Title backed by the facts after rewriting: '{title}'")
            return title
        log(f"Title not backed by the facts: '{title}' ({text(got.get('problem'), 'no proof in the article')[:150]})")
        new = text(got.get("better_title"))
        if short:
            new = short_title(new, name)
        elif new and not new.lower().startswith(name.lower()):
            new = f"{name}: " + re.sub(rf"^\W*{re.escape(name)}\W*", "", new, flags=re.I)
        if not new or new == title or len(new) > 60:
            continue
        title = new
    raise RuntimeError(f"title could not be backed by the facts - not uploading: '{title}'")


def quote_found(quote, facts):
    """The words (or almost all of them, in order) appear in the source text."""
    import difflib
    norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", (s or "").lower())).strip()
    q, f = norm(quote), norm(facts)
    if len(q.split()) < 4:
        return False
    if q in f:
        return True
    m = difflib.SequenceMatcher(None, f, q, autojunk=False).find_longest_match(0, len(f), 0, len(q))
    return m.size >= 0.8 * len(q)


def fact_check(scenes, facts, title=""):
    """One AI pass that compares every narration line (and the title) with the Wikipedia facts and rewrites
    any line with a name, date, number, place or event the facts do not support.
    Returns (lines fixed, corrected title or "")."""
    if not facts:
        return 0, ""
    numbered = "\n".join(f"{i}. {s['ur']}" for i, s in enumerate(scenes))
    title_part = (f"""TITLE: {title}
Also check the TITLE: if it states something the facts do not support (or a disputed theory as proven),
give a corrected title (same start before the colon, max 60 characters, Title Case, still gripping -
best as a question or "The Night the ..." / "What Killed ..." - never a flat list like "X and Y").
""" if title else "")
    try:
        got = ask_json(f"""You are the fact-checker of a history documentary. Compare EVERY numbered narration line
with the SOURCE FACTS. A line is wrong if it states a name, date, number, place, quote or event that the
facts do not support, or presents a legend or theory as proven fact. Dramatic wording is fine; invented
facts are not. For each wrong line give a corrected line of THE SAME LENGTH that only uses supported
facts (or clearly says "according to legend" / "one theory says"): keep the drama and the storytelling,
replace only the invented detail with a true detail from the facts. Never just shorten the line.
List ONLY lines that are really wrong - never a line that is fine or "acceptable".
SOURCE FACTS (Wikipedia):
{facts}

NARRATION:
{numbered}
{title_part}Return ONLY JSON: {{"fixes": [{{"line": 3, "problem": "short reason", "text": "corrected line"}}],
"title_fix": "corrected title, or empty if the title is fine"}}
(an empty fixes list if everything is supported)""", keys=("fixes",), max_tokens=3000)
    except Exception as e:
        log(f"Fact check skipped: {e}")
        return 0, ""
    n, before = 0, sum(len(s["ur"].split()) for s in scenes)
    for fx in got.get("fixes") or []:
        try:
            i = int(fx.get("line"))
        except (TypeError, ValueError, AttributeError):
            continue
        new = text(fx.get("text"))
        if (0 <= i < len(scenes) and new and new.strip() != scenes[i]["ur"].strip()
                and 0.5 <= len(new) / max(1, len(scenes[i]["ur"])) <= 1.6):
            log(f"Fact fix line {i}: {text(fx.get('problem'))[:120]}")
            scenes[i]["ur"] = new
            n += 1
    log(f"Fact check: {n} line(s) corrected ({before} -> {sum(len(s['ur'].split()) for s in scenes)} words)")
    tfix = text(got.get("title_fix"))
    head = title.split(":")[0].strip().lower()
    if title and tfix and tfix != title and len(tfix) <= 60 and tfix.lower().startswith(head):
        log(f"Fact check: title changed from '{title}' to '{tfix}'")
        return n, tfix
    return n, ""


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
- "film": real old film footage is the BEST visual. If the story happens after 1895, give "film" for
  as many scenes as possible: 2-4 English words to search Wikimedia Commons videos - the exact event if
  filmed ("Titanic 1912 film"), otherwise period footage of the same thing ("1950s newsreel airliner",
  "1920s steamship harbor", "World War II bomber"). Before 1895 or nothing fitting: "".
- The clip must SHOW what the line says (the object, place or action named in it), not just a mood.
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


def looks_modern(tags, query=""):
    """Modern things are rejected unless the scene asked for exactly that thing (e.g. 'military aircraft')."""
    words = set(re.split(r"[^a-z]+", (tags or "").lower()))
    return bool((words & MODERN) - set(re.split(r"[^a-z]+", query.lower())))


# Clips that make a documentary look cheap: always rejected, whatever the scene asked for.
BAD_CLIP = re.compile(r"cartoon|animat|anime|green.?screen|chroma|blue.?screen|\b3d\b|render|cgi|vector|"
                      r"illustrat|motion.?graphic|character|emoji|mascot|plasma|laser|hologram|futuristic|sci.?fi|"
                      r"glitch|particle|abstract|bokeh|loop(able)?\b|seamless|logo|\btext\b|\btitle|typograph|lettering|"
                      r"countdown|subscribe|banner|intro|outro|overlay|transition|template|mockup|screen\b|monitor|"
                      r"subway|metro|escalator|station|mall|airport|crowd|pedestrian|commuter|tourist|walking people|"
                      r"people walking|covid|corona|face mask|vr\b|robot|toy|kids?\b|children playing|wedding|"
                      r"yoga|fitness|workout|cooking|food|coffee|smiling|happy|cheerful|business|"
                      r"lava|liquid|ink\b|paint|macro|kaleidoscope|fractal|tunnel|light.?leak|lens.?flare|"
                      r"wallpaper|screensaver|visuali[sz]er|psychedelic|colou?rful|rainbow|disco|led\b", re.I)
# Archive films that carry a burned-in watermark / timecode from a reseller (e.g. "PF# ... PeriscopeFilm.com").
BURNED_IN = re.compile(r"periscope|timecode|time code|watermark|preview copy|stock footage library|"
                       r"criticalpast|british pathe|reuters|getty", re.I)
STOP = {"old", "dark", "the", "a", "an", "of", "in", "on", "at", "and", "with", "night", "scene", "shot",
        "footage", "video", "film", "clip", "close", "up", "view", "slow", "motion", "vintage", "historic",
        "historical", "cinematic", "dramatic", "moody", "background"}


def relevance(query, desc):
    """How many of the query's meaningful words (by 4-letter stem) appear in the clip's name/tags."""
    have = {w[:4] for w in re.split(r"[^a-z0-9]+", (desc or "").lower()) if len(w) > 2}
    want = {w[:4] for w in re.split(r"[^a-z0-9]+", query.lower()) if len(w) > 2 and w not in STOP}
    return len(want & have)


def clip_ok(desc, query):
    return not BAD_CLIP.search(desc or "") and not looks_modern(desc, query)


def is_used(topic, done):
    key = slugify(re.sub(r"\b(the|disappearance|mystery|tragedy|disaster)\b", "", topic.lower()))
    return any(t.strip().lower() == topic.strip().lower() or
               (len(key) >= 5 and (key in slugify(t) or slugify(t) in key)) for t in done)


def text(v, default=""):
    """AI fields sometimes come back as null, numbers or lists - always give back a clean string."""
    if isinstance(v, list):
        v = " ".join(str(x) for x in v if x)
    v = "" if v is None else str(v).strip()
    return v or default


def normalize_outline(o, n_ch=5):
    """Fills every field the rest of the pipeline reads, so a missing or oddly-shaped field in the AI
    answer gets a sensible default instead of a KeyError."""
    o["topic"] = text(o.get("topic")) or text(o.get("title")) or text(o.get("wikipedia"), "Dark History Story")
    o["title"] = text(o.get("title"), o["topic"])[:100]
    o["wikipedia"] = text(o.get("wikipedia"), o["topic"])
    o["thumbnail_text"] = text(o.get("thumbnail_text")) or " ".join(o["title"].split()[:3])
    o["setting"] = text(o.get("setting"))
    o["thumbnail_prompt"] = text(o.get("thumbnail_prompt"), f"{o['title']}, {o['setting']}".strip(", "))
    o["description"] = text(o.get("description"), f"The true story of {o['topic']}.")
    for k, default in (("tags", [o["topic"], "dark history", "history", "mystery"]),
                       ("hashtags", ["#history", "#mystery", "#darkhistory"])):
        v = o.get(k)
        v = [s.strip() for s in v.split(",")] if isinstance(v, str) else v
        o[k] = [text(x) for x in v if text(x)] if isinstance(v, list) and v else default
    o["hashtags"] = [h if h.startswith("#") else "#" + h.replace(" ", "") for h in o["hashtags"]]
    chs = o.get("chapters")
    if not isinstance(chs, list) or not chs:
        chs = [{} for _ in range(n_ch)]
    fixed = []
    for i, ch in enumerate(chs):
        ch = ch if isinstance(ch, dict) else {"title": text(ch)}
        title = text(ch.get("title")) or text(ch.get("name"), f"Part {i + 1}")
        summary = (text(ch.get("summary")) or text(ch.get("description")) or text(ch.get("content"))
                   or f"Part {i + 1} of the story of {o['topic']}: {title}.")
        fixed.append({**ch, "title": title, "summary": summary})
    o["chapters"] = fixed
    shorts = []
    for sh in o.get("shorts") if isinstance(o.get("shorts"), list) else []:
        sh = sh if isinstance(sh, dict) else {}
        try:
            c = min(max(int(sh.get("chapter", 0)), 0), len(fixed) - 1)
        except (TypeError, ValueError):
            c = 0
        shorts.append({"chapter": c, "title": text(sh.get("title"), fixed[c]["title"])[:90],
                       "hook": text(sh.get("hook")), "loop": text(sh.get("loop")), "angle": text(sh.get("angle"))})
    o["shorts"] = shorts or [{"chapter": c, "title": fixed[c]["title"], "hook": "", "loop": "", "angle": ""}
                             for c in sorted({0, min(2, len(fixed) - 1)})]
    o["hook"] = text(o.get("hook"))
    scored = []
    for h in o.get("hooks") if isinstance(o.get("hooks"), list) else []:
        try:
            if isinstance(h, dict) and text(h.get("text")):
                scored.append((float(h.get("score") or 0), text(h.get("text"))))
        except (TypeError, ValueError):
            continue
    if scored:  # the best-scoring cold open wins, even if the AI put a weaker one in "hook"
        best = max(scored)
        if best[0] >= 7 or not o["hook"]:
            o["hook"] = best[1]
    if len(o["title"]) > 60:  # keep it readable on a phone: cut at a word boundary
        cut = o["title"][:60].rsplit(" ", 1)[0].rstrip(" :,-–")
        o["title"] = cut if len(cut) >= 30 else o["title"][:60]
    title_words = {norm_word(w) for w in o["title"].split()}
    thumb = [w for w in o["thumbnail_text"].split() if norm_word(w) not in title_words]
    if len(thumb) >= 2 and len(thumb) < len(o["thumbnail_text"].split()):
        o["thumbnail_text"] = " ".join(thumb)  # thumbnail text must add curiosity, not repeat the title
    o["hook_query"] = text(o.get("hook_query"), "dark storm night")
    o["hook_film"] = text(o.get("hook_film"))
    try:
        o["reveal_chapter"] = min(max(int(o.get("reveal_chapter")), 1), len(fixed) - 1)
    except (TypeError, ValueError):
        o["reveal_chapter"] = max(1, int(len(fixed) * 0.75 + 0.5) - 1)  # biggest reveal at ~70-80% of the video
    return o


REJECTED = []  # topics turned down in this run (repeated / sensitive), so the retry picks another
GREETING = re.compile(r"\b(welcome|hello|hi there|hey guys|assalam|salaam|in this video|in today's video|"
                      r"subscribe|my name is)\b", re.I)


def scene_list(ch):
    """Scenes from one chapter of the AI answer, accepting the usual variations of field names."""
    raw = ch.get("scenes") if isinstance(ch, dict) else ch
    out = []
    for sc in raw if isinstance(raw, list) else []:
        if isinstance(sc, str):
            sc = {"ur": sc}
        if not isinstance(sc, dict):
            continue
        sc["ur"] = text(sc.get("ur")) or text(sc.get("narration")) or text(sc.get("text"))
        if not sc["ur"]:
            continue
        sc["type"] = text(sc.get("type"), "video").lower()
        for k in ("query", "query2", "film", "archive", "prompt", "sfx"):
            sc[k] = text(sc.get(k))
        sc["query"] = sc["query"] or sc["query2"] or "dark fog night"
        sc["sfx"] = sc["sfx"] or "dark drone"
        sc["hit"] = sc.get("hit") in (True, "true", "True", 1)
        out.append(sc)
    return out


def write_story(minutes):
    """3-4 AI requests in total: 1 outline (picks a trending topic) + the narration 2 chapters at a time."""
    done = used_topics()
    n_ch = max(4, round(minutes / 2.3))
    pick = choose_topic(done)
    name = pick["name"]
    # the NEXT video's topic is chosen now, so this video can honestly tell viewers what comes next
    nxt, nxt_facts = None, ""
    try:
        nxt = choose_topic(done + [name, pick.get("wikipedia", "")], extra_kinds=[pick["kind"]],
                           extra_series=[pick["series"]], use_promise=False)
        nxt_facts = wiki_extract(nxt["wikipedia"] or nxt["name"], chars=1500)
    except Exception as e:
        log(f"Next topic not chosen ({e}) - the video ends with a plain subscribe line")
        nxt = None
    # the WHOLE article: every chapter, the fact check and the title check get the parts they talk about
    # (before, only the first 8000 characters were used, so the later chapters had no facts and the AI
    # filled them from imagination)
    full = wiki_extract(pick["wikipedia"] or name, chars=60000)
    facts = full[:8000]
    if len(facts) < 1500:
        REJECTED.append(name)
        raise RuntimeError(f"not enough source facts for {name} ({len(facts)} chars) - picking another topic")
    rivals = "\n".join(f'- "{t}"' for t in (pick.get("demand") or {}).get("titles", [])) or "- (not checked)"
    if pick["kind"] == "evergreen":
        kind_note = f"""This is a FAMOUS story that people search for on YouTube every day, so many viewers know the
basic version. Give it a NEW ANGLE that is still 100% true: one little-known detail, a newer theory or
finding, or "what really happened" - and the angle MUST come from the FACTS below. The most viewed
videos on this topic (do NOT copy their angle or titles):
{rivals}"""
    else:
        kind_note = f"""This story is getting attention right now ({text(pick.get('why'), 'trending')}). Popular
videos on it (do NOT copy their angle or titles):
{rivals}"""
    outline_note = ""
    for attempt in range(2):
        outline = ask_json(f"""You are the head writer of "{CHANNEL}", a faceless YouTube channel that tells
true dark and mysterious stories from real history (gripping narrator, suspense, cliffhangers, a final
twist). Today is {dt.date.today():%d %B %Y}.

TODAY'S STORY (already chosen - do not change it): {name} (Wikipedia: "{LAST_WIKI['title'] or pick['wikipedia']}")
{kind_note}

FACTS (Wikipedia, the whole article in short - the ONLY facts you may use; nothing from memory):
{fact_digest(full, 7000)}
{('NEXT VIDEO (already chosen, only for next_teaser): ' + nxt['name'] + chr(10) + 'NEXT VIDEO FACTS: ' + nxt_facts[:1200]) if nxt else ''}

Never invent facts; mark legends and theories clearly as legends/theories. Respect victims: no gore for
shock, no glorifying violence. The angle must NOT contradict what the FACTS say most historians or
investigators concluded: if the angle is a theory, the title asks it as a question or says "theory" -
never state a disputed theory as proven. {outline_note}

Plan a {minutes:.0f}-minute documentary with {n_ch} chapters, built like the best retention documentaries:
- COLD OPEN: the very first line is a jolt - a secret or a shocking question with a real name, year or
  place, max 25 words. No greeting, no "in this video", no channel name. Example of the tone:
  "In 1980, America built a secret plane to rescue 52 hostages... and then made sure nobody ever heard of it."
- Then the central QUESTION the video will answer.
- Every chapter ends on a cliffhanger; earlier chapters only hint at the answer.
- The BIGGEST reveal comes in chapter {max(1, int(n_ch * 0.75 + 0.5) - 1) + 1} of {n_ch} (70-80% into the video).
- Every chapter summary names exact people, exact dates and exact places - nothing vague.
HOOKS: write 5 cold-open candidates, one of each type: curiosity gap, bold claim, story open (drop into
the scene mid-action), shock fact, direct challenge ("You have heard this story wrong"). Score each 0-10,
2 points each for: understood in one listen (under 20 words) / creates a question or tension / specific
(name, date, place) / matches the title + thumbnail promise / true and verifiable. Use the best as "hook";
if none scores 7+, rewrite until one does.
TITLE + THUMBNAIL are ONE pair: the title STARTS with the famous name exactly - "{name}: " - then the angle
(so people searching the name find it), under 60 characters in total. After the name, prefer our two
best-performing styles when they fit truthfully: "The Night the ..." (e.g. "Hindenburg: The Night the Sky
Burned") or "What Killed ..." (e.g. "Dyatlov Pass: What Killed the Nine Hikers?"); thumbnail text 2-4 words that
do NOT repeat any title word and add a second piece of curiosity; together one clear promise the video keeps.
Return ONLY JSON:
{{"comment_question": "one short question for viewers about THIS story that invites an answer in the comments (e.g. 'Had you heard of this case before?' or 'What do you think really happened?')",
  "next_teaser": "{('one short TRUE line (max 12 words) that makes people want the NEXT video, about: ' + nxt['name'] + ' - e.g. the night a whole town vanished') if nxt else ''}",
  "next_teaser_quote": "{'5-20 words copied EXACTLY from the NEXT VIDEO FACTS that support next_teaser' if nxt else ''}",
  "angle": "the new angle of this video in one sentence (little-known detail / newer theory / what really happened)",
  "angle_quote": "8-25 words copied EXACTLY from the FACTS that support the angle",
  "hooks": [{{"type": "curiosity gap", "text": "...", "score": 8}}, "... 5 in total"],
  "hook": "the best-scoring hook (rules above)",
  "hook_query": "2-4 English words for the most dramatic stock-footage shot to open the video",
  "hook_film": "2-4 English words for real historical film of this event on Wikimedia Commons, or empty",
  "question": "the central question of the video",
  "title": "{name}: <angle>, under 60 chars in total, curiosity + truthful",
  "thumbnail_text": "2-4 English words, none of them in the title",
  "setting": "visual setting for all pictures: year/era, place, period details",
  "thumbnail_prompt": "English prompt for a dramatic photorealistic cinematic thumbnail image of the story's key moment, no text",
  "description": "English description: the FIRST 2 LINES carry the hook and the main search keyword (they show in search results), then a 3-5 line summary, then 'Sources:' with 2-4 real sources",
  "tags": ["8-15 English SEO tags mixing broad (dark history, unsolved mystery) and specific (event, place, year)"],
  "hashtags": ["#darkhistory", "#mystery", "one specific hashtag - 3 in total"],
  "chapters": [{{"title": "short English chapter name", "summary": "3-4 sentences of the facts this chapter tells, with exact names, dates, places"}}]
}}""", keys=("chapters",), max_tokens=4000)
        if quote_found(text(outline.get("angle_quote")), full):
            break
        log(f"Angle not backed by the facts ({text(outline.get('angle'))[:100]}) - asking again")
        outline_note = ("Your last angle was not backed by the FACTS. Choose an angle you can support with words "
                        "copied exactly from the FACTS.")
    else:
        log("Angle quote still not found in the facts - the fact check below will correct the narration")
    outline["topic"], outline["wikipedia"] = name, LAST_WIKI["title"] or pick["wikipedia"]
    outline["kind"], outline["trend_reason"] = pick["kind"], text(pick.get("why"))
    # the famous name first, so the video is found when people search it; never cut a title mid-phrase
    title = text(outline.get("title"), name)
    if not title.lower().startswith(name.lower()):
        title = f"{name}: " + (re.sub(rf"^\W*{re.escape(name)}\W*", "", title, flags=re.I) or title)
    outline["title"] = fit_title(title, name, text(outline.get("angle")))
    outline = normalize_outline(outline, n_ch)
    outline["series"] = pick["series"]
    q = text(outline.get("comment_question"), f"Had you heard of {name} before this video?")
    outline["comment_question"] = q if q.endswith("?") else q.rstrip(".") + "?"
    # "Next time" line: must be true - backed by words from the next topic's Wikipedia article
    teaser = text(outline.get("next_teaser")).rstrip(".!") if nxt else ""
    if nxt and not (teaser and quote_found(text(outline.get("next_teaser_quote")), nxt_facts)):
        teaser = f"the true story of {nxt['name']}" if nxt else ""
    outline["next"] = ({k: nxt.get(k) for k in ("name", "wikipedia", "query", "series", "kind", "demand", "why")}
                       | {"teaser": teaser}) if nxt else None
    src = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(outline['wikipedia'].replace(' ', '_'))}"
    if "wikipedia.org" not in outline["description"]:
        outline["description"] = outline["description"].rstrip() + f"\n\nSource: {src}"
    outline["description"] = (outline["description"].rstrip() + f"\n\n{outline['comment_question']} "
                              "Tell us in the comments.\n\n" + f"Series: {pick['series']}"
                              + (f"\nNext time: {teaser[:1].upper() + teaser[1:]}." if teaser else ""))
    if SENSITIVE.search(" ".join([outline["topic"], outline["title"], outline["wikipedia"]])) and not pick.get("curated"):
        REJECTED.append(name)
        raise RuntimeError(f"topic tied to a living conflict, picking another: {name}")
    log(f"Topic: {name} ({pick['kind']}) | {outline['title']} | angle: {text(outline.get('angle'))[:160]}")

    LANG_RULE = ("simple, clear spoken English like a gripping documentary narrator (short everyday "
                 "words, easy for non-native listeners; no complicated vocabulary)." if LANG == "en" else
                 "natural spoken Urdu (Urdu script, simple everyday Urdu like a popular storyteller; "
                 "English words only for names/places/terms).")
    chapters = outline["chapters"]
    words_per_ch = int(minutes * (150 if LANG == "en" else 135) / len(chapters))
    scenes, previous = [], ""

    reveal = outline["reveal_chapter"]

    def role(i):
        """What this chapter must do in the story arc."""
        if i == 0:
            r = (f"OPENS the video right after the cold open line \"{outline['hook']}\" (already spoken - do not "
                 f"repeat it). No greeting, no 'in this video', no channel name. Its first scene asks the "
                 f"central question: {text(outline.get('question'), 'what really happened?')}")
        elif i == reveal:
            r = "contains the BIGGEST REVEAL of the whole video - the answer the viewer has been waiting for"
        elif i < reveal:
            r = "only HINTS at the answer, adds a new clue or twist"
        else:
            r = "the aftermath: what happened next and what it means today"
        if i == len(chapters) - 1:
            r += ("; it ends with a chilling final thought (a 'next time' + subscribe line is added after it "
                  "automatically - do NOT ask viewers to like or subscribe yourself)")
        else:
            r += "; its LAST scene is a cliffhanger line (e.g. 'But the real truth was still hidden...')"
        return r

    n_sc = max(6, round(words_per_ch / 26))  # AI models follow a scene count far better than a word count

    def write_part(batch, note=""):
        """Narration for these chapters -> {chapter index: [scenes]} (missing chapters are simply absent)."""
        plan = "\n".join(f"{n + 1}. {chapters[i]['title']} - {chapters[i]['summary']}\n   ROLE: {role(i)}"
                         for n, i in enumerate(batch))
        ch_facts = relevant_facts(full, plan, 7000)  # the parts of the article these chapters are about
        part = ask_json(f"""You write the narration of chapters {batch[0] + 1}-{batch[-1] + 1} of {len(chapters)} of the
YouTube documentary "{outline['title']}" (story: {outline['topic']}).
Chapters to write now:
{plan}
Story so far (last lines): {previous[-500:] or '(start of the video)'}
FACTS (Wikipedia - the parts of the article about these chapters; if the chapter plan disagrees with
them, the facts win):
{ch_facts or '(none - only use facts you are sure of)'}
TRUTH RULE: every name, date, time, number, place, quote and event you write must be in these FACTS.
Where the facts give no detail, tell it with general atmosphere (the dark sea, the waiting, the fear)
WITHOUT inventing details - no made-up times of day, counts, words someone said, weather, ranks or
thoughts. Never put words in anyone's mouth unless the FACTS quote them.

LENGTH: EACH chapter has EXACTLY {n_sc} scenes, every scene 2 full sentences (about 26 words), so each
chapter is about {words_per_ch} words. Do not shorten - the video must run {minutes:.0f} minutes. {note}
Language: {LANG_RULE}
Write it the way a great narrator TALKS, not like an essay: varied sentence length, contractions
(it's, didn't), natural connectors (but, and then, so), a few direct questions to the viewer,
"..." only for real dramatic pauses (max one per scene), "!" rarely. No headings, no lists.
Follow each chapter's ROLE exactly. Every chapter names at least one exact person, one exact date
(day, month, year) and one exact place from the FACTS - never "a man", "one day", "somewhere".
{VISUAL_RULES}
{SFX_RULES}
Return ONLY JSON with one entry per chapter, in order:
{{"chapters": [{{"scenes": [{{"ur": "narration", "type": "video|archive|image", "query": "...", "query2": "...",
"film": "", "archive": "...", "prompt": "...", "sfx": "...", "hit": false}}]}}]}}""",
                        max_tokens=min(6000, 1200 + int(len(batch) * words_per_ch * 5)))
        got = part.get("chapters")
        if not isinstance(got, list):  # answer came back as one flat {"scenes": [...]}
            got = [part] if len(batch) == 1 or "scenes" in part else []
        return {i: s for i, ch in zip(batch, got) if (s := scene_list(ch))}

    for b0 in range(0, len(chapters), 2):
        batch = list(range(b0, min(b0 + 2, len(chapters))))
        try:
            got = write_part(batch)
        except Exception as e:
            log(f"Chapters {batch[0] + 1}-{batch[-1] + 1} failed ({e}), writing them one by one")
            got = {}
        for i in batch:
            if i not in got:  # one extra request only for a chapter the AI skipped
                try:
                    got.update(write_part([i]))
                except Exception as e:
                    log(f"Chapter {i + 1} failed again: {e}")
            said = sum(len(sc["ur"].split()) for sc in got.get(i, []))
            if i in got and said < 0.65 * words_per_ch:  # far too short: one rewrite of just this chapter
                log(f"Chapter {i + 1} only {said} of ~{words_per_ch} words, asking for the full length")
                try:
                    longer = write_part([i], f"Your last version of this chapter had only {said} words - "
                                             f"write the full {n_sc} scenes this time.")
                    if sum(len(sc["ur"].split()) for sc in longer.get(i, [])) > said:
                        got[i] = longer[i]
                except Exception as e:
                    log(f"Longer chapter {i + 1} failed, keeping the short one: {e}")
            for sc in got.get(i, []):
                sc["chapter"] = i
                scenes.append(sc)
            log(f"Chapter {i + 1}: {len(got.get(i, []))} scenes")
        previous = " ".join(s["ur"] for s in scenes[-3:])
    if len({s["chapter"] for s in scenes}) < max(2, len(chapters) - 1):
        raise RuntimeError(f"Only {len({s['chapter'] for s in scenes})} of {len(chapters)} chapters written")
    # cold open: the hook line is always the very first thing heard, over the most dramatic shot
    opening = [s for s in scenes[:4] if s["chapter"] == 0]
    scenes = [s for s in scenes if s not in opening or not (GREETING.search(s["ur"]) or CHANNEL.lower() in s["ur"].lower())]
    if outline["hook"]:
        scenes.insert(0, {"ur": outline["hook"], "type": "video", "query": outline["hook_query"],
                          "query2": "dark storm night", "film": outline["hook_film"], "archive": "", "prompt":
                          outline["thumbnail_prompt"], "sfx": "dark drone", "hit": True, "chapter": 0,
                          "cold_open": True})
    for i in range(len(chapters)):
        said = " ".join(s["ur"] for s in scenes if s["chapter"] == i)
        if said and not re.search(r"\b1[0-9]{3}\b|\b20[0-2][0-9]\b", said):
            log(f"Note: chapter {i + 1} has no exact year in it")
    said_all = " ".join(s["ur"] for s in scenes)
    _, tfix = fact_check(scenes, relevant_facts(full, said_all, 10000), outline["title"])
    if tfix:
        outline["title"] = tfix
    # the title must be backed by the facts too - rewritten if not; no upload if it can't be fixed
    outline["title"] = verify_title(outline["title"], full, name)
    if scenes and scenes[0].get("cold_open"):
        outline["hook"] = scenes[0]["ur"]  # the Shorts and title checks use the corrected cold open
    # last words of every video: a true "next time" + a reason to subscribe (added after the fact check,
    # which only knows this story's facts)
    nt = (outline.get("next") or {}).get("teaser")
    end = (f"Next time on {CHANNEL}: {nt}. Subscribe so you don't miss it." if nt else
           f"Subscribe to {CHANNEL} for the next true story history tried to forget.")
    last = scenes[-1] if scenes else {}
    scenes.append({"ur": end, "type": "video", "query": last.get("query") or "dark storm night",
                   "query2": "dark fog night", "film": "", "archive": "", "prompt": outline["thumbnail_prompt"],
                   "sfx": "dark drone", "hit": False, "chapter": last.get("chapter", len(chapters) - 1),
                   "outro": True})
    outline["scenes"] = scenes
    log(f"Story written with {LLM_STATE['calls']} AI requests")
    return outline


# =============================================================== 2. VOICE (Google TTS)
# The narration is spoken in long passages (a whole chapter part per request) instead of one request per
# sentence: the voice keeps its natural flow and intonation, and a video needs ~8 requests instead of ~80.
# Scene timings are then found by snapping each sentence boundary to the nearest real pause in the audio.

GEMINI_TTS = {"on": bool(os.getenv("GEMINI_API_KEY")) and (os.getenv("TTS_ENGINE") or "gemini") == "gemini"}
TTS_CHUNK_BYTES = int(os.getenv("TTS_CHUNK_BYTES") or 2800)
VOICE_USED = {}  # engine that spoke the documentary, so Short hook/loop lines use the same voice
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


MARKUP = {"on": True}  # Chirp 3 HD pause tags; switched off for the run if the API refuses them


def with_pauses(text):
    """Short pause after every sentence (the slower, more cinematic read picked in the voice test)."""
    t = text.replace("[", "(").replace("]", ")")
    return re.sub(r"([.!?…])\s+(?=\S)", r"\1 [pause short] ", t)


EDGE_VOICE = os.getenv("EDGE_VOICE") or ("en-US-AndrewMultilingualNeural" if LANG == "en" else "ur-PK-AsadNeural")


def edge_tts_voice(text, path):
    """Free Microsoft neural voice (edge-tts, no key, no quota) - the backup when Google TTS fails or its
    free trial / billing stops, so the video is always made."""
    import asyncio
    import edge_tts
    rate = float(os.getenv("TTS_RATE") or 1.0)
    pct = f"{round((rate - 1) * 100):+d}%"
    last = None
    for attempt in range(4):
        try:
            asyncio.run(edge_tts.Communicate(text, EDGE_VOICE, rate=pct, pitch="-2Hz").save(str(path)))
            if Path(path).exists() and Path(path).stat().st_size > 1000:
                return
            last = "empty audio"
        except Exception as e:
            last = str(e)[:200]
        time.sleep(4 * (attempt + 1))
    raise RuntimeError(f"edge-tts failed: {last}")


def google_tts(text, path):
    key = os.environ["GOOGLE_TTS_KEY"]
    rate = float(os.getenv("TTS_RATE") or 1.0)  # Enceladus, picked from the voice samples, at normal pace
    last = None
    for voice in list(TTS_VOICES):
        for attempt in range(5):
            use_markup = MARKUP["on"] and "Chirp3" in voice
            body = {"input": {"markup": with_pauses(text)} if use_markup else {"text": text},
                    "voice": {"languageCode": "-".join(voice.split("-")[:2]), "name": voice},
                    "audioConfig": {"audioEncoding": "MP3", "speakingRate": rate, "sampleRateHertz": 24000}}
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
                if e.code == 400 and use_markup:
                    MARKUP["on"] = False
                    log("TTS pause tags not accepted - using plain text")
                    continue
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
    engines = ["offline"] if offline else (["gemini"] if GEMINI_TTS["on"] else []) + (
        ["google"] if os.getenv("GOOGLE_TTS_KEY") else []) + ["edge"]
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
                elif engine == "edge":
                    edge_tts_voice(text, raw)
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
            VOICE_USED["engine"] = engine
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
    url = ("https://api.pexels.com/videos/search?orientation=landscape&size=medium&per_page=40&query="
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
        desc = " ".join([re.sub(r"[-\d]+", " ", slug)] + [str(t) for t in v.get("tags") or []])
        if v.get("duration", 0) < 4 or not clip_ok(desc, query):
            continue
        files = [f for f in v.get("video_files", []) if f.get("width") and 1280 <= f["width"] <= 2560
                 and f.get("file_type") == "video/mp4"]
        if files:
            f = sorted(files, key=lambda f: f["width"])[0]
            out.append((f"px{v['id']}", f["link"], f"Video by {v.get('user', {}).get('name', 'Pexels')} on Pexels",
                        desc))
    return out


def pixabay_search(query):
    key = os.getenv("PIXABAY_API_KEY")
    if not key or "pixabay" in VID_OFF:
        return []
    url = (f"https://pixabay.com/api/videos/?key={key}&safesearch=true&video_type=film&per_page=40&q="
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
        desc = v.get("tags") or ""
        if v.get("type", "film") != "film" or v.get("duration", 0) < 4 or not clip_ok(desc, query):
            continue
        files = v.get("videos", {})
        for size in ("medium", "large", "small"):
            f = files.get(size) or {}
            if f.get("url") and 1200 <= (f.get("width") or 0) <= 2600:
                out.append((f"pb{v['id']}", f["url"], f"Video by {v.get('user', 'Pixabay')} on Pixabay", desc))
                break
    return out


def stock_video(queries, used, alt=0):
    """Best-matching unused clip: for each query (most specific first) the clip whose name/tags share the
    most words with it. A clip must share at least 2 of the query's words (or its only word) - an archive
    photo or a re-used matching clip is better than footage that only shares one loose word with the line."""
    sources = [("px", pexels_search), ("pb", pixabay_search)]
    if alt % 2:
        sources.reverse()
    for q in [q for q in queries if q]:
        need = min(2, relevance(q, q))  # the query's own meaningful words, capped at 2
        best, score = None, need - 1
        for tag, fn in sources:
            if (tag, q) not in VID_CACHE:
                VID_CACHE[(tag, q)] = fn(q)
            for c in VID_CACHE[(tag, q)]:
                s = relevance(q, c[3])
                if c[0] not in used and s > score:
                    best, score = c, s
        if best and score >= max(1, need):
            return best
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
        about = p.get("title", "") + " " + re.sub(r"<[^>]+>", " ", meta.get("ImageDescription", {}).get("value", ""))
        credit_txt = re.sub(r"<[^>]+>", " ", meta.get("Artist", {}).get("value", "") + " "
                            + meta.get("Credit", {}).get("value", ""))
        if relevance(query, about) < 1 or BAD_CLIP.search(p.get("title", "")) or BURNED_IN.search(about + credit_txt):
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
SYNTHETIC = {"on": True}  # safe default until gather_visuals has seen what is really in the video


def ai_image(prompt, path, seed):
    # AI pictures are OFF by default (owner's choice: only real footage/photos, so no synthetic-content
    # label is needed). Repository variable AI_PICTURES = on brings them back as a last resort.
    if (os.getenv("AI_PICTURES") or "off").strip().lower() != "on":
        return False
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
                            # long scene: a second matching shot halfway, so the picture changes every ~4-6 s
                            two = durs[i] > 7.5 and stock_video([sc.get("query2"), q], used_vid, i + 1)
                            if two:
                                p2 = work / f"v_{i:03d}b.mp4"
                                download(two[1], p2)
                                used_vid.add(two[0])
                                credits.append(two[2])
                                sc["asset2"] = p2
                    elif kind == "image" and sc.get("prompt"):
                        p = work / f"i_{i:03d}.jpg"
                        if ai_image(sc["prompt"], p, 1000 + i):
                            sc["asset"], sc["kind"], sc["ai_made"] = p, "image", True
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
        f"{stats['video']}, archive photos {stats['archive']}, other pictures {stats['image']} "
        f"(AI-made {sum(1 for s in scenes if s.get('ai_made'))})")
    if not offline and len(real) < max(5, 0.5 * len(scenes)):
        raise RuntimeError(f"Only {len(real)} real visuals found - refusing to make a low-quality video. "
                           "Add PEXELS_API_KEY / PIXABAY_API_KEY secrets.")
    found = [i for i, s in enumerate(scenes) if s.get("asset")]
    for i, sc in enumerate(scenes):  # fill gaps with the nearest matching footage, preferring the same chapter
        if not sc.get("asset"):
            j = min(found, key=lambda j: (scenes[j]["chapter"] != sc["chapter"], abs(j - i), j > i))
            sc["asset"], sc["kind"] = scenes[j]["asset"], scenes[j]["kind"]
            sc["ai_made"] = scenes[j].get("ai_made", False)
    # YouTube's "altered or synthetic content" label is needed only for realistic AI scenes (an AI voice
    # narrating real footage does not need it): set it only if an AI picture really is in the video
    n_ai = sum(1 for s in scenes if s.get("ai_made"))
    SYNTHETIC["on"] = n_ai > 0
    log(f"AI-made realistic pictures in the video: {n_ai} scene(s) -> synthetic-content label "
        f"{'ON' if SYNTHETIC['on'] else 'off'}")
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
                # the attribution incompetech.com asks for: title, artist, source, licence name + link
                MUSIC_CREDIT.append(f'Music: "{name}" Kevin MacLeod (incompetech.com)\n'
                                    "Licensed under Creative Commons: By Attribution 4.0 License\n"
                                    "http://creativecommons.org/licenses/by/4.0/")
                return p
        except Exception as e:
            log(f"Music download failed '{name}': {e}")
    return None


def pick_music(total, work, seed):
    tracks = sorted(MUSIC_DIR.glob("*.mp3")) + sorted(MUSIC_DIR.glob("*.wav")) if MUSIC_DIR.exists() else []
    out = work / "music.wav"
    t = random.Random(seed).choice(tracks) if tracks else fetch_music(work, seed)
    if t and tracks and t.with_suffix(".txt").exists():  # own track: credit text kept next to it
        MUSIC_CREDIT.append(t.with_suffix(".txt").read_text(encoding="utf-8").strip())
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


def caption_font():
    """Anton (bold condensed, free OFL font installed by the workflow) if present, else DejaVu Sans Bold."""
    if LANG != "en":
        return URDU_FONT_NAME, 1.0
    try:
        if "Anton" in subprocess.run(["fc-list", ":family"], capture_output=True, text=True).stdout:
            return "Anton", 1.0
    except Exception:
        pass
    return "DejaVu Sans", 0.8


KEY_WORDS = {"dead", "death", "died", "die", "killed", "kill", "murder", "murdered", "secret", "secretly", "never",
             "vanished", "missing", "disappeared", "poison", "poisoned", "truth", "nobody", "no", "curse", "cursed",
             "blood", "fire", "burned", "body", "bodies", "mystery", "hidden", "lie", "lied", "cover", "crash",
             "crashed", "exploded", "explosion", "drowned", "buried", "alive", "ghost", "terrifying", "shocking",
             "impossible", "forgotten", "first", "last", "only", "everyone", "gone", "screams", "silence"}
PLAIN_CAPS = {"I", "The", "A", "An", "But", "And", "So", "Then", "In", "On", "At", "It", "He", "She", "They",
              "We", "You", "This", "That", "What", "Why", "How", "When", "Where", "Who", "Nobody", "No", "Yes"}


def key_word(word, first):
    w = re.sub(r"[^\w']", "", word)
    if not w:
        return False
    if any(ch.isdigit() for ch in w) or w.lower() in KEY_WORDS:
        return True
    return w[:1].isupper() and not first and w not in PLAIN_CAPS  # names and places


def caption_text(words):
    """One caption chunk: pops in from 70% size, key words in yellow, ALL CAPS for English."""
    out = []
    for word, first in words:
        shown = word.replace("{", "(").replace("}", ")").replace("\\", "")
        shown = shown.upper() if LANG == "en" else shown
        out.append("{\\c&H00FFFF&}" + shown + "{\\c&HFFFFFF&}" if key_word(word, first) else shown)
    return "{\\fad(60,0)\\fscx70\\fscy70\\t(0,120,\\fscx100\\fscy100)}" + " ".join(out)


def caption_file(items, path, w, h, size):
    """items: (start, end, text, style). 'Cap' = big centred word-group captions, 'Top' = small headline."""
    fname, k = caption_font()
    size = int(size * k)
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{fname},{size},&H00FFFFFF,&H0000FFFF,&H00000000,&H96000000,-1,0,0,0,100,100,1,0,1,{max(4, size // 14)},{max(2, size // 30)},5,{w // 12},{w // 12},0,1
Style: Top,{fname},{int(size * 0.6)},&H0000D7FF,&H00FFFFFF,&H00000000,&H90000000,-1,0,0,0,100,100,1,0,1,{max(3, size // 18)},2,8,60,60,{int(h * 0.08)},1
"""
    lines = [f"Dialogue: 0,{ass_time(s)},{ass_time(e)},{st},,0,0,0,,{t}" for s, e, t, st in items if e - s > 0.05]
    path.write_text(head + "\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
                    + "\n".join(lines) + "\n", encoding="utf-8")


def word_groups(sentence, most=4):
    """Splits a line into groups of 2-4 words that never run across a comma or a full stop."""
    words = sentence.split()
    clauses, cur = [], []
    for n, wd in enumerate(words):
        cur.append((wd, n == 0 or bool(re.search(r"[.!?]$", words[n - 1]))))
        if re.search(r"[.,!?;:…]$", wd):
            clauses.append(cur)
            cur = []
    if cur:
        clauses.append(cur)
    merged = []  # a one-word clause joins its neighbour
    for c in clauses:
        if merged and (len(c) == 1 or len(merged[-1]) == 1):
            merged[-1] += c
        else:
            merged.append(c)
    groups = []
    for c in merged:  # even split: 5 words -> 3+2, 9 -> 3+3+3
        k = -(-len(c) // most)
        size, extra = divmod(len(c), k)
        at = 0
        for j in range(k):
            n = size + (j < extra)
            groups.append(c[at:at + n])
            at += n
    return groups


def line_captions(sentence, start, dur, shift=0.0):
    """Timed caption items for one spoken line (time shared by character count)."""
    groups = word_groups(sentence)
    speak = max(0.5, dur - 0.3)
    tot = sum(len(" ".join(w for w, _ in g)) + 2 for g in groups) or 1
    t, items = start, []
    for g in groups:
        d = speak * (len(" ".join(w for w, _ in g)) + 2) / tot
        items.append((t - shift, t + d - shift, caption_text(g), "Cap"))
        t += d
    return items


WORD_TIMES = {}  # scene index -> [(start, end)] for every word of its text, heard in the real narration


def norm_word(w):
    return re.sub(r"[^a-z0-9]", "", w.lower())


def align_words(scenes, voice_wav):
    """Finds when every word is actually spoken (free offline speech recognition, faster-whisper), so the
    captions follow the voice instead of an estimate. Any problem -> captions fall back to the estimate."""
    WORD_TIMES.clear()
    if LANG != "en" or (os.getenv("CAPTION_SYNC") or "on") == "off":
        return
    try:
        import difflib
        from faster_whisper import WhisperModel
        import numpy as np
        model = WhisperModel(os.getenv("WHISPER_MODEL") or "base.en", device="cpu", compute_type="int8")
        # decode with ffmpeg ourselves (16 kHz mono float) - faster-whisper's own PyAV decoder breaks on new PyAV
        pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", str(voice_wav), "-f", "f32le", "-ac", "1", "-ar", "16000",
                              "-"], capture_output=True, check=True).stdout
        segs, _ = model.transcribe(np.frombuffer(pcm, np.float32), language="en", word_timestamps=True,
                                   beam_size=1, condition_on_previous_text=False)
        heard = [(norm_word(w.word), w.start, w.end) for s in segs for w in (s.words or []) if norm_word(w.word)]
    except Exception as e:
        log(f"Caption sync off ({str(e)[:120]}) - using estimated timing")
        return
    ours = [(i, norm_word(wd)) for i, sc in enumerate(scenes) for wd in sc["ur"].split()]
    times = [None] * len(ours)
    sm = difflib.SequenceMatcher(None, [w for _, w in ours], [h[0] for h in heard], autojunk=False)
    for a, b, n in sm.get_matching_blocks():
        for j in range(n):
            times[a + j] = (heard[b + j][1], heard[b + j][2])
    hit = sum(t is not None for t in times)
    if hit < 0.5 * len(ours):
        log(f"Caption sync: only {hit}/{len(ours)} words recognised - using estimated timing")
        return
    known = [k for k, t in enumerate(times) if t]
    for k, t in enumerate(times):  # words not recognised (numbers, names): spread between their neighbours
        if t:
            continue
        p = max((j for j in known if j < k), default=None)
        q = min((j for j in known if j > k), default=None)
        first = p + 1 if p is not None else 0
        gap = (q if q is not None else len(times)) - first  # unknown words in this run
        a = times[p][1] if p is not None else max(0.0, times[q][0] - 0.4 * gap)
        b = times[q][0] if q is not None else a + 0.4 * gap
        pos = k - first
        step = max(0.0, b - a) / max(1, gap)
        times[k] = (a + pos * step, a + (pos + 1) * step)
    for (i, _), t in zip(ours, times):
        WORD_TIMES.setdefault(i, []).append(t)
    log(f"Caption sync: {hit}/{len(ours)} words matched to the voice")


def timed_captions(sentence, times, lo, hi):
    """Caption groups that appear exactly when their first word is spoken."""
    groups, out, k = word_groups(sentence), [], 0
    firsts = []
    for g in groups:
        firsts.append(times[k][0])
        k += len(g)
    k = 0
    for gi, g in enumerate(groups):
        s = min(max(lo, times[k][0]), hi)
        e = firsts[gi + 1] if gi + 1 < len(groups) else times[k + len(g) - 1][1] + 0.25
        e = min(hi, max(e, s + 0.3))
        out.append((s, e, caption_text(g), "Cap"))
        k += len(g)
    return out


def chunk_captions(scenes, starts, durs, t0=0.0, t1=1e9):
    items = []
    for n, (sc, st, du) in enumerate(zip(scenes, starts, durs)):
        if st + du <= t0 or st >= t1:
            continue
        timed = WORD_TIMES.get(n)
        lines = (timed_captions(sc["ur"], timed, st, st + du) if timed and len(timed) == len(sc["ur"].split())
                 else line_captions(sc["ur"], st, du))
        for s, e, t, style in lines:
            if s >= t0 and s < t1:
                items.append((s - t0, min(e, t1) - t0, t, style))
    return items


def render_long(story, work, offline):
    scenes = story["scenes"]
    log(f"Voice for {len(scenes)} scenes...")
    durs = narrate(scenes, work, offline)
    total = sum(durs)
    log(f"Narration: {total / 60:.1f} min")
    if not offline:
        align_words(scenes, work / "voice.wav")

    log("Visuals...")
    credits = gather_visuals(scenes, work, offline, durs)

    log("Scene clips...")
    clips = []
    for i, (sc, du) in enumerate(zip(scenes, durs)):
        clip = work / f"c_{i:03d}.mp4"
        if sc["kind"] == "video":
            if sc.get("asset2"):  # two shots in one scene: cut halfway (frame-exact so the timing stays)
                half = round(du / 2 * FPS) / FPS
                video_clip(sc["asset"], half, clip, idx=i)
                clips.append(clip)
                clip = work / f"c_{i:03d}b.mp4"
                video_clip(sc["asset2"], du - half, clip, idx=i + 1)
            else:
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
    ass = work / "long.ass"
    caption_file(chunk_captions(scenes, starts, durs), ass, W, H, 118)
    subs = str(ass).replace("\\", "/").replace(":", "\\:")

    log("Final long render...")
    final, clean = work / "long.mp4", work / "long_clean.mp4"  # clean copy (no captions) is cut into Shorts
    fc = (f"[0:v]setpts=PTS-STARTPTS[base];[1:v]setpts=PTS-STARTPTS[d];"
          f"[base][d]blend=c0_expr='min(255,A+0.4*B)':c1_expr='A':c2_expr='A',"
          f"vignette=PI/5,noise=alls=7:allf=t+u,fade=t=in:d=1,fade=t=out:st={max(0, total - 1.2):.2f}:d=1.2,"
          f"split=2[v][vc0];[vc0]subtitles='{subs}'[vc];"
          "[2:a]highpass=f=70,acompressor=threshold=-18dB:ratio=3:attack=5:release=150,"
          "loudnorm=I=-15:TP=-1.5:LRA=9,aresample=44100,asplit=3[vo][sc][sc2];"
          "[3:a]volume=0.8[mu];[mu][sc]sidechaincompress=threshold=0.06:ratio=3:attack=40:release=700[duck];"
          "[4:a]volume=1.0[fx];[fx][sc2]sidechaincompress=threshold=0.08:ratio=2:attack=40:release=500[fxd];"
          "[vo][duck][fxd]amix=inputs=3:duration=longest:normalize=0,alimiter=limit=0.95,asplit=2[a][a2]")
    enc = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-r", FPS, "-t", f"{total:.2f}", "-movflags", "+faststart"]
    run(["ffmpeg", "-y", "-i", work / "visual.mp4", "-stream_loop", "-1", "-i", dust, "-i", work / "voice.wav",
         "-i", music, "-i", sfx, "-filter_complex", fc,
         "-map", "[vc]", "-map", "[a2]", *enc, final,
         "-map", "[v]", "-map", "[a]", *enc, clean])
    return final, clean, credits, starts, durs, total


def voice_line(line, wav, work):
    """One extra spoken line for a Short (hook / loop) in the same voice as the video."""
    raw = wav.with_suffix(".mp3")
    engine = VOICE_USED.get("engine", "google")
    if engine == "offline":
        run(["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=f=180:r=24000", "-af", "volume=0.05",
             "-t", f"{max(1.5, len(line) / 14):.2f}", raw])
    elif engine == "gemini":
        gemini_tts(speakable(line), raw)
    elif engine == "edge":
        edge_tts_voice(speakable(line), raw)
    else:
        google_tts(speakable(line), raw)
    # no silence before the first word: in a Short the voice starts on the very first frame
    run(["ffmpeg", "-y", "-i", raw, "-af", "silenceremove=start_periods=1:start_threshold=-45dB,"
         "highpass=f=70,acompressor=threshold=-18dB:ratio=3:attack=5:release=150,"
         "loudnorm=I=-15:TP=-1.5,aresample=44100,apad=pad_dur=0.3", "-ac", "2", wav])
    return duration(wav)


def short_piece(sc, voice, dur, work, out, idx, hit=False):
    """A 9:16 piece of a Short with its own voice line over a scene's footage, music bed and optional hit."""
    v = work / f"{out.stem}_v.mp4"
    if sc["kind"] == "video":
        video_clip(sc["asset"], dur, v, 1080, 1920, idx=idx + 3)
    else:
        fr = work / f"{out.stem}.jpg"
        still_frame(sc["asset"], fr, 1080, 1920)
        animate_still(fr, dur, idx, v, 1080, 1920)
    music = work / "music.wav"
    inputs = ["-i", v, "-i", voice, "-ss", f"{(idx * 13.7) % 60:.1f}", "-stream_loop", "-1", "-i", music]
    fc = (f"[1:a]apad,atrim=0:{dur:.3f}[vo];[2:a]atrim=0:{dur:.3f},volume=0.35,"
          f"afade=t=in:d=0.3[mu];")
    mix = "[vo][mu]"
    if hit:
        inputs += ["-i", work / "hit.wav"]
        fc += "[3:a]volume=0.8[h];"
        mix += "[h]"
    fc += f"{mix}amix=inputs={mix.count('[')}:duration=first:normalize=0,alimiter=limit=0.95[a]"
    run(["ffmpeg", "-y", *inputs, "-filter_complex", fc, "-map", "0:v", "-map", "[a]", "-t", f"{dur:.3f}",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-ar", "44100", "-ac", "2", out])
    return out


# =============================================================== 4b. STANDALONE SHORTS
# Every Short is its own complete true story - NOT cut from the documentary - on a topic different from the
# documentary and from the other Short, in one of 5 rotating styles. Facts come from the topic's Wikipedia
# article and are checked line by line; a topic never comes back within 6 months (documentaries and Shorts
# together); only real footage and photos (never AI pictures); 35-45 s; secret/question title; the hook
# line and a big caption on the very first frame.

SHORT_STYLES = {
    "on_this_day": "ON THIS DAY: a real dark event that happened on this calendar date ({date}) in some past year",
    "vanished": "VANISHED: real people, a ship, a plane or a whole village that disappeared",
    "strange_accident": "STRANGE ACCIDENT: a real, bizarre, little-known accident or disaster",
    "one_night": "THE NIGHT THE...: the story of one single real night when everything went wrong",
    "hidden_truth": "HIDDEN TRUTH: the dark, little-known truth behind something famous (a place, landmark, "
                    "object, invention, food or tradition)",
}
SHORTS_LOG_FIELDS = ["date", "style", "topic", "wikipedia", "title", "video"]


def norm_topic(t):
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def shorts_log():
    f = DATA / "shorts_log.csv"
    if not f.exists():
        return []
    import csv
    with open(f, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def short_used_topics(extra=()):
    """Topics of every documentary and Short of the last 6 months (+ extra): (normalised set, names)."""
    since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=REPEAT_DAYS)
    names = [t for t in list(used_topics()) + list(extra) if t]
    for r in shorts_log():
        try:
            if dt.datetime.fromisoformat(r["date"]) >= since:
                names += [t for t in (r.get("topic", ""), r.get("wikipedia", "")) if t]
        except (KeyError, ValueError):
            continue
    return {norm_topic(t) for t in names}, names


def pick_short_styles():
    """All 5 styles, the one used longest ago first: the day's 2 Shorts never share a style, and the
    styles take turns day after day."""
    last = {}
    for i, r in enumerate(shorts_log()):
        last[r.get("style")] = i
    return sorted(SHORT_STYLES, key=lambda s: last.get(s, -1))


# On This Day: only real dark EVENTS - accidents, disasters, murders, wartime massacres, disappearances -
# and the linked article must be about the event itself, never about an organisation, place or thing
DARK_EVENT = re.compile(DARK.pattern + r"|massacre|shipwreck|derail|earthquake|tsunami|hurricane|avalanche|"
                        r"blizzard|inferno|fire\b|blaze|bombing|fatal|killed|died|deaths|victims|missing", re.I)
THING = re.compile(r"\b(air forces?|warfare branch|army|navy|military branch|organi[sz]ation|company|corporation|"
                   r"agency|airline|manufacturer|political party|government|ministry|university|newspaper|"
                   r"country|city|town|village|capital|province|human settlement|politician|aircraft family|"
                   r"class of|actor|actress|singer|writer|footballer|band|film|album)\b", re.I)


# the article's first sentence says it is about an organisation / place / thing ("X is a city in ...")
LEAD_THING = re.compile(r"^[^.]{0,120}?\b(?:is|was|are|were)\s+(?:an?|the)\s+[^.]{0,50}?\b(air forces?|army|navy|"
                        r"organi[sz]ation|company|corporation|agency|airline|manufacturer|political party|"
                        r"government|ministry|university|newspaper|country|city|town|village|province|"
                        r"politician|military unit|squadron|regiment|aircraft|ship class)\b", re.I)


def otd_sensitive(txt):
    """The usual sensitive-topic rule, except that massacres in past wars may be told on their date."""
    return SENSITIVE.search(re.sub(r"massacre\w*|atrocit\w*", " ", txt or "", flags=re.I))


def otd_events(d=None, limit=20):
    """Today's date in history: dark events of every past year, each with the article about the event."""
    d = d or dt.date.today()
    out = []
    for kind in ("selected", "events"):
        try:
            evs = http(f"https://en.wikipedia.org/api/rest_v1/feed/onthisday/{kind}/{d:%m}/{d:%d}",
                       timeout=30).get(kind, [])
        except Exception as e:
            log(f"On-this-day {kind} {d} failed: {e}")
            continue
        for ev in evs:
            txt = ev.get("text", "")
            if ev.get("year", 3000) > 2000 or not DARK_EVENT.search(txt):
                continue
            # the page about the EVENT (its title/description says what happened), never a thing or place
            pg = next((p for p in ev.get("pages") or []
                       if DARK_EVENT.search(f"{p.get('title', '')} {p.get('description', '')}")
                       and not THING.search(p.get("description", ""))), None)
            if not pg:
                continue
            title = (pg.get("normalizedtitle") or pg.get("title", "")).replace("_", " ")
            if otd_sensitive(f"{txt} {title} {pg.get('description', '')}") or any(o[0] == title for o in out):
                continue
            out.append((title, f"{d:%d %B} {ev['year']}: {txt}"))
    return out[:limit]


def short_candidates(style, avoid_names):
    if style == "on_this_day":  # every past year's dark event on today's date, one after another
        return [{"name": t, "wikipedia": t, "why": d} for t, d in otd_events()]
    try:
        got = ask_json(f"""You find topics for 40-second YouTube Shorts on a dark-history channel.
Style: {SHORT_STYLES[style]}.
List 10 REAL, well-documented events of this kind, each with its own English Wikipedia article. Mix famous
and little-known ones, all before the year 2000. Never: living wars or conflicts, politics, race, religion,
terrorism, shootings, genocide or massacres.
Never any of these (already used): {'; '.join(avoid_names[-80:]) or 'none'}
Return ONLY JSON: {{"topics": [{{"name": "short name, 2-5 words", "wikipedia": "exact English Wikipedia article title"}}]}}""",
                       keys=("topics",), max_tokens=1500)
    except Exception as e:
        log(f"Short topics ({style}) failed: {e}")
        return []
    out = []
    for t in got.get("topics") or []:
        if isinstance(t, dict) and text(t.get("name")) and text(t.get("wikipedia")):
            out.append({"name": text(t["name"]), "wikipedia": text(t["wikipedia"]), "why": ""})
    return out


def pick_short_topic(style, used, names):
    """The first candidate that is new (6-month rule), not sensitive and has a real Wikipedia article
    (for On This Day: an article about the dark EVENT itself, not about an organisation, place or thing)."""
    otd = style == "on_this_day"
    sensitive = otd_sensitive if otd else SENSITIVE.search
    for c in short_candidates(style, names):
        if {norm_topic(c["name"]), norm_topic(c["wikipedia"])} & used:
            continue
        if sensitive(f"{c['name']} {c['wikipedia']} {c.get('why', '')}"):
            continue
        full = wiki_extract(c["wikipedia"], chars=60000)
        wt = LAST_WIKI["title"]
        if len(full) < 1200 or norm_topic(wt) in used or sensitive(wt):
            continue
        if otd and not (DARK_EVENT.search(full[:600]) and not LEAD_THING.search(full[:300])):
            log(f"On this day: '{wt}' is not an article about a dark event - skipped")
            continue
        return {**c, "wikipedia": wt, "facts": fact_digest(full, 7000), "full": full}
    return None


def write_short_script(pick, style, other_titles=()):
    """One complete Short: 7-9 spoken lines (95-110 words), hook first, every line backed by the facts."""
    facts, note = pick["facts"], ""
    otd = ""
    if style == "on_this_day":
        otd = ('\n- The hook says it happened on this day, e.g. "On this day in 1912, ..." (the year must be in '
               'the facts).')
    for attempt in range(3):
        o = ask_json(f"""You write a complete 40-second YouTube Short for the dark-history channel {CHANNEL}.
STYLE: {SHORT_STYLES[style].format(date=dt.date.today().strftime('%B %d'))}
STORY: {pick['name']}{(' - ' + pick['why']) if pick.get('why') else ''}
It is ONE complete true story on its own - what happened, the twist, the ending. Someone who has never seen
anything else from us understands all of it. Use ONLY facts from the SOURCE FACTS: no invented names,
numbers, dates, quotes or scenes; a theory or legend must be called a theory or legend.
- 7 to 9 short spoken lines, 95 to 110 words in total (35-45 seconds), simple gripping English.
- Line 1 = the hook, heard on the very first frame: a jolt or a question, max 14 words, says briefly
  who / where / when. No greeting, no "in this video", no slow start. Its "query" is the most dramatic
  shot of the story.{otd}
- The last line flows straight back into the hook when the Short replays.
- Title: a secret or a question like our best Shorts "The Night the Mob Came", "What Killed United Air
  Lines", "What Vanished Over the Atlantic". Max 50 characters. It must NOT start with the famous name.
  Different from: {'; '.join(other_titles) or 'none'}
- For every line a real-footage search: "query" (2-4 English words for stock footage that SHOWS what the
  line says, in the right era - never modern things), "query2" (2-3 more general words), "film" (2-4 words
  to find real old film of it on Wikimedia Commons, or ""), "archive" (2-5 words for a real photo of a
  named person, ship, place or object, or "").
{note}
SOURCE FACTS (Wikipedia: {pick['wikipedia']}):
{facts}

Return ONLY JSON: {{"title": "...", "key_quote": "8-25 words copied EXACTLY from the SOURCE FACTS that prove the story's main claim",
"description": "2-3 true sentences for the YouTube description", "hashtag": "one specific hashtag, e.g. #MaryCeleste",
"lines": [{{"text": "...", "query": "...", "query2": "...", "film": "...", "archive": "..."}}]}}""",
                     keys=("lines", "title"), max_tokens=2500)
        lines = [l for l in o.get("lines") or [] if isinstance(l, dict) and text(l.get("text"))]
        words = sum(len(text(l["text"]).split()) for l in lines)
        if not quote_found(text(o.get("key_quote")), pick.get("full") or facts):
            note = "Your last script's main claim was not backed by the SOURCE FACTS. Use only what they say."
            log(f"Short script ({pick['name']}): main claim not found in the facts - writing again")
            continue
        if not (80 <= words <= 125 and len(lines) >= 5):
            note = f"Your last script had {words} words in {len(lines)} lines - it must be 95-110 words in 7-9 lines."
            log(f"Short script ({pick['name']}): {words} words - writing again")
            continue
        break
    else:
        raise RuntimeError(f"no usable Short script for {pick['name']}")
    scenes = [{"ur": text(l["text"]), "type": "video", "query": text(l.get("query")) or "dark night storm",
               "query2": text(l.get("query2")) or "dark fog", "film": text(l.get("film")),
               "archive": text(l.get("archive")), "prompt": "", "chapter": 0, "hit": i == 0}
              for i, l in enumerate(lines)]
    full = pick.get("full") or facts
    fact_check(scenes, relevant_facts(full, " ".join(s["ur"] for s in scenes), 8000))
    # the title is checked on its own: rewritten if the facts don't back it, the Short dropped if still not
    title = verify_title(short_title(text(o.get("title")), pick["name"]) or "What Really Happened That Night?",
                         full, pick["name"], short=True)
    tag = "#" + re.sub(r"[^A-Za-z0-9]", "", text(o.get("hashtag")) or pick["name"])
    return {"title": title, "scenes": scenes, "description": text(o.get("description")), "hashtag": tag}


def fit_short_audio(wavs, durs, work):
    """35-45 s: a slightly faster read (max 12%) when too long, a little more air between lines when
    too short; far outside the range -> the script is written again."""
    total = sum(durs)
    if total > SHORT_MAX:
        tempo = total / SHORT_MAX
        if tempo > 1.12:
            raise RuntimeError(f"Short too long ({total:.1f}s)")
        for i, w in enumerate(wavs):
            fast = w.with_name(w.stem + "_f.wav")
            run(["ffmpeg", "-y", "-i", w, "-af", f"atempo={tempo:.4f}", fast])
            wavs[i], durs[i] = fast, duration(fast)
    elif total < SHORT_MIN:
        add = (SHORT_MIN - total) / len(wavs)
        if add > 0.8:
            raise RuntimeError(f"Short too short ({total:.1f}s)")
        for i, w in enumerate(wavs):
            pad = w.with_name(w.stem + "_p.wav")
            run(["ffmpeg", "-y", "-i", w, "-af", f"apad=pad_dur={add + 0.05:.3f}", pad])
            wavs[i], durs[i] = pad, duration(pad)
    return wavs, durs


def concat_short(parts, caps, work, tag):
    """Joins the 9:16 pieces and burns in the big centred captions."""
    ass = work / f"{tag}.ass"
    caption_file(caps, ass, 1080, 1920, 150)
    subs = str(ass).replace("\\", "/").replace(":", "\\:")
    ins, fc = [], ""
    for n, p in enumerate(parts):
        ins += ["-i", p]
        fc += (f"[{n}:v]scale=1080:1920,setsar=1,fps={FPS},format=yuv420p[v{n}];"
               f"[{n}:a]aresample=44100,aformat=channel_layouts=stereo[a{n}];")
    fc += "".join(f"[v{n}][a{n}]" for n in range(len(parts)))
    fc += f"concat=n={len(parts)}:v=1:a=1[cv][a];[cv]subtitles='{subs}',format=yuv420p[v]"
    out = work / f"{tag}.mp4"
    run(["ffmpeg", "-y", *ins, "-filter_complex", fc, "-map", "[v]", "-map", "[a]",
         "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-c:a", "aac", "-b:a", "160k",
         "-movflags", "+faststart", out])
    return out


def render_story_short(script, work, k, offline):
    """Every line in the documentary's voice over its own real clip; the hook (with a hit) first."""
    tag = f"short{k}"
    sdir = work / tag
    sdir.mkdir(exist_ok=True)
    scenes = script["scenes"]
    wavs = [sdir / f"l{i:02d}.wav" for i in range(len(scenes))]
    durs = [voice_line(sc["ur"], w, work) for sc, w in zip(scenes, wavs)]
    wavs, durs = fit_short_audio(wavs, durs, work)
    saved, old = SYNTHETIC["on"], os.environ.get("AI_PICTURES")
    os.environ["AI_PICTURES"] = "off"  # Shorts: real footage and photos only, never AI pictures
    try:
        credits = gather_visuals(scenes, sdir, offline, durs)
    finally:
        SYNTHETIC["on"] = saved  # the documentary's own label stays as it was
        if old is None:
            os.environ.pop("AI_PICTURES", None)
        else:
            os.environ["AI_PICTURES"] = old
    parts, caps, t = [], [], 0.0
    for i, (sc, w, d) in enumerate(zip(scenes, wavs, durs)):
        parts.append(short_piece(sc, w, d, work, sdir / f"{tag}_p{i:02d}.mp4", i, hit=(i == 0)))
        caps += line_captions(sc["ur"], t, d)
        t += d
    out = concat_short(parts, caps, work, tag)
    log(f"Short ready: {out.name} ({t:.1f}s, {len(scenes)} lines, '{script['title']}')")
    return out, credits


def make_daily_shorts(story, work, offline):
    """The day's 2 Shorts: each its own story, style and topic (never the documentary's)."""
    if offline:
        return []
    used, names = short_used_topics([story.get("topic", ""), story.get("wikipedia", "")])
    out = []
    for style in pick_short_styles():
        if len(out) >= SHORTS_COUNT:
            break
        try:
            pick = pick_short_topic(style, used, names)
            if not pick:
                log(f"Short ({style}): no new topic found")
                continue
            log(f"Short {len(out) + 1} ({style}): {pick['name']} (Wikipedia '{pick['wikipedia']}')")
            for attempt in range(2):
                script = write_short_script(pick, style, [s["title"] for s in out])
                try:
                    path, credits = render_story_short(script, work, len(out), offline)
                    break
                except RuntimeError as e:
                    if "Short too" not in str(e) or attempt:
                        raise
                    log(f"{e} - writing it again")
        except Exception as e:
            log(f"Short ({style}) failed: {str(e)[:300]}")
            continue
        used |= {norm_topic(pick["name"]), norm_topic(pick["wikipedia"])}
        names.append(pick["name"])
        out.append({"path": path, "credits": credits, "style": style, "topic": pick["name"],
                    "wikipedia": pick["wikipedia"], **script})
    return out


def short_description(sh, credits, doc_vid=""):
    src = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(sh['wikipedia'].replace(' ', '_'))}"
    body = (f"{sh['description']}\n\nThe true story of {sh['topic']}.\nSource: {src}").strip()
    extra = f"Today's full documentary: https://youtu.be/{doc_vid}\n\n" if doc_vid else ""
    return description({"description": body, "hashtags": [sh["hashtag"], "#darkhistory", "#shorts"]},
                       credits, extra, synthetic=False)


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


def yt_upload(tok, video, title, desc, tags, publish_at=None, thumb=None, synthetic=None):
    synthetic = SYNTHETIC["on"] if synthetic is None else synthetic
    status = {"privacyStatus": PRIVACY, "selfDeclaredMadeForKids": False, "embeddable": True,
              "containsSyntheticMedia": bool(synthetic)}  # true only if a realistic AI scene is in it
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
    CHANNEL_ID["id"] = (res.get("snippet") or {}).get("channelId") or CHANNEL_ID["id"]
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


def srt_time(t):
    ms = int(round(max(0.0, t) * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def make_srt(scenes, starts, durs, path):
    """Normal subtitle file (sentence pieces of up to 9 words, ordinary case) from the same timing as the
    burned-in captions, for YouTube's CC track: searchable text and no wrong auto-captions."""
    cues = []
    for n, (sc, st, du) in enumerate(zip(scenes, starts, durs)):
        words = sc["ur"].split()
        times = WORD_TIMES.get(n)
        times = times if times and len(times) == len(words) else None
        chunk, total = [], sum(len(w) + 1 for w in words) or 1
        for k, w in enumerate(words):
            chunk.append(k)
            left = len(words) - k - 1  # never leave 1-2 words alone on the last line
            if (k == len(words) - 1 or (left > 2 and (len(chunk) >= 9 or
                                                     (len(chunk) >= 4 and re.search(r"[.,!?;:…]$", w))))):
                if times:
                    a, b = times[chunk[0]][0], times[chunk[-1]][1]
                else:  # no word timing: share the scene by character count
                    before = sum(len(x) + 1 for x in words[:chunk[0]])
                    upto = before + sum(len(words[j]) + 1 for j in chunk)
                    a, b = st + du * before / total, st + du * upto / total
                cues.append([max(a, st), min(max(b, a + 0.6), st + du), " ".join(words[j] for j in chunk)])
                chunk = []
    for c, nxt in zip(cues, cues[1:]):  # never overlap the next line
        c[1] = min(c[1], nxt[0])
    path.write_text("".join(f"{i}\n{srt_time(a)} --> {srt_time(b)}\n{t}\n\n" for i, (a, b, t) in
                            enumerate(cues, 1) if b > a), encoding="utf-8")
    return path


def yt_captions(tok, vid, srt):
    """Adds our subtitles as the video's English CC track (needs the youtube.force-ssl scope on the
    refresh token; 400 quota units)."""
    meta = json.dumps({"snippet": {"videoId": vid, "language": "en", "name": "English", "isDraft": False}})
    bnd = "dhcaptions" + str(random.randint(10 ** 8, 10 ** 9))
    body = (f"--{bnd}\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n{meta}\r\n"
            f"--{bnd}\r\nContent-Type: application/x-subrip\r\n\r\n").encode() + Path(srt).read_bytes() + \
        f"\r\n--{bnd}--\r\n".encode()
    try:
        http("https://www.googleapis.com/upload/youtube/v3/captions?uploadType=multipart&part=snippet", body,
             headers={"Authorization": f"Bearer {tok}", "Content-Type": f"multipart/related; boundary={bnd}"})
        log("Subtitles (CC) uploaded")
    except urllib.error.HTTPError as e:
        msg = e.read()[:200]
        hint = (" - the YouTube refresh token needs the youtube.force-ssl scope for this"
                if e.code in (401, 403) else "")
        log(f"Subtitles not uploaded: HTTP {e.code} {msg!r}{hint}")
    except Exception as e:
        log(f"Subtitles not uploaded: {e}")


CHANNEL_ID = {"id": ""}
SCOPE_HINT = " - the YouTube refresh token needs the youtube.force-ssl scope for this"


def yt_api(tok, path, body, part):
    return http(f"https://www.googleapis.com/youtube/v3/{path}?part={part}", body,
                headers={"Authorization": f"Bearer {tok}"}, timeout=60)


def series_playlist(tok, series):
    """The series' playlist id: from data/series_playlists.csv, or a new public playlist (50 quota units)."""
    f = DATA / "series_playlists.csv"
    known = {}
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            if "," in line:
                k, v = line.rsplit(",", 1)
                known[k.strip()] = v.strip()
    if known.get(series):
        return known[series]
    info = series_list().get(series, "")
    r = yt_api(tok, "playlists", {"snippet": {"title": f"{series} | {CHANNEL}"[:150],
                                              "description": f"{CHANNEL}: {info}. True stories, checked against real sources.",
                                              "defaultLanguage": "en"},
                                  "status": {"privacyStatus": "public"}}, "snippet,status")
    with open(f, "a", encoding="utf-8") as fh:
        fh.write(f"{series},{r['id']}\n")
    log(f"Created playlist '{series}' ({r['id']})")
    return r["id"]


def add_to_series(tok, vid, series):
    """Puts the documentary into its series playlist (needs youtube.force-ssl; 50 + 50 quota units)."""
    if not series:
        return
    try:
        pid = series_playlist(tok, series)
        body = {"snippet": {"playlistId": pid, "resourceId": {"kind": "youtube#video", "videoId": vid}}}
        # a brand-new playlist / just-uploaded video often answers 409 SERVICE_UNAVAILABLE for a few seconds
        for wait in (10, 30, 60, 0):
            try:
                yt_api(tok, "playlistItems", body, "snippet")
                break
            except urllib.error.HTTPError as e:
                if e.code not in (409, 500, 503) or not wait:
                    raise
                log(f"Playlist busy (HTTP {e.code}) - trying again in {wait}s")
                time.sleep(wait)
        log(f"Added to playlist '{series}'")
    except urllib.error.HTTPError as e:
        log(f"Playlist not updated: HTTP {e.code} {e.read()[:200]!r}{SCOPE_HINT if e.code in (401, 403) else ''}")
    except Exception as e:
        log(f"Playlist not updated: {e}")


def post_question(tok, vid, question):
    """Posts the comment question under the video as the channel (needs youtube.force-ssl; 50 units).
    YouTube's API cannot pin comments - pin it in YouTube Studio with one click."""
    if not (question and CHANNEL_ID["id"]):
        return
    try:
        yt_api(tok, "commentThreads", {"snippet": {"channelId": CHANNEL_ID["id"], "videoId": vid,
                                                   "topLevelComment": {"snippet": {"textOriginal": question}}}},
               "snippet")
        log(f"Comment posted: {question}")
    except urllib.error.HTTPError as e:
        log(f"Comment not posted: HTTP {e.code} {e.read()[:200]!r}{SCOPE_HINT if e.code in (401, 403) else ''}")
    except Exception as e:
        log(f"Comment not posted: {e}")


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


def description(story, credits, extra="", synthetic=None):
    """Music credit comes right after the intro so YouTube's 5000-character limit can only ever cut the
    footage list, never the attribution the music licence requires."""
    synthetic = SYNTHETIC["on"] if synthetic is None else synthetic
    tags = " ".join(story.get("hashtags", [])[:3])
    head = (f"{story['description'].strip()}\n\n{extra}"
            f"Subscribe to {CHANNEL} for the stories history tried to forget.\n\n"
            f"Narration: {'English' if LANG == 'en' else 'Urdu'}."
            + (" Some scenes are AI-generated illustrations of historical events." if synthetic else "")
            + "\n\n"
            + "".join(c + "\n\n" for c in MUSIC_CREDIT))
    tail = "".join(c + "\n" for c in SFX_CREDITS[:15]) + f"\n{tags}"
    cred = ""
    for c in credits[:30]:
        if len(head) + len(cred) + len(tail) + len(c) + 40 > 4900:
            break
        cred += f"- {c}\n"
    return head + (f"Footage & images:\n{cred}" if cred else "") + tail


# =============================================================== MAIN

def video_day(t, start_hour):
    """The daily 'slot' a moment belongs to. GitHub often starts scheduled runs many hours late (even after
    midnight UTC), so a day's slot runs from start_hour UTC to start_hour UTC of the next day."""
    return (t - dt.timedelta(hours=start_hour)).date()


def already_done(lines, marker, start_hour, now=None):
    """True when a line with marker (e.g. ',long,') was logged in the current slot."""
    now = now or dt.datetime.now(dt.timezone.utc)
    for line in lines:
        if marker not in line:
            continue
        try:
            t = dt.datetime.fromisoformat(line.split(",", 1)[0])
        except ValueError:
            continue
        if t.tzinfo is None:
            t = t.replace(tzinfo=dt.timezone.utc)
        if video_day(t, start_hour) == video_day(now, start_hour):
            return True
    return False


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
    log_f = DATA / "upload_log.csv"
    if (os.getenv("RUN_EVENT") == "schedule" and not (a.no_upload or a.offline or a.script) and log_f.exists()
            and already_done(log_f.read_text(encoding="utf-8").splitlines(), ",long,", 13)):
        log("Today's documentary is already uploaded - this backup run has nothing to do")
        return

    story = json.loads(Path(a.script).read_text(encoding="utf-8")) if a.script else None
    if not story:
        for attempt in range(3):
            try:
                story = write_story(LONG_MINUTES)
                break
            except Exception as e:
                log(f"Story attempt {attempt + 1} failed: {e}")
        if not story:
            raise RuntimeError("Could not write a story")
    story = normalize_outline(story)
    story["scenes"] = [dict(s, chapter=int(s.get("chapter") or 0)) for s in scene_list(story)]

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M")
    work = OUT / f"{stamp}-{slugify(story['topic'])}"
    work.mkdir(parents=True, exist_ok=True)
    (work / "story.json").write_text(json.dumps(story, indent=1, ensure_ascii=False, default=str), encoding="utf-8")

    global SETTING
    SETTING = (story.get("setting") or "").strip()[:300]
    long_v, clean_v, credits, starts, durs, total = render_long(story, work, a.offline)
    srt = make_srt(story["scenes"], starts, durs, work / "captions.srt")
    thumb = thumbnail(story, work, a.offline)
    shorts = make_daily_shorts(story, work, a.offline)  # own stories, not cut from the documentary

    frames = [grab(long_v, total * f, work / f"pv{n}.jpg") for n, f in enumerate((0.05, 0.3, 0.55, 0.8))]
    frames += [grab(s["path"], 4, work / f"pvs{n}.jpg") for n, s in enumerate(shorts)] + [thumb]
    preview(frames, DATA / "previews" / "latest.jpg")
    log(f"Long video ready: {total / 60:.1f} min, {len(shorts)} shorts")

    results, short_rows = [], []
    if not (a.offline or a.no_upload):
        tok = yt_token()
        vid, st = yt_upload(tok, long_v, story["title"], description(story, credits, chapter_lines(story, starts)),
                            story.get("tags", []), thumb=thumb)
        results.append(("long", story["title"], st, vid))
        if (os.getenv("UPLOAD_CAPTIONS") or "on") != "off":
            yt_captions(tok, vid, srt)
        add_to_series(tok, vid, story.get("series"))
        post_question(tok, vid, story.get("comment_question"))
        now = dt.datetime.now(dt.timezone.utc)
        for k, sh in enumerate(shorts):
            when = (now + dt.timedelta(hours=4 + 5 * k)).replace(microsecond=0).isoformat().replace("+00:00", "Z")
            t = f"{sh['title']} #shorts"
            # the famous name lives in the description, hashtag and tags (not at the start of the title)
            try:
                svid, sst = yt_upload(tok, sh["path"], t, short_description(sh, sh["credits"], vid),
                                      [sh["topic"], "dark history", "history shorts", "true story",
                                       sh["style"].replace("_", " ")], publish_at=when, synthetic=False)
                results.append(("short", t, sst, svid))
                short_rows.append({"date": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                                   "style": sh["style"], "topic": sh["topic"], "wikipedia": sh["wikipedia"],
                                   "title": sh["title"], "video": f"https://youtu.be/{svid}"})
            except Exception as e:
                log(f"Short upload failed: {e}")

    if results:  # a --no-upload test run does not use up the topic
        import csv
        fields = ["date", "topic", "wikipedia", "kind", "title", "angle", "video", "series"]
        rows = topic_log()
        rows.append({"date": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                     "topic": story["topic"].strip(), "wikipedia": story.get("wikipedia", ""),
                     "kind": story.get("kind", "trending"), "title": story["title"],
                     "angle": text(story.get("angle"))[:200], "video": f"https://youtu.be/{results[0][3]}",
                     "series": story.get("series", "")})
        with open(DATA / "topics_log.csv", "w", encoding="utf-8", newline="") as f:  # rewritten: header may grow
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        if story.get("next"):  # the promise made at the end of this video decides the next topic
            (DATA / "next_topic.json").write_text(json.dumps(story["next"], ensure_ascii=False, default=str),
                                                  encoding="utf-8")
        with open(DATA / "upload_log.csv", "a", encoding="utf-8") as f:
            for kind, t, st, vid in results:
                f.write(f"{dt.datetime.now(dt.timezone.utc).isoformat()},{kind},\"{t}\",{st},https://youtu.be/{vid}\n")
    if short_rows:  # Shorts topics: never again within 6 months
        import csv
        sf = DATA / "shorts_log.csv"
        new = not sf.exists()
        with open(sf, "a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SHORTS_LOG_FIELDS)
            if new:
                w.writeheader()
            w.writerows(short_rows)
    import shutil
    keep = {"long.mp4", "thumb.jpg", "story.json", "captions.srt"} | {s["path"].name for s in shorts}
    for p in work.iterdir():
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.name not in keep:
            p.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FAILED: {e}")
        sys.exit(1)
