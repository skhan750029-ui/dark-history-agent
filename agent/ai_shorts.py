#!/usr/bin/env python3
"""
Everyday AI Easy - YouTube Shorts that show real, well-known AI tools in plain, calm English.

Formats
  - listicle (most days): "3 AI tools for students / small business / job seekers / creators / your phone"
  - explainer (now and then): one tool, 3 things it can do
Every tool comes from agent/ai_tools.json (real, well-known tools only). Each run opens the tool's official
page in a real browser (Playwright): the tool is used only if its name is on that page, and every feature
line in the script must be backed by words copied from that page (checked in code) - otherwise it is dropped.

Short = hook (9 rotating styles, never the same twice in a row, one of them a two-voice question/answer)
-> one card per tool ("#1/3", big logo + name, the real official website scrolling inside the card,
what it does + one example) -> "Follow for one new AI tool every day."
Light, clean design; white sentence-case captions, 2-3 words at a time; calm, friendly voice.
Every uploaded Short's hook style is logged in data/ai/hooks.csv to see which style works best.

Uploads use Dark History's OAuth client (YT_CLIENT_ID / YT_CLIENT_SECRET) with this channel's own
AI_YT_REFRESH_TOKEN; the workflow only uploads once the repository variable AI_UPLOAD = on.
Usage:  python agent/ai_shorts.py [--no-upload] [--count 2]
"""
import argparse
import asyncio
import base64
import csv
import datetime as dt
import difflib
import html
import io
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from pathlib import Path

# channel settings must be in place before studio.py reads them at import time
os.environ.setdefault("CHANNEL_NAME", "Everyday AI Easy")
os.environ["NARRATION_LANG"] = "en"
os.environ["CAPTION_SYNC"] = os.getenv("CAPTION_SYNC") or "on"
os.environ.pop("GEMINI_API_KEY", None)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import studio as base  # noqa: E402  (shared building blocks; nothing in studio.py is changed)

from PIL import Image, ImageDraw, ImageFilter, ImageFont  # noqa: E402

log, run, duration, http = base.log, base.run, base.duration, base.http
ROOT = base.ROOT
DATA = ROOT / "data" / "ai"
OUT = ROOT / "out" / "ai"
TOOLS_FILE = Path(__file__).resolve().parent / "ai_tools.json"
CHANNEL = os.environ["CHANNEL_NAME"]
PRIVACY = (os.getenv("PRIVACY") or "public").strip().lower()
W, H, FPS = 1080, 1920, 30

# voices: A = main narrator, B = second voice for the question/answer hook (picked from the voice samples)
VOICE_A = os.getenv("AI_TTS_VOICE") or "en-US-Chirp3-HD-Achird"
VOICE_B = os.getenv("AI_TTS_VOICE_B") or "en-US-Chirp3-HD-Sulafat"
if VOICE_B == VOICE_A:  # the second voice must sound different
    VOICE_B = "en-US-Chirp3-HD-Achird" if VOICE_A.endswith("Sulafat") else "en-US-Chirp3-HD-Sulafat"
EDGE_A = os.getenv("AI_EDGE_VOICE") or "en-US-BrianMultilingualNeural"
EDGE_B = os.getenv("AI_EDGE_VOICE_B") or "en-US-EmmaMultilingualNeural"
RATE = float(os.getenv("AI_TTS_RATE") or 1.0)
# YouTube asks for the "altered or synthetic content" label only for realistic AI content (a real person
# saying something they didn't, AI footage of real places/events, realistic scenes that never happened).
# A generic narration voice + real website screenshots + AI-helped script needs no label; AI_LABEL=on forces it.
AI_LABEL = (os.getenv("AI_LABEL") or "off").strip().lower() == "on"

OUTRO = "Follow for one new AI tool every day."
AUDIENCES = {"students": "students", "small_business": "small business owners", "job_seekers": "job seekers",
             "creators": "creators", "phone": "your phone"}
STYLES = {
    "question": "a direct, honest question the viewer would ask themselves (e.g. 'Still making study notes by hand?')",
    "little_known": "a little-known but REAL feature taken from the official text (e.g. 'Few people know NotebookLM "
                    "can ...' - only if that feature is in the official text)",
    "problem_solution": "name one real everyday problem of this audience, then say these tools help with it",
    "numbered_list": "a plain numbered promise (e.g. 'Three AI tools that make job hunting easier.')",
    "before_after": "a calm before/after contrast of the task (e.g. 'Before: a blank page. After: a first draft to "
                    "edit.') - no invented numbers or times",
    "warning": "a genuine, useful caution (e.g. 'Don't do this the hard way.' or 'Check AI answers before you use "
               "them - but these tools can still save you work.') - never fear or false alarms",
    "comparison": "an honest comparison (e.g. 'ChatGPT is not the only option.' or 'The old way versus the AI way.')",
    "challenge": "a friendly challenge (e.g. 'Try one of these this week and see if it helps.')",
    "qa_dialog": "two voices: speaker A asks a short question like a viewer would, speaker B answers in one "
                 "short sentence that leads into the list",
}
HYPE = re.compile(r"revolution|game.?chang|mind.?blow|insane|crazy|unbelievable|shocking|you won'?t believe|"
                  r"changes everything|world.?changing|groundbreaking|magic(al)? (wand|trick)|destroy|killer|"
                  r"the end of|terrifying|scary|replace (all|every)|jobs? (are )?gone|must.?see|secret|"
                  r"blow your mind|10x|100x|literally", re.I)
FAKE_EXPERIENCE = re.compile(r"\bI (tried|tested|used|use|love|found|switched|built|made)\b|\bI'?ve (been|tried|used|"
                             r"tested)|\bmy (favorite|favourite|results?|life|workflow)\b|changed my life|\bwe tested|"
                             r"\bI personally\b", re.I)
NUM_WORDS = ["one", "two", "three", "four", "five"]
MUSIC = ["Wallpaper", "Carefree", "Bossa Antigua", "Local Forecast - Elevator", "Easy Lemon"]
UA_MOBILE = ("Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) "
             "Version/17.5 Mobile/15E148 Safari/604.1")
BLOCKED = re.compile(r"just a moment|verify you are human|are you a robot|access denied|attention required|"
                     r"captcha|enable javascript and cookies|request blocked|403 forbidden|page not found|404",
                     re.I)

# light design
BG_TOP, BG_BOTTOM = (248, 250, 252), (236, 241, 255)
INK, MUTED, ACCENT = (17, 24, 39), (100, 116, 139), (79, 70, 229)
CARD_X, CARD_Y, CARD_W, CARD_H = 90, 390, 900, 960
CAP_Y = 1455            # captions sit under the card, above the Shorts buttons/title area


# =============================================================== small helpers

def clean(s):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s or ""))).strip()


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", (s or "").lower())).strip()


def load_tools():
    return json.loads(TOOLS_FILE.read_text(encoding="utf-8"))["tools"]


LOG_FIELDS = ["date", "video", "privacy", "format", "audience", "hook_style", "title", "tools"]


def history():
    f = DATA / "hooks.csv"
    if not f.exists():
        return []
    with open(f, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def pick_least_recent(options, recent, rng, banned=()):
    """Option used least in `recent` (and longest ago); never one in `banned`."""
    opts = [o for o in options if o not in banned] or list(options)
    rng.shuffle(opts)
    last_at = {o: max((i for i, r in enumerate(recent) if r == o), default=-1) for o in opts}
    return min(opts, key=lambda o: (recent.count(o), last_at[o]))


def pick_style(recent_styles, rng):
    """Never the same style twice in a row; the two-voice style at most once in 6 Shorts."""
    banned = set(recent_styles[-1:])
    if "qa_dialog" in recent_styles[-5:]:
        banned.add("qa_dialog")
    return pick_least_recent(list(STYLES), recent_styles[-9:], rng, banned)


def recent_tools(hist, days):
    since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)
    out = set()
    for r in hist:
        try:
            when = dt.datetime.fromisoformat(r["date"])
        except Exception:
            continue
        if when >= since:
            out |= {t.strip().lower() for t in (r.get("tools") or "").split(";") if t.strip()}
    return out


# =============================================================== 1. OFFICIAL PAGES (verify + screenshot)

HIDE_POPUPS_JS = """() => {
  const vw = innerWidth, vh = innerHeight;
  for (const el of document.querySelectorAll('body *')) {
    const s = getComputedStyle(el);
    if ((s.position === 'fixed' || s.position === 'sticky') ) {
      const r = el.getBoundingClientRect();
      const big = r.width * r.height > 0.25 * vw * vh;
      const bottom = r.top > vh * 0.45;
      const txt = (el.id + ' ' + el.className + ' ' + (el.getAttribute('aria-label') || '')).toString().toLowerCase();
      if (big || bottom || /cookie|consent|gdpr|banner|modal|popup|dialog|newsletter|onetrust|cky|cmp/.test(txt))
        el.style.setProperty('display', 'none', 'important');
    }
  }
  // embedded players (YouTube etc.) show "sign in / not a bot" boxes in a headless browser - hide them
  for (const el of document.querySelectorAll('iframe, embed, object, lite-youtube, [class*="youtube" i], [id*="youtube" i]'))
    el.style.setProperty('display', 'none', 'important');
  document.documentElement.style.setProperty('overflow', 'auto', 'important');
  document.body.style.setProperty('overflow', 'auto', 'important');
}"""
META_JS = """() => {
  const m = n => (document.querySelector(`meta[name="${n}"],meta[property="${n}"]`) || {}).content || '';
  const icons = [...document.querySelectorAll('link[rel~="icon"],link[rel="apple-touch-icon"],link[rel="apple-touch-icon-precomposed"]')]
    .map(l => ({href: l.href, sizes: l.getAttribute('sizes') || '', rel: l.rel}));
  return {desc: m('description') + ' ' + m('og:description'), site: m('og:site_name'), icons};
}"""


def verify_page(tool, title, text, meta):
    """The tool's own name must be on its official page, and the page must be a real page (not a block/404)."""
    blob = f"{title} {meta.get('desc', '')} {meta.get('site', '')} {text}"
    if len(text) < 300 or BLOCKED.search(title) or (len(text) < 1500 and BLOCKED.search(text[:600])):
        return False
    names = [tool["name"]] + list(tool.get("aliases") or [])
    return any(re.search(r"(?<![a-z0-9])" + re.escape(n.lower()) + r"(?![a-z0-9])", blob.lower()) for n in names)


def icon_candidates(meta, url):
    def size(i):
        m = re.findall(r"(\d+)x\d+", i.get("sizes") or "")
        return max([int(x) for x in m] or [180 if "apple" in i.get("rel", "") else 32])
    icons = sorted((i for i in meta.get("icons") or [] if i.get("href") and not i["href"].endswith(".svg")),
                   key=lambda i: -size(i))
    host = urllib.parse.urlparse(url).netloc
    return [i["href"] for i in icons] + [f"https://www.google.com/s2/favicons?domain={host}&sz=256"]


def get_logo(cands, out):
    for u in cands:
        try:
            body, _ = http(u, raw=True, timeout=30, headers={"User-Agent": UA_MOBILE})
            im = Image.open(io.BytesIO(body))
            im.load()
            if min(im.size) >= 48:
                im.convert("RGBA").save(out)
                return out
        except Exception:
            continue
    return None


def capture(tools, work, want, min_text=0):
    """Opens each tool's official page (mobile browser), verifies it and takes a long screenshot.
    Returns verified tools (up to `want`) with page text, screenshot and logo."""
    from playwright.sync_api import sync_playwright
    got = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": 412, "height": 880}, device_scale_factor=2.5, is_mobile=True,
                                  has_touch=True, user_agent=UA_MOBILE, locale="en-US")
        for t in tools:
            if len(got) >= want:
                break
            for url in t["urls"]:
                page = ctx.new_page()
                try:
                    resp = page.goto(url, wait_until="domcontentloaded", timeout=45000)
                    if resp is not None and resp.status >= 400:
                        raise RuntimeError(f"HTTP {resp.status}")
                    try:
                        page.wait_for_load_state("networkidle", timeout=12000)
                    except Exception:
                        pass
                    for _ in range(5):  # wake lazy images, then back to the top
                        page.mouse.wheel(0, 900)
                        page.wait_for_timeout(300)
                    page.evaluate("window.scrollTo(0, 0)")
                    page.wait_for_timeout(900)
                    page.evaluate(HIDE_POPUPS_JS)
                    title, text, meta = page.title(), page.inner_text("body"), page.evaluate(META_JS)
                    text = re.sub(r"[ \t]+", " ", re.sub(r"\n\s*\n+", "\n", text)).strip()
                    if not verify_page(t, title, text, meta) or len(text) < min_text:
                        log(f"Not verified: {t['name']} at {url} (title: {title[:60]!r})")
                        continue
                    slug = base.slugify(t["name"])
                    shot = work / f"site_{slug}.png"
                    height = min(2600, page.evaluate("document.documentElement.scrollHeight") or 2600)
                    page.screenshot(path=str(shot), full_page=True, clip={"x": 0, "y": 0, "width": 412,
                                                                          "height": max(880, height)})
                    logo = get_logo(icon_candidates(meta, page.url), work / f"logo_{slug}.png")
                    facts = (clean(meta.get("desc")) + "\n" + text)[:6000]
                    got.append({**t, "url": page.url, "domain": urllib.parse.urlparse(page.url).netloc.replace("www.", ""),
                                "facts": facts, "shot": shot, "logo": logo})
                    log(f"Verified on the official site: {t['name']} ({page.url})")
                    break
                except Exception as e:
                    log(f"Official page failed {t['name']} {url}: {str(e)[:120]}")
                finally:
                    page.close()
        browser.close()
    return got


# =============================================================== 2. SCRIPT (facts only from the official pages)

def quote_ok(quote, facts):
    q, f = norm(quote), norm(facts)
    if len(q.split()) < 3:
        return False
    if q in f:
        return True
    m = difflib.SequenceMatcher(None, f, q, autojunk=False).find_longest_match(0, len(f), 0, len(q))
    return m.size >= 0.8 * len(q)


def spoken_lines(s, fmt):
    """The narration: hook line(s), then per item 'Number one: Name.' + what it does + example, then outro."""
    lines = [{"speaker": h["speaker"], "text": h["text"], "seg": 0} for h in s["hook"]]
    n = len(s["items"])
    for k, it in enumerate(s["items"]):
        lines.append({"speaker": "A", "text": f"Number {NUM_WORDS[k]}: {it['label']}.", "seg": k + 1})
        lines.append({"speaker": "A", "text": it["does"], "seg": k + 1})
        lines.append({"speaker": "A", "text": it["example"], "seg": k + 1})
    lines.append({"speaker": "A", "text": OUTRO, "seg": n + 1})
    return lines


def problems(s, fmt, tools, style):
    """Everything that would make the Short untrue, hypey or off-format."""
    out = []
    by_name = {t["name"].lower(): t for t in tools}
    hook = s.get("hook") or []
    if not hook or not all(h.get("text") for h in hook):
        out.append("hook missing")
    if style == "qa_dialog" and [h.get("speaker") for h in hook] != ["A", "B"]:
        out.append("qa_dialog hook must be exactly 2 lines: speaker A asks, speaker B answers")
    if style != "qa_dialog" and any(h.get("speaker") != "A" for h in hook):
        out.append("only speaker A in this hook")
    items = s.get("items") or []
    if len(items) != 3:
        out.append("need exactly 3 items")
    for it in items:
        tool = by_name.get((it.get("tool") or "").lower())
        if not tool:
            out.append(f"'{it.get('tool')}' is not one of the verified tools")
            continue
        if not quote_ok(it.get("quote") or "", tool["facts"]):
            out.append(f"{tool['name']}: the quote is not copied word for word from its official text")
        said = f"{it.get('does', '')} {it.get('example', '')}"
        if re.search(r"\bfree\b", said, re.I) and not re.search(r"\bfree\b", tool["facts"], re.I):
            out.append(f"{tool['name']}: says 'free' but its official text does not")
        if len((it.get("does") or "").split()) > 18 or len((it.get("example") or "").split()) > 24:
            out.append(f"{tool['name']}: lines too long")
    if fmt == "listicle" and len({(it.get("tool") or "").lower() for it in items}) != len(items):
        out.append("the 3 tools must be different")
    said = " ".join([h.get("text", "") for h in hook] + [f"{i.get('does', '')} {i.get('example', '')}" for i in items]
                    + [s.get("title", ""), s.get("headline", "")])
    if HYPE.search(said):
        out.append(f"hype words ({HYPE.search(said).group(0)})")
    if FAKE_EXPERIENCE.search(said):
        out.append(f"fake personal experience ({FAKE_EXPERIENCE.search(said).group(0)})")
    used = [by_name.get((it.get("tool") or "").lower()) for it in items]
    if re.search(r"\bfree\b", " ".join([h.get("text", "") for h in hook] + [s.get("title", "")]), re.I) \
            and not all(t and re.search(r"\bfree\b", t["facts"], re.I) for t in used):
        out.append("hook/title says 'free' but not every tool's official text says free")
    return out


def write_script(fmt, audience, style, tools, note=""):
    if fmt == "listicle":
        what = (f'a list Short: "3 AI tools for {AUDIENCES[audience]}". Pick the 3 tools below that are most '
                f"useful for {AUDIENCES[audience]}; each item = one tool")
        item_hint = '"tool": "exact tool name from the list", "label": "the tool name as spoken"'
    else:
        what = (f"an explainer Short about ONE tool, {tools[0]['name']}: 3 different useful things it can do, "
                "for ordinary people; each item = one feature")
        item_hint = (f'"tool": "{tools[0]["name"]}", "label": "feature name, 2-4 words, sentence case"')
    facts = "\n\n".join(f"### {t['name']} (official site {t['domain']})\n{t['facts'][:2600 if fmt == 'listicle' else 5000]}"
                        for t in tools)
    return base.ask_json(f"""You write YouTube Shorts for "{CHANNEL}": calm, friendly explanations of real AI tools in
plain English, like a helpful friend - never like an ad. This Short is {what}.

OFFICIAL WEBSITE TEXT (the ONLY facts you may use; nothing from memory, no guesses):
{facts}

HOOK STYLE for this Short: {style} = {STYLES[style]}
Rules:
- The hook is 1 short line (under 14 words){' - here exactly 2 lines: A asks, B answers' if style == 'qa_dialog' else ''}.
  It must be TRUE. No invented numbers, prices, times, rankings or "best". No "I tried / I use / my favorite" -
  you have no personal experience. No hype words (revolutionary, game-changer, insane, secret, mind-blowing...).
- For every item: "does" = ONE plain sentence (max 15 words) saying what it does; "example" = ONE sentence
  (max 20 words) with a concrete everyday example of how to use it ("Upload your class notes and ask for a quiz.").
- "quote" = 5-20 words copied EXACTLY, word for word, from that tool's official text above, that prove "does".
- Only say "free" if the official text says free. Do not mention prices.
- Simple words, sentence case, no emojis, no exclamation marks.
{note}
Return ONLY JSON:
{{"headline": "on-screen title, max 6 words, sentence case (e.g. '3 AI tools for students')",
  "hook": [{{"speaker": "A", "text": "..."}}],
  "items": [{{{item_hint}, "does": "...", "example": "...", "quote": "..."}}],
  "title": "YouTube title under 60 chars, honest, sentence case, ends with #shorts",
  "description": "2 short plain lines about what the viewer will learn",
  "tags": ["8-12 tags"], "hashtags": ["#aitools", "#ai", "one specific"]}}""", keys=("items", "hook"), max_tokens=2500)


def tidy(s):
    """Sentence case, no dashes or bullets, tidy punctuation."""
    s = base.text(s)
    s = re.sub(r"^\s*[-–—•*]+\s*", "", re.sub(r"\s*,?\s+[-–—]+\s+", ", ", s)).strip()
    if s.isupper():
        s = s.capitalize()
    return s.replace("!", ".")


def make_script(fmt, audience, style, tools):
    note, s = "", None
    for attempt in range(3):
        s = write_script(fmt, audience, style, tools, note)
        s["hook"] = [{"speaker": ("B" if str(h.get("speaker", "A")).strip().upper().startswith("B") else "A"),
                      "text": tidy(h.get("text"))} for h in (s.get("hook") or []) if isinstance(h, dict)]
        items = []
        for it in s.get("items") or []:
            if isinstance(it, dict):
                items.append({k: tidy(it.get(k)) for k in ("tool", "label", "does", "example", "quote")})
        s["items"] = items
        if fmt == "explainer":
            for it in items:
                it["tool"] = tools[0]["name"]
        for it in items:
            it["label"] = it["label"] or it["tool"]
        bad = problems(s, fmt, tools, style)
        if not bad:
            break
        log(f"Script attempt {attempt + 1} rejected: {'; '.join(bad)[:300]}")
        note = "Your last version broke these rules - fix them: " + "; ".join(bad)
    else:
        raise RuntimeError("no script passed the fact and style checks")
    by_name = {t["name"].lower(): t for t in tools}
    s["tools"] = [by_name[it["tool"].lower()] for it in s["items"]]
    s["headline"] = tidy(s.get("headline")) or (f"3 AI tools for {AUDIENCES[audience]}" if fmt == "listicle"
                                                 else tools[0]["name"])
    s["title"] = tidy(s.get("title")) or s["headline"]
    s["title"] = re.sub(r"\s*#shorts\b", "", s["title"], flags=re.I)[:80].strip() + " #shorts"
    s["format"], s["audience"], s["hook_style"] = fmt, audience, style
    s["lines"] = spoken_lines(s, fmt)
    return s


# =============================================================== 3. VOICE (calm, two voices for Q&A)

def google_line(text, voice, path):
    key = os.environ["GOOGLE_TTS_KEY"]
    body = {"input": {"text": text}, "voice": {"languageCode": "-".join(voice.split("-")[:2]), "name": voice},
            "audioConfig": {"audioEncoding": "MP3", "speakingRate": RATE, "sampleRateHertz": 24000}}
    last = None
    for attempt in range(5):
        try:
            r = http(f"https://texttospeech.googleapis.com/v1/text:synthesize?key={key}", body, timeout=120)
            Path(path).write_bytes(base64.b64decode(r["audioContent"]))
            return
        except urllib.error.HTTPError as e:
            txt = e.read()[:300]
            last = f"{e.code} {txt!r}"
            if e.code in (429, 500, 503):
                time.sleep(min(base.MAX_WAIT, base.retry_after(e, txt) or 6 * (attempt + 1)))
                continue
            break
        except Exception as e:
            last = str(e)[:200]
            time.sleep(3)
    raise RuntimeError(f"Google TTS {voice}: {last}")


def edge_line(text, voice, path):
    import edge_tts
    pct = f"{round((RATE - 1) * 100):+d}%"
    last = None
    for attempt in range(4):
        try:
            asyncio.run(edge_tts.Communicate(text, voice, rate=pct).save(str(path)))
            if Path(path).exists() and Path(path).stat().st_size > 1000:
                return
            last = "empty audio"
        except Exception as e:
            last = str(e)[:200]
        time.sleep(4 * (attempt + 1))
    raise RuntimeError(f"edge-tts {voice}: {last}")


def wav_frames(src, dst):
    run(["ffmpeg", "-y", "-v", "error", "-i", src, "-ac", "1", "-ar", "24000", "-sample_fmt", "s16", dst])
    with wave.open(str(dst), "rb") as w:
        return w.readframes(w.getnframes())


LINE_GAP, SEG_GAP, LEAD, TAIL = 0.22, 0.5, 0.08, 0.7


def narrate(lines, work):
    """One TTS call per line (exact timing per card). Google Chirp 3 HD; if any line fails, the whole
    Short is redone with the free edge-tts voices so the two engines are never mixed."""
    engines = (["google"] if os.getenv("GOOGLE_TTS_KEY") else []) + ["edge"]
    for eng in engines:
        try:
            for i, ln in enumerate(lines):
                p = work / f"line{i:02d}.mp3"
                if eng == "google":
                    google_line(ln["text"], VOICE_B if ln["speaker"] == "B" else VOICE_A, p)
                else:
                    edge_line(ln["text"], EDGE_B if ln["speaker"] == "B" else EDGE_A, p)
            log(f"Voice: {eng} ({VOICE_A if eng == 'google' else EDGE_A})")
            break
        except Exception as e:
            log(f"Voice engine {eng} failed ({str(e)[:150]}) - trying the next one")
    else:
        raise RuntimeError("no voice engine worked")
    rate, pcm, t = 24000, bytearray(), 0.0

    def silence(sec):
        return b"\x00\x00" * int(sec * rate)
    pcm += silence(LEAD)
    t = LEAD
    for i, ln in enumerate(lines):
        if i:
            gap = SEG_GAP if ln["seg"] != lines[i - 1]["seg"] else LINE_GAP
            pcm += silence(gap)
            t += gap
        frames = wav_frames(work / f"line{i:02d}.mp3", work / f"line{i:02d}.wav")
        ln["start"], ln["end"] = t, t + len(frames) / 2 / rate
        pcm += frames
        t = ln["end"]
    pcm += silence(TAIL)
    voice = work / "voice.wav"
    with wave.open(str(voice), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(pcm))
    return voice, t + TAIL


# =============================================================== 4. DESIGN (light cards)

def font(size, bold=True):
    names = (["Poppins-Bold.ttf"] if bold else ["Poppins-SemiBold.ttf"]) + ["Poppins-SemiBold.ttf"]
    dirs = [Path.home() / ".local/share/fonts", Path("/usr/share/fonts/truetype/poppins"), Path("C:/Windows/Fonts")]
    for d in dirs:
        for n in names:
            if (d / n).exists():
                return ImageFont.truetype(str(d / n), size)
    for f in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "C:/Windows/Fonts/segoeuib.ttf",
              "C:/Windows/Fonts/arialbd.ttf"]:
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def background():
    grad = Image.new("RGB", (1, H))
    grad.putdata([tuple(int(a + (b - a) * y / (H - 1)) for a, b in zip(BG_TOP, BG_BOTTOM)) for y in range(H)])
    return grad.resize((W, H)).convert("RGBA")


def shadowed(img, box, radius, fill=(255, 255, 255, 255), blur=26, dy=16, alpha=46):
    x0, y0, x1, y1 = box
    sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(sh).rounded_rectangle((x0, y0 + dy, x1, y1 + dy), radius, fill=(30, 41, 59, alpha))
    img.alpha_composite(sh.filter(ImageFilter.GaussianBlur(blur)))
    ImageDraw.Draw(img).rounded_rectangle(box, radius, fill=fill)


def fit_font(text, max_w, start, bold=True, low=40):
    size = start
    while size > low:
        f = font(size, bold)
        if f.getlength(text) <= max_w:
            return f
        size -= 4
    return font(low, bold)


def wrap(text, f, max_w):
    words, lines, cur = text.split(), [], ""
    for w in words:
        nxt = (cur + " " + w).strip()
        if f.getlength(nxt) <= max_w or not cur:
            cur = nxt
        else:
            lines.append(cur)
            cur = w
    return lines + ([cur] if cur else [])


def logo_tile(img, tool, x, y, size):
    shadowed(img, (x, y, x + size, y + size), size // 4, blur=18, dy=10, alpha=40)
    pad = size // 7
    if tool.get("logo") and Path(tool["logo"]).exists():
        lg = Image.open(tool["logo"]).convert("RGBA")
        lg.thumbnail((size - 2 * pad, size - 2 * pad), Image.LANCZOS)
        if lg.width < size - 2 * pad and lg.height < size - 2 * pad:  # small favicon: scale up to fill
            k = (size - 2 * pad) / max(lg.size)
            lg = lg.resize((max(1, int(lg.width * k)), max(1, int(lg.height * k))), Image.LANCZOS)
        img.alpha_composite(lg, (x + (size - lg.width) // 2, y + (size - lg.height) // 2))
    else:
        d = ImageDraw.Draw(img)
        f = font(int(size * 0.5))
        ch = tool["name"][:1].upper()
        d.text((x + size / 2, y + size / 2), ch, font=f, fill=ACCENT, anchor="mm")


def pill(img, text, x, y, f, fill=ACCENT, ink=(255, 255, 255)):
    d = ImageDraw.Draw(img)
    w = int(f.getlength(text)) + 56
    h = int(f.size * 1.5)
    d.rounded_rectangle((x, y, x + w, y + h), h // 2, fill=fill)
    d.text((x + w / 2, y + h / 2), text, font=f, fill=ink, anchor="mm")
    return w, h


def item_card(path, k, n, tool, label):
    img = background()
    pill(img, f"#{k}/{n}", CARD_X, 112, font(44))
    logo_tile(img, tool, CARD_X, 200, 150)
    d = ImageDraw.Draw(img)
    tx = CARD_X + 150 + 34
    d.text((tx, 212), label, font=fit_font(label, W - tx - CARD_X, 84), fill=INK)
    sub = tool["domain"] if label.lower() == tool["name"].lower() else f"{tool['name']}  ·  {tool['domain']}"
    d.text((tx, 318), sub, font=fit_font(sub, W - tx - CARD_X, 36, bold=False, low=26), fill=MUTED)
    shadowed(img, (CARD_X, CARD_Y, CARD_X + CARD_W, CARD_Y + CARD_H), 40, blur=34, dy=22, alpha=60)
    img.convert("RGB").save(path)


def title_card(path, headline, tools, kicker=CHANNEL):
    img = background()
    d = ImageDraw.Draw(img)
    f = font(46, bold=False)
    d.text((W / 2, 560), kicker, font=f, fill=ACCENT, anchor="mm")
    hf = font(108)
    lines = wrap(headline, hf, W - 180)
    if len(lines) > 3:
        hf = font(88)
        lines = wrap(headline, hf, W - 180)
    y = 660
    for ln in lines:
        d.text((W / 2, y), ln, font=hf, fill=INK, anchor="mt")
        y += int(hf.size * 1.18)
    size = 200 if len(tools) > 1 else 300
    gap = 60
    total = len(tools) * size + (len(tools) - 1) * gap
    x = (W - total) // 2
    for t in tools:
        logo_tile(img, t, x, max(y + 100, 1060), size)
        x += size + gap
    img.convert("RGB").save(path)


def outro_card(path, tools):
    img = background()
    d = ImageDraw.Draw(img)
    size, gap = 150, 44
    total = len(tools) * size + (len(tools) - 1) * gap
    x = (W - total) // 2
    for t in tools:
        logo_tile(img, t, x, 600, size)
        x += size + gap
    hf = font(96)
    y = 870
    for ln in wrap(OUTRO, hf, W - 200):
        d.text((W / 2, y), ln, font=hf, fill=INK, anchor="mt")
        y += int(hf.size * 1.2)
    f = font(46)
    w = int(f.getlength(CHANNEL)) + 56
    pill(img, CHANNEL, (W - w) // 2, y + 70, f)
    img.convert("RGB").save(path)


def prepare_shot(src, out):
    im = Image.open(src).convert("RGB")
    k = CARD_W / im.width
    im = im.resize((CARD_W, max(CARD_H, int(im.height * k))), Image.LANCZOS)
    if im.height < CARD_H:
        pad = Image.new("RGB", (CARD_W, CARD_H), (255, 255, 255))
        pad.paste(im, (0, 0))
        im = pad
    im.save(out)
    return out, im.height


def card_mask(out):
    m = Image.new("L", (CARD_W, CARD_H), 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, CARD_W - 1, CARD_H - 1), 40, fill=255)
    m.save(out)
    return out


# =============================================================== 5. RENDER

def segment(bg, dur, out, shot=None, shot_h=0, start_y=0, mask=None):
    enc = ["-r", str(FPS), "-c:v", "libx264", "-preset", "veryfast", "-crf", "19", "-pix_fmt", "yuv420p",
           "-t", f"{dur:.3f}", out]
    if not shot:
        run(["ffmpeg", "-y", "-v", "error", "-loop", "1", "-framerate", str(FPS), "-t", f"{dur:.3f}", "-i", bg,
             "-vf", "fade=t=in:st=0:d=0.25:color=white,format=yuv420p", *enc])
        return
    maxy = max(0, shot_h - CARD_H)
    start = min(start_y, maxy)
    travel = maxy - start
    speed = min(70.0, travel / max(1.0, dur - 1.6)) if travel > 0 else 0.0
    yexpr = f"{start}+min({travel},max(0,t-1.0)*{speed:.2f})"
    fc = (f"[1:v]crop={CARD_W}:{CARD_H}:0:'{yexpr}',format=rgba[s];[2:v]format=gray[m];[s][m]alphamerge[sm];"
          f"[0:v][sm]overlay={CARD_X}:{CARD_Y},fade=t=in:st=0:d=0.25:color=white,format=yuv420p[v]")
    run(["ffmpeg", "-y", "-v", "error",
         "-loop", "1", "-framerate", str(FPS), "-t", f"{dur:.3f}", "-i", bg,
         "-loop", "1", "-framerate", str(FPS), "-t", f"{dur:.3f}", "-i", shot,
         "-loop", "1", "-framerate", str(FPS), "-t", f"{dur:.3f}", "-i", mask,
         "-filter_complex", fc, "-map", "[v]", *enc])


def caption_items(lines):
    """White sentence-case captions, 2-3 words at a time, timed to the spoken words."""
    items = []
    for i, ln in enumerate(lines):
        groups = [[w for w, _ in g] for g in base.word_groups(ln["text"], most=3)]
        times = base.WORD_TIMES.get(i)
        n_words = sum(len(g) for g in groups)
        if times and len(times) == n_words:
            k, starts = 0, []
            for g in groups:
                starts.append(max(ln["start"], min(times[k][0], ln["end"])))
                k += len(g)
            ends = starts[1:] + [min(ln["end"] + 0.25, times[-1][1] + 0.25)]
        else:  # estimate by characters
            tot = sum(len(" ".join(g)) + 2 for g in groups) or 1
            starts, ends, t = [], [], ln["start"]
            for g in groups:
                d = (ln["end"] - ln["start"]) * (len(" ".join(g)) + 2) / tot
                starts.append(t)
                ends.append(t + d)
                t += d
        for g, s, e in zip(groups, starts, ends):
            txt = re.sub(r"[.,;:]+$", "", " ".join(g))
            items.append((s, max(e, s + 0.25), txt))
    return items


def caption_ass(items, path):
    fam = "DejaVu Sans"
    try:
        if "Poppins" in base.subprocess.run(["fc-list", ":family"], capture_output=True, text=True).stdout:
            fam = "Poppins"
    except Exception:
        pass
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{fam},66,&H00FFFFFF,&H00FFFFFF,&H4D1F1611,&H00000000,0,0,0,0,100,100,0,0,3,20,0,5,80,80,0,1
"""
    ev = []
    for s, e, t in items:
        t = t.replace("{", "(").replace("}", ")").replace("\\", "")
        ev.append(f"Dialogue: 0,{base.ass_time(s)},{base.ass_time(e)},Cap,,0,0,0,,{{\\an5\\pos({W // 2},{CAP_Y})\\fad(70,0)}}{t}")
    path.write_text(head + "\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
                    + "\n".join(ev) + "\n", encoding="utf-8")


def music(work, seed):
    names = MUSIC[:]
    random.Random(seed).shuffle(names)
    for name in names:
        try:
            body, _ = http("https://incompetech.com/music/royalty-free/mp3-royaltyfree/" + urllib.parse.quote(name)
                           + ".mp3", raw=True, timeout=90, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            if len(body) > 300_000:
                p = work / "music.mp3"
                p.write_bytes(body)
                return p, (f'Music: "{name}" Kevin MacLeod (incompetech.com)\n'
                           "Licensed under Creative Commons: By Attribution 4.0 License\n"
                           "http://creativecommons.org/licenses/by/4.0/")
        except Exception as e:
            log(f"Music download failed '{name}': {e}")
    return None, ""


def render(s, work, k):
    lines = s["lines"]
    voice, total = narrate(lines, work)
    if total > 59.0:  # keep it a classic under-a-minute Short; never faster than 1.1x
        f = min(1.1, total / 58.5)
        fast = work / "voice_fast.wav"
        run(["ffmpeg", "-y", "-v", "error", "-i", voice, "-af", f"atempo={f:.4f}", fast])
        fast.replace(voice)
        for ln in lines:
            ln["start"], ln["end"] = ln["start"] / f, ln["end"] / f
        total /= f
        log(f"Short {k + 1}: voice x{f:.2f} to stay under a minute")
    base.WORD_TIMES.clear()
    base.align_words([{"ur": ln["text"]} for ln in lines], voice)

    n = len(s["items"])
    seg_start = [0.0]
    for j in range(1, n + 2):
        first = next(ln for ln in lines if ln["seg"] == j)
        seg_start.append(first["start"] - SEG_GAP / 2)
    seg_start.append(total)
    mask = card_mask(work / "mask.png")
    segs = []
    for j in range(n + 2):
        dur = seg_start[j + 1] - seg_start[j]
        bg, out = work / f"card{j}.png", work / f"seg{j}.mp4"
        if j == 0:
            title_card(bg, s["headline"], s["tools"] if s["format"] == "listicle" else s["tools"][:1])
            segment(bg, dur, out)
        elif j == n + 1:
            outro_card(bg, list({t["name"]: t for t in s["tools"]}.values()))
            segment(bg, dur, out)
        else:
            it, tool = s["items"][j - 1], s["tools"][j - 1]
            item_card(bg, j, n, tool, it["label"])
            shot, sh = prepare_shot(tool["shot"], work / f"shot{j}.png")
            start_y = 0 if s["format"] == "listicle" else int((sh - CARD_H) * (j - 1) / 3)
            segment(bg, dur, out, shot, sh, start_y, mask)
        segs.append(out)
    (work / "segs.txt").write_text("".join(f"file '{p.name}'\n" for p in segs))
    run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", work / "segs.txt", "-c", "copy",
         work / "visual.mp4"])
    ass = work / "caps.ass"
    caption_ass(caption_items(lines), ass)
    subs = str(ass).replace("\\", "/").replace(":", "\\:")
    track, mcredit = music(work, s["title"])
    out = work / f"ai_short{k}.mp4"
    ins = ["-i", work / "visual.mp4", "-i", voice]
    fc = (f"[0:v]subtitles='{subs}',format=yuv420p[v];"
          "[1:a]highpass=f=70,acompressor=threshold=-20dB:ratio=2.5:attack=8:release=200,"
          "loudnorm=I=-15:TP=-1.5,aresample=44100,asplit=2[vo][sc];")
    if track:
        ins += ["-stream_loop", "-1", "-i", track]
        fc += ("[2:a]aresample=44100,loudnorm=I=-31:TP=-6[m];"
               "[m][sc]sidechaincompress=threshold=0.05:ratio=3:attack=40:release=700[duck];"
               "[vo][duck]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.95[a]")
    else:
        fc += "[sc]anullsink;[vo]alimiter=limit=0.95[a]"
    run(["ffmpeg", "-y", "-v", "error", *ins, "-filter_complex", fc, "-map", "[v]", "-map", "[a]",
         "-t", f"{total:.2f}", "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-c:a", "aac", "-b:a", "160k",
         "-movflags", "+faststart", out])
    s["seg_start"] = seg_start
    log(f"Short {k + 1} ready: {out.name} ({total:.1f}s, {s['format']}, hook {s['hook_style']}, "
        f"tools {', '.join(t['name'] for t in s['tools'])})")
    return out, mcredit


# =============================================================== 6. YOUTUBE

def yt_token():
    r = http("https://oauth2.googleapis.com/token", urllib.parse.urlencode({
        "client_id": os.environ["AI_YT_CLIENT_ID"], "client_secret": os.environ["AI_YT_CLIENT_SECRET"],
        "refresh_token": os.environ["AI_YT_REFRESH_TOKEN"], "grant_type": "refresh_token"}).encode(),
        headers={"Content-Type": "application/x-www-form-urlencoded"})
    return r["access_token"]


def yt_upload(tok, video, title, desc, tags, publish_at=None):
    status = {"privacyStatus": PRIVACY, "selfDeclaredMadeForKids": False, "embeddable": True,
              "containsSyntheticMedia": AI_LABEL}
    if publish_at and PRIVACY == "public":
        status["privacyStatus"], status["publishAt"] = "private", publish_at
    meta = {"snippet": {"title": title[:100], "description": desc[:4900], "tags": tags[:15], "categoryId": "28",
                        "defaultLanguage": "en", "defaultAudioLanguage": "en"}, "status": status}
    size = Path(video).stat().st_size
    _, hdr = http("https://www.googleapis.com/upload/youtube/v3/videos?uploadType=resumable&part=snippet,status",
                  meta, headers={"Authorization": f"Bearer {tok}", "X-Upload-Content-Type": "video/mp4",
                                 "X-Upload-Content-Length": str(size)}, raw=True)
    req = urllib.request.Request(hdr["Location"], data=Path(video).read_bytes(), method="PUT",
                                 headers={"Authorization": f"Bearer {tok}", "Content-Type": "video/mp4",
                                          "User-Agent": base.UA})
    with urllib.request.urlopen(req, timeout=3600) as r:
        res = json.loads(r.read())
    chan = (res.get("snippet") or {}).get("channelTitle") or "?"
    log(f"Uploaded https://youtu.be/{res['id']} to channel '{chan}' ({res['status'].get('privacyStatus')}"
        f"{', publishes ' + publish_at if publish_at else ''})")
    return res["id"], res["status"].get("privacyStatus"), chan


def same_channel(a, b):
    k = lambda s: re.sub(r"[^a-z0-9]", "", (s or "").lower())
    return k(a) == k(b)


def description(s, mcredit):
    tags = " ".join(([h if str(h).startswith("#") else "#" + str(h) for h in s.get("hashtags") or []]
                     or ["#aitools", "#ai"])[:3])
    seen, sites = set(), ""
    for t in s["tools"]:
        if t["name"] not in seen:
            seen.add(t["name"])
            sites += f"- {t['name']}: {t['url']}\n"
    return (f"{base.text(s.get('description'), s['headline'])}\n\n"
            f"Official websites:\n{sites}\n"
            f"{OUTRO}\n\n" + (mcredit + "\n\n" if mcredit else "") + tags)


# =============================================================== MAIN

def preview(frames, out):
    try:
        ims = [Image.open(f).convert("RGB").resize((270, 480)) for f in frames if f and Path(f).exists()]
        sheet = Image.new("RGB", (len(ims) * 280 + 10, 500), (226, 232, 240))
        for n, im in enumerate(ims):
            sheet.paste(im, (10 + n * 280, 10))
        out.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(out, quality=82)
    except Exception as e:
        log(f"Preview skipped: {e}")


def plan_short(k, hist, taken, rng):
    """Format, audience and hook style for Short k (rotations based on the log + this run)."""
    recent_fmt = [r.get("format") for r in hist][-5:] + [t["format"] for t in taken]
    fmt = "explainer" if k >= 1 and "explainer" not in recent_fmt else "listicle"
    recent_aud = [r.get("audience") for r in hist if r.get("format") == "listicle"][-5:] + \
                 [t["audience"] for t in taken if t["format"] == "listicle"]
    audience = pick_least_recent(list(AUDIENCES), recent_aud, rng, set(recent_aud[-1:]))
    style = pick_style([r.get("hook_style") for r in hist] + [t["hook_style"] for t in taken], rng)
    return fmt, audience, style


def choose_tools(fmt, audience, hist, taken_tools, rng):
    tools = load_tools()
    avoid = recent_tools(hist, 14 if fmt == "listicle" else 30) | taken_tools
    if fmt == "listicle":
        pool = [t for t in tools if audience in t["for"]]
    else:
        pool = list(tools)
    rng.shuffle(pool)
    fresh = [t for t in pool if t["name"].lower() not in avoid]
    return fresh + [t for t in pool if t["name"].lower() in avoid and t["name"].lower() not in taken_tools]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--count", type=int, default=int(os.getenv("AI_SHORTS_COUNT") or 2))
    a = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    hist = history()
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    if os.getenv("RUN_EVENT") == "schedule" and not a.no_upload and any(r.get("date", "").startswith(today) for r in hist):
        log("Today's Shorts are already uploaded - this backup run has nothing to do")
        return
    rng =random.Random(dt.date.today().toordinal())
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M")
    made, taken, taken_tools = [], [], set()
    for k in range(a.count):
        fmt, audience, style = plan_short(k, hist, taken, rng)
        failed_aud = set()
        for attempt in range(3):  # a failed Short is retried with another audience, last try as an explainer
            if attempt:
                failed_aud.add(audience)
                if attempt == 2:
                    fmt = "explainer"
                else:
                    audience = pick_least_recent([x for x in AUDIENCES if x not in failed_aud], [], rng)
            log(f"Short {k + 1} (try {attempt + 1}): {fmt}, "
                f"{'for ' + AUDIENCES[audience] if fmt == 'listicle' else 'one tool'}, hook '{style}'")
            work = OUT / f"{stamp}-{k}-{attempt}-{fmt}"
            work.mkdir(parents=True, exist_ok=True)
            try:
                pool = choose_tools(fmt, audience, hist, taken_tools, rng)
                tools = capture(pool[:12], work, 5 if fmt == "listicle" else 1, 0 if fmt == "listicle" else 900)
                if len(tools) < (3 if fmt == "listicle" else 1):
                    raise RuntimeError(f"only {len(tools)} tools verified on their official sites")
                s = make_script(fmt, audience, style, tools)
                (work / "script.json").write_text(json.dumps({x: v for x, v in s.items() if x != "tools"} |
                                                             {"tools": [{"name": t["name"], "url": t["url"]} for t in s["tools"]]},
                                                             indent=1, ensure_ascii=False, default=str), encoding="utf-8")
                video, mcredit = render(s, work, k)
                made.append((s, video, mcredit, work))
                taken.append({"format": fmt, "audience": audience, "hook_style": style})
                taken_tools |= {t["name"].lower() for t in s["tools"]}
                break
            except Exception as e:  # one bad Short never stops the other
                log(f"Short {k + 1} try {attempt + 1} failed: {type(e).__name__}: {e}")
                import traceback
                log(traceback.format_exc()[-1500:])
    if not made:
        raise RuntimeError("No Short could be made")

    frames = []
    for s, video, _, work in made:
        for j in (0, 1, len(s["items"]) + 1):  # hook, first tool card, outro
            t = (s["seg_start"][j] + s["seg_start"][j + 1]) / 2
            frames.append(base.grab(video, t, work / f"pv{j}.jpg"))
    preview(frames, DATA / "previews" / "latest.jpg")

    results, wrong_channel = [], None
    missing = [x for x in ("AI_YT_CLIENT_ID", "AI_YT_CLIENT_SECRET", "AI_YT_REFRESH_TOKEN") if not os.getenv(x)]
    if a.no_upload:
        log("Test run: nothing uploaded (the Shorts are in the run's Artifacts)")
    elif missing:
        log(f"Not uploading: missing GitHub secrets {', '.join(missing)}")
    else:
        tok = yt_token()
        now = dt.datetime.now(dt.timezone.utc)
        for k, (s, video, mcredit, work) in enumerate(made):
            when = (now + dt.timedelta(minutes=30) + dt.timedelta(hours=4 * k)).replace(microsecond=0)
            when = when.isoformat().replace("+00:00", "Z")
            try:
                vid, st, chan = yt_upload(tok, video, s["title"], description(s, mcredit),
                                          [base.text(t) for t in s.get("tags") or [] if base.text(t)], publish_at=when)
            except Exception as e:
                log(f"Upload failed for '{s['title']}': {e}")
                continue
            if chan != "?" and not same_channel(chan, CHANNEL):
                wrong_channel = (vid, chan, when)
                log(f"WRONG CHANNEL: https://youtu.be/{vid} went to '{chan}', not '{CHANNEL}'. No more uploads. "
                    f"Delete it in YouTube Studio before {when} and make AI_YT_REFRESH_TOKEN again, "
                    f"choosing the {CHANNEL} channel.")
                break
            results.append((s, st, vid))
        new = not (DATA / "hooks.csv").exists()
        with open(DATA / "hooks.csv", "a", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
            if new:
                w.writeheader()
            for s, st, vid in results:
                w.writerow({"date": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                            "video": f"https://youtu.be/{vid}", "privacy": st, "format": s["format"],
                            "audience": s["audience"], "hook_style": s["hook_style"], "title": s["title"],
                            "tools": ";".join(dict.fromkeys(t["name"] for t in s["tools"]))})
    log(f"Done: {len(made)} Shorts made, {len(results)} uploaded")
    for s, video, _, work in made:
        for p in work.iterdir():
            if p.name not in {video.name, "script.json", "card0.png", "card1.png"} and not p.name.startswith("pv"):
                p.unlink(missing_ok=True)
    if wrong_channel:
        raise RuntimeError(f"Short https://youtu.be/{wrong_channel[0]} was uploaded to '{wrong_channel[1]}' "
                           f"instead of '{CHANNEL}' - delete it before {wrong_channel[2]}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FAILED: {e}")
        sys.exit(1)
