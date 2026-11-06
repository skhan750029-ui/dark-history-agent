#!/usr/bin/env python3
"""
Everyday AI Easy - 2 YouTube Shorts a day about new AI tools and AI news, in simple English.

Separate channel, separate bot: it only *uses* the tested building blocks of studio.py (free AI
writers with 429 backoff, Google TTS with the free edge-tts backup, word-synced captions, stock clip
search) and never changes the Dark History bot. Its own data lives in data/ai/.

Pipeline: official AI news feeds (OpenAI, Google, DeepMind, NVIDIA, Hugging Face, Apple, AWS,
Mistral) ranked by what people are talking about on Hacker News -> the 2 most useful stories for
ordinary people -> facts ONLY from the official post -> 45-55 s script (hook first, no hype, loop
ending) -> bright friendly voice -> modern stock clips (vertical) -> big centred captions -> upload
to the Everyday AI Easy channel. It uses Dark History's Google Cloud project / OAuth client
(YT_CLIENT_ID / YT_CLIENT_SECRET) with the new channel's own refresh token (AI_YT_REFRESH_TOKEN), so
both channels share one daily YouTube quota - hence 2 Shorts a day here.

Secrets: GROQ_API_KEY (+ optional OPENROUTER_API_KEY / CF_*), GOOGLE_TTS_KEY (optional, edge-tts
otherwise), PEXELS_API_KEY / PIXABAY_API_KEY, YT_CLIENT_ID / YT_CLIENT_SECRET, AI_YT_REFRESH_TOKEN
Usage:  python agent/ai_shorts.py [--no-upload] [--count 2]
"""
import argparse
import datetime as dt
import html
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# channel settings must be in place before studio.py reads them at import time
os.environ.setdefault("CHANNEL_NAME", "Everyday AI Easy")
os.environ["NARRATION_LANG"] = "en"
os.environ["TTS_RATE"] = os.getenv("AI_TTS_RATE") or "1.05"   # bright, lively pace
os.environ["CAPTION_SYNC"] = os.getenv("CAPTION_SYNC") or "on"
os.environ.pop("GEMINI_API_KEY", None)                       # Gemini voice is the Dark History narrator style

sys.path.insert(0, str(Path(__file__).resolve().parent))
import studio as base  # noqa: E402  (shared building blocks; nothing in studio.py is changed)

from PIL import Image  # noqa: E402

log, run, duration, http = base.log, base.run, base.duration, base.http
ROOT = base.ROOT
DATA = ROOT / "data" / "ai"
OUT = ROOT / "out" / "ai"
CHANNEL = os.environ["CHANNEL_NAME"]
PRIVACY = (os.getenv("PRIVACY") or "public").strip().lower()
W, H, FPS = 1080, 1920, base.FPS

# bright, friendly voices (Dark History uses the deep, slow Enceladus)
base.TTS_VOICES[:] = [v for v in [os.getenv("AI_TTS_VOICE"), "en-US-Chirp3-HD-Zephyr", "en-US-Chirp3-HD-Puck",
                                  "en-US-Chirp3-HD-Aoede"] if v]
base.EDGE_VOICE = os.getenv("AI_EDGE_VOICE") or "en-US-AvaMultilingualNeural"
base.GEMINI_TTS["on"] = False

FEEDS = [("OpenAI", "https://openai.com/news/rss.xml"),
         ("Google", "https://blog.google/technology/ai/rss/"),
         ("Google DeepMind", "https://deepmind.google/blog/rss.xml"),
         ("NVIDIA", "https://blogs.nvidia.com/feed/"),
         ("Hugging Face", "https://huggingface.co/blog/feed.xml"),
         ("Apple", "https://machinelearning.apple.com/rss.xml"),
         ("AWS", "https://aws.amazon.com/blogs/machine-learning/feed/"),
         ("Mistral AI", "https://mistral.ai/rss.xml")]
UA = "Mozilla/5.0 (compatible; EverydayAIEasyBot/1.0; +https://github.com/skhan750029-ui/dark-history-agent)"
HYPE = re.compile(r"revolution|game.?chang|mind.?blow|insane|crazy|unbelievable|shocking|you won'?t believe|"
                  r"changes everything|world.?changing|groundbreaking|magic|destroy|killer|the end of|"
                  r"terrifying|scary|replace (all|every)|jobs? (are )?gone|must.?see|secret", re.I)
MUSIC = ["Carefree", "Wallpaper", "Life of Riley", "Inspired", "Digital Lemonade", "Easy Lemon",
         "Local Forecast - Elevator", "Bossa Antigua"]


def get_text(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "en"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def clean(s):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s or ""))).strip()


# =============================================================== 1. TOPICS (official sources only)

def parse_feed(source, xml):
    items = []
    for block in re.findall(r"<(?:item|entry)\b.*?</(?:item|entry)>", xml, re.S)[:25]:
        title = clean((re.search(r"<title[^>]*>(.*?)</title>", block, re.S) or [None, ""])[1]
                      .replace("<![CDATA[", "").replace("]]>", ""))
        link = (re.search(r'<link[^>]*href="([^"]+)"', block) or re.search(r"<link>\s*(?:<!\[CDATA\[)?([^<\]]+)",
                                                                           block))
        date = re.search(r"<(?:pubDate|published|updated)>([^<]+)<", block)
        summ = re.search(r"<(?:description|summary|content)[^>]*>(.*?)</(?:description|summary|content)>", block, re.S)
        if not (title and link):
            continue
        when = None
        if date:
            for fmt in ("%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S %Z", "%Y-%m-%dT%H:%M:%S%z",
                        "%Y-%m-%dT%H:%M:%S.%f%z"):
                try:
                    when = dt.datetime.strptime(date.group(1).strip().replace("Z", "+0000"), fmt)
                    when = when if when.tzinfo else when.replace(tzinfo=dt.timezone.utc)
                    break
                except ValueError:
                    continue
        items.append({"source": source, "title": title, "url": link.group(1).strip(), "date": when,
                      "summary": clean((summ.group(1) if summ else "").replace("<![CDATA[", "").replace("]]>", ""))[:400]})
    return items


def hn_buzz(days=4):
    """Popular Hacker News stories of the last days: tells which official announcements people discuss."""
    since = int(time.time() - days * 86400)
    url = ("https://hn.algolia.com/api/v1/search_by_date?tags=story&hitsPerPage=400&numericFilters="
           + urllib.parse.quote(f"created_at_i>{since},points>15"))
    try:
        return [(h.get("title") or "", h.get("url") or "", int(h.get("points") or 0))
                for h in http(url, timeout=30).get("hits", [])]
    except Exception as e:
        log(f"Hacker News buzz failed: {e}")
        return []


def words(s):
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 2} - {"the", "and", "for", "with", "new",
                                                                               "how", "you", "our", "now", "its"}


def candidates(done, max_age_days=6):
    items = []
    for source, url in FEEDS:
        try:
            got = parse_feed(source, get_text(url))
            items += got
        except Exception as e:
            log(f"Feed {source} failed: {str(e)[:100]}")
    now = dt.datetime.now(dt.timezone.utc)
    items = [it for it in items if it["date"] and (now - it["date"]).days <= max_age_days
             and it["url"].strip().lower() not in done and it["title"].lower() not in done]
    buzz = hn_buzz()
    for it in items:
        tw, host = words(it["title"]), urllib.parse.urlparse(it["url"]).netloc
        it["buzz"] = sum(p for t, u, p in buzz if (u and u.rstrip("/") == it["url"].rstrip("/"))
                         or (host and host in u and len(tw & words(t)) >= 2) or len(tw & words(t)) >= 3)
        age_h = (now - it["date"]).total_seconds() / 3600
        it["score"] = it["buzz"] + max(0, 120 - age_h)  # discussed + fresh
    items.sort(key=lambda it: -it["score"])
    log(f"Official AI stories: {len(items)} fresh, top buzz {[it['buzz'] for it in items[:5]]}")
    return items[:18]


def article_text(url, chars=4500):
    """Main text of the official post (paragraphs, headings, list items)."""
    try:
        page = get_text(url, timeout=40)
    except Exception as e:
        log(f"Article fetch failed {url}: {e}")
        return ""
    page = re.sub(r"<(script|style|noscript|svg|nav|footer|header)\b.*?</\1>", " ", page, flags=re.S | re.I)
    parts = [clean(p) for p in re.findall(r"<(?:p|h1|h2|h3|li)\b[^>]*>(.*?)</(?:p|h1|h2|h3|li)>", page, re.S | re.I)]
    text = "\n".join(p for p in parts if len(p) > 40)
    return text[:chars]


def pick_stories(cands, count):
    lst = "\n".join(f"{i}. [{c['source']}, {c['date']:%d %b}] {c['title']} - {c['summary'][:200]}"
                    f" (people discussing: {c['buzz']})" for i, c in enumerate(cands))
    got = base.ask_json(f"""You run "{CHANNEL}", a YouTube Shorts channel that explains new AI tools and AI news to
ordinary people (not engineers) in simple English. From these OFFICIAL announcements pick the {count} that
ordinary people would find most useful or interesting today: new tools or features they can try, big
changes in apps they use, clear real-world impact. Skip deep research papers, enterprise-only cloud
details, hiring/partnership/finance posts and anything only engineers care about. Prefer stories people
are discussing. Each pick must be a different story.
{lst}
Return ONLY JSON: {{"picks": [{{"index": 0, "why": "one short reason"}}]}}""", keys=("picks",), max_tokens=1200)
    out, seen = [], set()
    for p in got.get("picks", []):
        try:
            i = int(p.get("index"))
        except (TypeError, ValueError):
            continue
        if 0 <= i < len(cands) and i not in seen:
            seen.add(i)
            out.append(cands[i])
    for i, c in enumerate(cands):  # fill up if the AI picked too few
        if len(out) >= count:
            break
        if i not in seen:
            out.append(c)
    return out[:count]


# =============================================================== 2. SCRIPT

def write_short(story, facts, note=""):
    s = base.ask_json(f"""Write a YouTube Short (45-55 seconds, 100-125 words) for "{CHANNEL}" about this OFFICIAL
announcement from {story['source']} ({story['date']:%d %B %Y}): "{story['title']}"
OFFICIAL SOURCE TEXT (the ONLY facts you may use - no outside facts, no guesses, no rumours):
{facts or story['summary']}

Rules:
- Simple, friendly, clear English for ordinary people; explain any technical word in a few plain words.
- NO hype: never use words like revolutionary, game-changer, insane, mind-blowing, shocking, secret,
  "changes everything", "will replace". Calm and honest. If something is not available yet or only in
  some countries/plans, say so.
- Line 1 = HOOK (under 12 words): a clear question or a useful surprise, e.g. "Your phone can now ... ?"
- Then: what it is, what you can do with it, who can use it / how to try it, one honest limit.
- Last line = LOOP line that flows back into the hook when the Short replays.
- No greeting, no "in this video", no channel name, no "like and subscribe" (the loop must stay clean).
- Split into scenes of ONE sentence each (6-9 scenes). For each scene give a stock-footage search "query"
  (2-4 simple English words that SHOW the sentence: e.g. "person using smartphone", "typing laptop office",
  "data center servers", "video editing computer", "student studying laptop", "doctor tablet").
{note}
Return ONLY JSON:
{{"title": "Shorts title under 55 chars, clear and honest, ends with #shorts",
  "scenes": [{{"text": "sentence", "query": "..."}}],
  "description": "2 short lines: what it is + why it matters (plain words)",
  "tags": ["8-12 tags"], "hashtags": ["#ai", "#aitools", "one specific"]}}""", keys=("scenes",), max_tokens=2500)
    scenes = []
    for sc in s.get("scenes") or []:
        t = base.text(sc.get("text") if isinstance(sc, dict) else sc)
        if t:
            scenes.append({"ur": t, "chapter": 0, "query": base.text(sc.get("query") if isinstance(sc, dict) else "",
                                                                       "person using laptop")})
    s["scenes"] = scenes
    s["title"] = base.text(s.get("title"), story["title"])[:95]
    if "#shorts" not in s["title"].lower():
        s["title"] = (s["title"][:86] + " #shorts").strip()
    return s


def script_for(story):
    facts = article_text(story["url"])
    log(f"Story: [{story['source']}] {story['title']} ({len(facts)} chars of official text)")
    s = write_short(story, facts)
    said = " ".join(sc["ur"] for sc in s["scenes"])
    if HYPE.search(said + " " + s["title"]) or len(said.split()) < 70:
        log("Script had hype words or was too short - rewriting once")
        s2 = write_short(story, facts, "Your last version broke the rules (hype words or too short). Fix that.")
        said2 = " ".join(sc["ur"] for sc in s2["scenes"])
        if len(s2["scenes"]) >= 5 and not HYPE.search(said2):
            s = s2
    if len(s["scenes"]) < 4:
        raise RuntimeError("script too short")
    s["source"], s["url"] = story["source"], story["url"]
    return s


# =============================================================== 3. VISUALS (modern, vertical)

AI_BAD = re.compile(r"cartoon|anime|animat|green.?screen|chroma|blue.?screen|emoji|mascot|logo|watermark|"
                    r"lettering|typograph|wedding|yoga|fitness|cooking|food|kids?\b|toy|horror|blood|gun|weapon|"
                    r"\bwar\b|explosion|halloween|zombie|skull", re.I)
CLIP_CACHE = {}


def find_clip(query, used):
    """Best matching unused stock clip (Pexels portrait first, then Pixabay), or None."""
    if query not in CLIP_CACHE:
        found = []
        key = os.getenv("PEXELS_API_KEY")
        if key:
            try:
                for v in http("https://api.pexels.com/videos/search?orientation=portrait&size=medium&per_page=30&query="
                              + urllib.parse.quote(query), headers={"Authorization": key}, timeout=40).get("videos", []):
                    slug = re.sub(r"[-\d]+", " ", (v.get("url") or "").rstrip("/").split("/")[-1])
                    files = [f for f in v.get("video_files", []) if f.get("file_type") == "video/mp4"
                             and 720 <= (f.get("width") or 0) <= 1440 and (f.get("height") or 0) > (f.get("width") or 0)]
                    if files and v.get("duration", 0) >= 3 and not AI_BAD.search(slug):
                        f = sorted(files, key=lambda f: f["width"])[-1]
                        found.append((f"px{v['id']}", f["link"], f"Video by {v.get('user', {}).get('name', 'Pexels')} on Pexels", slug))
            except Exception as e:
                log(f"Pexels failed '{query}': {e}")
        key = os.getenv("PIXABAY_API_KEY")
        if key:
            try:
                for v in http(f"https://pixabay.com/api/videos/?key={key}&safesearch=true&video_type=film&per_page=30&q="
                              + urllib.parse.quote(query[:100]), timeout=40).get("hits", []):
                    tags = v.get("tags") or ""
                    f = (v.get("videos") or {}).get("medium") or {}
                    if f.get("url") and v.get("duration", 0) >= 3 and not AI_BAD.search(tags):
                        found.append((f"pb{v['id']}", f["url"], f"Video by {v.get('user', 'Pixabay')} on Pixabay", tags))
            except Exception as e:
                log(f"Pixabay failed '{query}': {e}")
        CLIP_CACHE[query] = found
    best = sorted((c for c in CLIP_CACHE[query] if c[0] not in used), key=lambda c: -base.relevance(query, c[3]))
    return best[0] if best and base.relevance(query, best[0][3]) >= 1 else None


def clip(src, dur, out, idx):
    """Fits a clip to 1080x1920, bright and clean (this channel is not dark/desaturated)."""
    try:
        have = duration(src)
    except Exception:
        have = dur
    slow = f"setpts={dur / have:.4f}*PTS," if 0 < have < dur <= have * 1.6 else ""
    start = (have - dur) * ((idx * 0.37) % 1) if have > dur + 1 else 0
    vf = (f"{slow}scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},fps={FPS},"
          "eq=saturation=1.08:contrast=1.04:brightness=0.01,format=yuv420p")
    run(["ffmpeg", "-y", "-ss", f"{start:.2f}", "-stream_loop", "-1", "-i", src, "-t", f"{dur:.3f}", "-vf", vf,
         "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", out])


def backdrop(dur, out, idx):
    """Never fail for lack of footage: a soft moving gradient in the channel colours."""
    hue = [200, 260, 170, 30][idx % 4]
    run(["ffmpeg", "-y", "-f", "lavfi", "-i", f"gradients=s={W}x{H}:c0=0x1e3a8a:c1=0x06b6d4:speed=0.02:d={dur:.3f}",
         "-vf", f"hue=h={hue - 200},format=yuv420p", "-t", f"{dur:.3f}", "-c:v", "libx264", "-preset", "veryfast",
         "-crf", "22", out])


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


# =============================================================== 4. RENDER

def render(s, work, k, used, offline=False):
    scenes = s["scenes"]
    durs = base.narrate(scenes, work, offline)
    total = sum(durs)
    voice = work / "voice.wav"
    if total > 58.0:  # Shorts must stay under 60 s: gently speed the read up
        f = min(1.25, total / 57.5)
        fast = work / "voice_fast.wav"
        run(["ffmpeg", "-y", "-i", voice, "-af", f"atempo={f:.4f}", fast])
        fast.replace(voice)
        durs = [d / f for d in durs]
        total = sum(durs)
        log(f"Short {k + 1}: voice sped up x{f:.2f} to fit")
    base.WORD_TIMES.clear()  # never reuse the previous Short's word timing
    if not offline:
        base.align_words(scenes, voice)
    starts, t = [], 0.0
    for d in durs:
        starts.append(t)
        t += d
    credits, clips, last = [], [], None
    for i, (sc, du) in enumerate(zip(scenes, durs)):
        c = work / f"c{i:02d}.mp4"
        got = None if offline else (find_clip(sc["query"], used) or find_clip(" ".join(sc["query"].split()[:2]), used))
        if got:
            src = work / f"v{i:02d}.mp4"
            try:
                base.download(got[1], src)
                used.add(got[0])
                credits.append(got[2])
                last = src
            except Exception as e:
                log(f"Clip download failed: {e}")
                got = None
        if got or last:
            clip(src if got else last, du, c, i)
        else:
            backdrop(du, c, i)
        clips.append(c)
    (work / "clips.txt").write_text("".join(f"file '{c.name}'\n" for c in clips))
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", work / "clips.txt", "-c", "copy", work / "visual.mp4"])
    ass = work / "caps.ass"
    base.caption_file(base.chunk_captions(scenes, starts, durs), ass, W, H, 165)
    subs = str(ass).replace("\\", "/").replace(":", "\\:")
    track, mcredit = (None, "") if offline else music(work, s["title"])
    out = work / f"ai_short{k}.mp4"
    ins = ["-i", work / "visual.mp4", "-i", voice]
    fc = (f"[0:v]subtitles='{subs}',format=yuv420p[v];"
          "[1:a]highpass=f=70,acompressor=threshold=-18dB:ratio=3:attack=5:release=150,"
          "loudnorm=I=-14:TP=-1.5,aresample=44100,asplit=2[vo][sc];")
    if track:
        ins += ["-stream_loop", "-1", "-i", track]
        fc += ("[2:a]aresample=44100,loudnorm=I=-26:TP=-4,volume=1.0[m];"
               "[m][sc]sidechaincompress=threshold=0.05:ratio=4:attack=30:release=600[duck];"
               "[vo][duck]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.95[a]")
    else:
        fc += "[sc]anullsink;[vo]alimiter=limit=0.95[a]"
    run(["ffmpeg", "-y", *ins, "-filter_complex", fc, "-map", "[v]", "-map", "[a]", "-t", f"{total:.2f}",
         "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-c:a", "aac", "-b:a", "160k",
         "-movflags", "+faststart", out])
    log(f"Short {k + 1} ready: {out.name} ({total:.1f}s, {len(scenes)} scenes, {len(credits)} stock clips)")
    return out, list(dict.fromkeys(credits)), mcredit


# =============================================================== 5. YOUTUBE (own channel + project)

def yt_token():
    r = http("https://oauth2.googleapis.com/token", urllib.parse.urlencode({
        "client_id": os.environ["AI_YT_CLIENT_ID"], "client_secret": os.environ["AI_YT_CLIENT_SECRET"],
        "refresh_token": os.environ["AI_YT_REFRESH_TOKEN"], "grant_type": "refresh_token"}).encode(),
        headers={"Content-Type": "application/x-www-form-urlencoded"})
    return r["access_token"]


def yt_upload(tok, video, title, desc, tags, publish_at=None):
    status = {"privacyStatus": PRIVACY, "selfDeclaredMadeForKids": False, "embeddable": True,
              "containsSyntheticMedia": True}  # AI voice: honest disclosure
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
    log(f"Uploaded https://youtu.be/{res['id']} ({res['status'].get('privacyStatus')}"
        f"{', publishes ' + publish_at if publish_at else ''})")
    return res["id"], res["status"].get("privacyStatus")


def description(s, credits, mcredit):
    tags = " ".join(([h if str(h).startswith("#") else "#" + str(h) for h in s.get("hashtags") or []]
                     or ["#ai", "#aitools"])[:3])
    cred = "".join(f"- {c}\n" for c in credits[:12])
    return (f"{base.text(s.get('description'), s['title'])}\n\n"
            f"Official source ({s['source']}): {s['url']}\n\n"
            f"{CHANNEL}: new AI tools and AI news, explained simply. No hype, just what it does.\n"
            "Narration voice is AI-generated.\n\n"
            + (mcredit + "\n\n" if mcredit else "") + (f"Footage:\n{cred}\n" if cred else "") + tags)


# =============================================================== MAIN

def preview(frames, out):
    try:
        ims = [Image.open(f).convert("RGB").resize((270, 480)) for f in frames]
        sheet = Image.new("RGB", (len(ims) * 280 + 10, 500), (20, 20, 24))
        for n, im in enumerate(ims):
            sheet.paste(im, (10 + n * 280, 10))
        out.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(out, quality=80)
    except Exception as e:
        log(f"Preview skipped: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--count", type=int, default=int(os.getenv("AI_SHORTS_COUNT") or 2))
    a = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    done_f = DATA / "topics_done.txt"
    done = {l.strip().lower() for l in done_f.read_text(encoding="utf-8").splitlines()} if done_f.exists() else set()

    cands = candidates(done)
    if not cands:
        raise RuntimeError("No fresh official AI stories found today")
    stories = pick_stories(cands, a.count)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M")
    made, used = [], set()
    for k, story in enumerate(stories):
        work = OUT / f"{stamp}-{k}-{base.slugify(story['title'])[:40]}"
        work.mkdir(parents=True, exist_ok=True)
        try:
            s = script_for(story)
            (work / "script.json").write_text(json.dumps(s, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
            video, credits, mcredit = render(s, work, k, used, a.offline)
            made.append((s, video, credits, mcredit, work))
        except Exception as e:  # one bad story never stops the other Shorts
            log(f"Short {k + 1} failed ({story['title'][:60]}): {e}")
    if not made:
        raise RuntimeError("No Short could be made")

    frames = []
    for s, video, *_ in made:
        frames.append(base.grab(video, 1.0, video.with_name("pv_a.jpg")))
        frames.append(base.grab(video, min(20.0, duration(video) / 2), video.with_name("pv_b.jpg")))
    preview(frames, DATA / "previews" / "latest.jpg")

    results = []
    missing = [k for k in ("AI_YT_CLIENT_ID", "AI_YT_CLIENT_SECRET", "AI_YT_REFRESH_TOKEN") if not os.getenv(k)]
    if missing and not (a.no_upload or a.offline):
        log(f"Not uploading: add the GitHub secrets {', '.join(missing)} for the Everyday AI Easy channel "
            "(the Shorts are in the run's Artifacts)")
    elif not (a.no_upload or a.offline):
        tok = yt_token()
        now = dt.datetime.now(dt.timezone.utc)
        for k, (s, video, credits, mcredit, work) in enumerate(made):
            when = None if k == 0 else (now + dt.timedelta(hours=4 * k)).replace(microsecond=0).isoformat().replace("+00:00", "Z")
            try:
                vid, st = yt_upload(tok, video, s["title"], description(s, credits, mcredit),
                                    [base.text(t) for t in s.get("tags") or [] if base.text(t)], publish_at=when)
                results.append((s, st, vid))
            except Exception as e:
                log(f"Upload failed for '{s['title']}': {e}")
        with open(done_f, "a", encoding="utf-8") as f:
            for s, st, vid in results:
                f.write(s["url"].strip() + "\n")
        with open(DATA / "upload_log.csv", "a", encoding="utf-8") as f:
            for s, st, vid in results:
                f.write(f"{dt.datetime.now(dt.timezone.utc).isoformat()},short,\"{s['title']}\",{st},"
                        f"https://youtu.be/{vid},{s['url']}\n")
    log(f"Done: {len(made)} Shorts made, {len(results)} uploaded")
    for s, video, credits, mcredit, work in made:
        for p in work.iterdir():
            if p.name not in {video.name, "script.json"}:
                p.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FAILED: {e}")
        sys.exit(1)
