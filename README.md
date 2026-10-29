# Dark History – Faceless Video Agent

Yeh agent roz khud:
- **3 Shorts** (vertical, ~55 sec) aur **1 Long video** (~6 min) banata hai
- Topic chunta hai (kabhi repeat nahi), Google Search se facts check karke script likhta hai
- Insaan jaisi English awaaz, asli archive tasveerein (Wikimedia Commons), Ken Burns motion,
  film grain, captions aur apna dark background music lagata hai
- Thumbnail, title, description, tags, hashtags aur image credits khud banata hai
- YouTube par upload karta hai

Sab kuch GitHub Actions par chalta hai. Aap ka computer band ho tab bhi chalega.

## Time table (UK time, summer)
| Video | Time |
|---|---|
| Short 1 | ~08:47 |
| Short 2 | ~12:52 |
| Short 3 | ~17:43 |
| Long | ~19:13 |

Sardiyon mein (GMT) yeh sab 1 ghanta pehle chalein ge. GitHub kabhi kabhi 5–30 minute late chalata hai.

## Secrets (Settings → Secrets and variables → Actions)
| Name | Zaroori? | Kahan se |
|---|---|---|
| GEMINI_API_KEY | Haan | aistudio.google.com/apikey (free) |
| YT_CLIENT_ID | Haan | Google Cloud → YouTube Data API v3 → OAuth client |
| YT_CLIENT_SECRET | Haan | wahi |
| YT_REFRESH_TOKEN | Haan | developers.google.com/oauthplayground |
| PEXELS_API_KEY | Nahi | pexels.com/api (free, extra tasveerein) |
| ELEVENLABS_API_KEY | Nahi | elevenlabs.io (paid, aur zyada asli awaaz) |

Variables (optional): `PRIVACY` = public / unlisted / private, `CHANNEL_NAME`, `TTS_RATE` (awaaz ki speed, default 1.05).

### Documentary + Shorts (studio.py) – sab free
| Name | Zaroori? | Kaam |
|---|---|---|
| GROQ_API_KEY | Haan | console.groq.com (free) – script likhta hai |
| GOOGLE_TTS_KEY | Haan | Google Cloud Text-to-Speech (Chirp HD free tier) – awaaz |
| PEXELS_API_KEY, PIXABAY_API_KEY | Haan | asli video clips (free) |
| OPENROUTER_API_KEY | Nahi | openrouter.ai (free) – Groq busy ho to sirf `:free` models |
| GEMINI_API_KEY | Nahi | sirf **free-tier** project ki key (billing wale project par 402 aata hai) |
| CF_ACCOUNT_ID, CF_API_TOKEN | Nahi | Cloudflare free – aakhri backup writer + AI tasveer |
| YT_API_KEY | Nahi | YouTube Data API key – trending dark-history videos dekh kar topic chunta hai |

Topic roz trending data se chuna jata hai (YouTube par chalne wali dark-history videos, Wikipedia par
sab se zyada parhe jane wale history articles, aur is hafte ki barsiyan), aur facts Wikipedia se check hote hain.
Har run mein sirf 3–4 AI requests jati hain; 429 aaye to API jitna kahe utna wait karke dobara koshish hoti hai.

Test run: Actions → "Daily Documentary + Shorts" → Run workflow → upload = **no** (video sirf Artifacts mein aati hai).

## Hath se chalana
Actions tab → "Daily Shorts" ya "Daily Long Video" → **Run workflow**.
Bani hui video har run ke neeche "Artifacts" mein 7 din tak download ke liye rehti hai.

## Zaroori baatein
- Naya Google Cloud project jab tak YouTube ka audit pass na kare, YouTube us ki uploads **private** rakhta hai.
  Public ke liye "YouTube API Services – Audit and Quota Extension" form bharein.
- Google Cloud ki OAuth consent screen ko **"In production"** par rakhein, warna token 7 din mein expire ho jata hai.
- Long videos par custom thumbnail ke liye channel ko phone se verify karein (youtube.com/verify).
- Agar 60 din tak repo mein koi activity na ho to GitHub schedule band kar deta hai. Agent khud roz commit karta hai, is liye yeh masla nahi hona chahiye.
