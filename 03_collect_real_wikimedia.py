"""
Stage 03: REAL-IMAGE COLLECTION (the "real" class).

Pulls Creative-Commons / public-domain photographs from Wikimedia Commons for
each topic, to sit opposite the SDXL synthetic images. Timestamped live progress
so you can tell it is running on a long scrape.

  * 5 topics collect well: wildfire, flood, protest_crowd,
    war_torn_destroyed_building, plane_crash_debris
  * 3 WEAK topics return noisy/off-topic results (documents, fake_news, memes).
    See NOTES.md -- these likely need a different real source.

Requires: pip install requests   (no GPU)
Safe to re-run: skips files already downloaded, de-dupes by Wikimedia sha1.
TIP: run unbuffered so nothing is delayed:  python -u 03_collect_real_wikimedia.py
"""

import csv
import os
import re
import time

import requests

import config

# ----------------------------------------------------------------------
# Config (paths + UA come from config.py; knobs stay local)
# ----------------------------------------------------------------------
REAL_ROOT = str(config.REAL_ROOT / "real")
METADATA_CSV = str(config.METADATA_REAL)
USER_AGENT = config.USER_AGENT

TARGET_PER_TOPIC = 450          # 150 x 3 synthetic folders
TARGET_WIDTH = 1280
MIN_SIDE = 400
ACCEPT_MIME = {"image/jpeg", "image/png"}
SLEEP_BETWEEN_CALLS = 0.5
API_LIMIT = 50
MAX_PAGES_PER_QUERY = 40

API = "https://commons.wikimedia.org/w/api.php"

# NOTE: the war-torn key here is SINGULAR (war_torn_destroyed_building). The
# generator/normalize slug is PLURAL (war_torn_destroyed_buildings). See NOTES.md
# -- this mismatch currently tags real war-torn images as topic "unknown".
TOPIC_QUERIES = {
    "wildfire": ["wildfire", "forest fire", "bushfire", "wildfire smoke", "grass fire"],
    "flood": ["flood", "flooding", "flooded street", "river flood", "flood damage"],
    "protest_crowd": ["protest demonstration", "protest crowd", "street protest",
                      "political demonstration", "rally crowd"],
    "war_torn_destroyed_building": ["war destroyed building", "bombed building",
                                    "building destruction war", "ruined building conflict",
                                    "shelled building"],
    "plane_crash_debris": ["plane crash wreckage", "aircraft accident debris",
                           "aviation crash site", "airplane wreckage", "crashed aircraft"],
    # ---- WEAK: expect noisy/off-topic results from Commons ----
    "documents": ["official document", "printed document", "archival document", "paper form"],
    "fake_news": ["newspaper front page", "printed newspaper", "tabloid newspaper"],
    "memes": ["internet meme", "image macro"],
}
WEAK_TOPICS = {"documents", "fake_news", "memes"}

os.makedirs(REAL_ROOT, exist_ok=True)
session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})

START = time.time()


def hms(seconds):
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def log(msg):
    print(f"[{hms(time.time() - START)}] {msg}", flush=True)


def strip_html(s):
    return re.sub(r"<[^>]+>", "", s).strip() if s else ""


def meta_value(extmeta, key):
    d = (extmeta or {}).get(key)
    return strip_html(d.get("value")) if isinstance(d, dict) else ""


def ext_from(url, mime):
    m = re.search(r"\.(jpg|jpeg|png)(?:$|\?)", url, re.IGNORECASE)
    if m:
        return "jpg" if m.group(1).lower() in ("jpg", "jpeg") else "png"
    return "png" if mime == "image/png" else "jpg"


def api_get(params, retries=4):
    delay = 1.0
    for attempt in range(retries):
        try:
            r = session.get(API, params=params, timeout=30)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == retries - 1:
                log(f"    API failed after {retries} tries: {e}")
                return None
            time.sleep(delay)
            delay *= 2
    return None


def load_existing_sha1(csv_path):
    seen = set()
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("sha1"):
                    seen.add(row["sha1"])
    return seen


def next_index(topic_dir, slug):
    mx = -1
    if os.path.isdir(topic_dir):
        for fn in os.listdir(topic_dir):
            m = re.match(rf"{re.escape(slug)}_(\d+)\.", fn)
            if m:
                mx = max(mx, int(m.group(1)))
    return mx + 1


def download(url, path):
    try:
        with session.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    if chunk:
                        f.write(chunk)
        return True
    except Exception as e:
        log(f"    download failed: {e}")
        if os.path.exists(path):
            os.remove(path)
        return False


def collect_topic(slug, queries, target, seen_sha1, rows):
    topic_start = time.time()
    topic_dir = os.path.join(REAL_ROOT, slug)
    os.makedirs(topic_dir, exist_ok=True)
    idx = next_index(topic_dir, slug)
    got = len([f for f in os.listdir(topic_dir)
               if f.lower().endswith((".jpg", ".png"))])
    added_this_topic = 0

    tag = "  [WEAK topic - results may be off]" if slug in WEAK_TOPICS else ""
    log(f"[{slug}] START - have {got}, want {target}{tag}")

    for query in queries:
        if got >= target:
            break
        log(f"[{slug}] query: '{query}'")
        params = {
            "action": "query", "format": "json", "generator": "search",
            "gsrsearch": query, "gsrnamespace": 6, "gsrlimit": API_LIMIT,
            "prop": "imageinfo",
            "iiprop": "url|extmetadata|mime|size|sha1",
            "iiurlwidth": TARGET_WIDTH,
        }
        pages_done = 0
        while got < target and pages_done < MAX_PAGES_PER_QUERY:
            data = api_get(params)
            time.sleep(SLEEP_BETWEEN_CALLS)
            if not data:
                break
            pages = (data.get("query") or {}).get("pages")
            if not pages:
                break

            scanned_this_page = 0
            kept_this_page = 0
            for page in pages.values():
                if got >= target:
                    break
                scanned_this_page += 1
                infos = page.get("imageinfo")
                if not infos:
                    continue
                ii = infos[0]

                mime = ii.get("mime", "")
                if mime not in ACCEPT_MIME:
                    continue
                w, h = ii.get("width", 0), ii.get("height", 0)
                if min(w, h) < MIN_SIDE:
                    continue
                sha1 = ii.get("sha1", "")
                if not sha1 or sha1 in seen_sha1:
                    continue

                dl_url = ii.get("thumburl") or ii.get("url")
                if not dl_url:
                    continue
                ext = ext_from(dl_url, mime)
                fpath = os.path.join(topic_dir, f"{slug}_{idx:04d}.{ext}")

                if not download(dl_url, fpath):
                    continue

                seen_sha1.add(sha1)
                extmeta = ii.get("extmetadata", {})
                rows.append({
                    "filepath": fpath, "topic": slug, "label": "real",
                    "source": "wikimedia_commons",
                    "page_title": page.get("title", ""), "query": query,
                    "license": meta_value(extmeta, "LicenseShortName"),
                    "artist": meta_value(extmeta, "Artist"),
                    "credit": meta_value(extmeta, "Credit"),
                    "sha1": sha1, "orig_width": w, "orig_height": h,
                    "source_url": ii.get("descriptionurl", ii.get("url", "")),
                })
                idx += 1
                got += 1
                added_this_topic += 1
                kept_this_page += 1

            pages_done += 1
            rate = added_this_topic / max(1e-9, time.time() - topic_start)
            eta = (target - got) / rate if rate > 0 else float("inf")
            eta_str = hms(eta) if rate > 0 and got < target else "-"
            log(f"[{slug}] page {pages_done}: kept {kept_this_page}/"
                f"{scanned_this_page} | total {got}/{target} | "
                f"{rate:.1f} img/s | ETA {eta_str}")

            cont = data.get("continue")
            if not cont:
                break
            params.update(cont)

    dur = time.time() - topic_start
    log(f"[{slug}] DONE - collected {got} (+{added_this_topic} this run) in {hms(dur)}\n")
    return got


def main():
    log(f"Collector starting. Target {TARGET_PER_TOPIC}/topic across "
        f"{len(TOPIC_QUERIES)} topics.")
    log(f"Saving under: {REAL_ROOT}")

    seen_sha1 = load_existing_sha1(METADATA_CSV)
    log(f"Loaded {len(seen_sha1)} known sha1(s) from previous runs.")

    rows, summary = [], {}
    for slug, queries in TOPIC_QUERIES.items():
        summary[slug] = collect_topic(slug, queries, TARGET_PER_TOPIC, seen_sha1, rows)

    if rows:
        write_header = not os.path.exists(METADATA_CSV)
        fields = ["filepath", "topic", "label", "source", "page_title", "query",
                  "license", "artist", "credit", "sha1",
                  "orig_width", "orig_height", "source_url"]
        with open(METADATA_CSV, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    total = time.time() - START
    log("=" * 50)
    log("SUMMARY (total on disk, per topic):")
    for slug, n in summary.items():
        mark = "  <- WEAK, review these" if slug in WEAK_TOPICS else ""
        log(f"  {slug:32s} {n}{mark}")
    log(f"New rows this run: {len(rows)}  ->  {METADATA_CSV}")
    log(f"TOTAL RUNTIME: {hms(total)}")


if __name__ == "__main__":
    main()
