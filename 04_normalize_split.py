"""
Stage 04: NORMALIZE + SPLIT for real-vs-AI detection.

Samples TARGET_PER_CLASS from each class, de-dupes by file CONTENT (md5),
re-encodes everything (EXIF dropped), and writes a stratified train/val/test
split + manifest.csv.

NOTE (see NOTES.md before trusting results):
  * AI and real are currently re-encoded at DIFFERENT JPEG quality (70 vs 100).
    That is a class-conditional signal a detector can exploit -- resolve before
    reporting accuracy.
  * "stratified" here means by CLASS, not by TOPIC.
  * This reads the raw dataset folders, not the curated *_final.csv files.

Requires: pip install pillow
"""

import csv, hashlib, os, random, shutil, time
from pathlib import Path
from PIL import Image, ImageFile

import config

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ----------------------------------------------------------------------
# Config (paths from config.py)
# ----------------------------------------------------------------------
DATA_ROOT = config.DATA_ROOT
REAL_DIRS = ["dataset_real"]
AI_DIRS   = ["datasetV1", "datasetV2", "datasetV3"]

TARGET_PER_CLASS = 1500

OUT_ROOT = config.FINAL_ROOT
MANIFEST = config.MANIFEST

OUT_FORMAT = "JPEG"
MAX_SIDE = 1024
REAL_QUALITY = 100
AI_QUALITY = 70            # NOTE: differs from REAL_QUALITY on purpose-of-record; see NOTES.md

SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42

TOPICS = sorted([
    "war_torn_destroyed_buildings", "plane_crash_debris", "protest_crowd",
    "fake_news", "documents", "wildfire", "flood", "memes",
], key=len, reverse=True)

IMG_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")

START = time.time()
def hms(s):
    s = int(s); m, s = divmod(s, 60); return f"{m}m{s:02d}s" if m else f"{s}s"
def log(msg):
    print(f"[{hms(time.time()-START)}] {msg}", flush=True)


def guess_topic(filename):
    low = filename.lower()
    for t in TOPICS:
        if t in low:
            return t
    return "unknown"


def collect(dir_names, class_name):
    seen_hashes, imgs, exact_dups = {}, [], 0
    for name in dir_names:
        folder = DATA_ROOT / name
        if not folder.is_dir():
            raise SystemExit(f"{class_name}: folder not found -> {folder}")
        found = 0
        for p in sorted(folder.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMG_EXT:
                try:
                    h = hashlib.md5(p.read_bytes()).hexdigest()
                except Exception as e:
                    log(f"    unreadable, skipped: {p.name}: {e}")
                    continue
                if h in seen_hashes:
                    exact_dups += 1
                    continue
                seen_hashes[h] = p
                imgs.append(p)
                found += 1
        log(f"  {class_name}: {name} -> {found} new images")
    if exact_dups:
        log(f"  {class_name}: skipped {exact_dups} exact-content duplicates")
    log(f"  {class_name} TOTAL unique available: {len(imgs)}")
    return imgs


def sample_target(imgs, class_name, target, seed):
    if len(imgs) < target:
        raise SystemExit(
            f"{class_name}: only {len(imgs)} unique images but "
            f"TARGET_PER_CLASS={target}. Add sources or lower the target.")
    rng = random.Random(seed)
    picked = rng.sample(imgs, target)
    log(f"  {class_name}: sampled {target} of {len(imgs)}")
    return picked


def stratified_split(items, splits, seed):
    rng = random.Random(seed)
    items = items[:]
    rng.shuffle(items)
    n = len(items)
    n_train = int(n * splits["train"])
    n_val = int(n * splits["val"])
    return {"train": items[:n_train],
            "val": items[n_train:n_train + n_val],
            "test": items[n_train + n_val:]}


def normalize_save(src, dst, quality):
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            w, h = im.size
            scale = min(1.0, MAX_SIDE / max(w, h))
            if scale < 1.0:
                im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                               Image.LANCZOS)
            im.save(dst, format=OUT_FORMAT, quality=quality, optimize=True)
        return True
    except Exception as e:
        log(f"    skip (bad image) {os.path.basename(src)}: {e}")
        return False


def main():
    log("Collecting images:")
    real = collect(REAL_DIRS, "real")
    ai = collect(AI_DIRS, "ai")

    log(f"Sampling {TARGET_PER_CLASS} per class:")
    real = sample_target(real, "real", TARGET_PER_CLASS, SEED)
    ai = sample_target(ai, "ai", TARGET_PER_CLASS, SEED + 1)

    real = [(p, "real") for p in real]
    ai = [(p, "ai") for p in ai]

    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    for split in SPLITS:
        for label in ("real", "ai"):
            (OUT_ROOT / split / label).mkdir(parents=True, exist_ok=True)

    real_split = stratified_split(real, SPLITS, SEED)
    ai_split = stratified_split(ai, SPLITS, SEED + 1)

    rows = []
    counts = {s: {"real": 0, "ai": 0} for s in SPLITS}
    ext = ".jpg" if OUT_FORMAT == "JPEG" else "." + OUT_FORMAT.lower()

    for split in SPLITS:
        for label, bucket in (("real", real_split[split]), ("ai", ai_split[split])):
            quality = REAL_QUALITY if label == "real" else AI_QUALITY
            for i, (src, _) in enumerate(bucket):
                topic = guess_topic(os.path.basename(src))
                newname = f"{label}_{split}_{i:04d}{ext}"
                dst = OUT_ROOT / split / label / newname
                if not normalize_save(src, dst, quality=quality):
                    continue
                counts[split][label] += 1
                rows.append({
                    "filepath": str(dst), "label": label,
                    "label_id": 0 if label == "ai" else 1,  # ImageFolder order
                    "topic": topic, "split": split, "orig_path": str(src),
                })
            log(f"  {split}/{label}: {counts[split][label]} done (Quality={quality})")

    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filepath", "label", "label_id",
                                          "topic", "split", "orig_path"])
        w.writeheader()
        w.writerows(rows)

    log("=" * 52)
    log("SPLIT SUMMARY (images written):")
    for s in SPLITS:
        log(f"  {s:5s}  real={counts[s]['real']:4d}  ai={counts[s]['ai']:4d}"
            f"  total={counts[s]['real']+counts[s]['ai']:4d}")
    log(f"Total written: {len(rows)}")
    log(f"Tree:     {OUT_ROOT}/<split>/<label>/")
    log(f"Manifest: {MANIFEST}")
    log(f"TOTAL RUNTIME: {hms(time.time()-START)}")


if __name__ == "__main__":
    main()
