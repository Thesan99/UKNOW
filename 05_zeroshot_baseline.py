"""
Stage 05: ZERO-SHOT BASELINE.

Runs Organika/sdxl-detector on the test set AS-IS (no fine-tuning) and reports
accuracy -- the reference point for the 2x2. Run BEFORE fine-tuning.

Label mapping is read from model.config.id2label at runtime and matched to the
ai/real folders by text, so the accuracy can never silently invert.

Reads:  dataset_final/test/{ai,real}/
Writes: baseline_results.csv
Requires: pip install torch transformers pillow
"""

import csv
import os
import time

import config

CACHE_DIR = config.setup_hf_env()  # before torch/transformers import

import torch                                                            # noqa: E402
from PIL import Image, ImageFile                                        # noqa: E402
from transformers import AutoImageProcessor, AutoModelForImageClassification  # noqa: E402

ImageFile.LOAD_TRUNCATED_IMAGES = True

TEST_DIR = str(config.FINAL_ROOT / "test")
MODEL_ID = config.MODEL_ID
RESULTS_CSV = str(config.BASELINE_CSV)
BATCH_SIZE = 4
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")

START = time.time()
def hms(s):
    s = int(s); m, s = divmod(s, 60); return f"{m}m{s:02d}s" if m else f"{s}s"
def log(msg):
    print(f"[{hms(time.time()-START)}] {msg}", flush=True)


def build_label_map(id2label):
    ai_words = ("art", "artificial", "fake", "ai", "synthetic", "generated",
                "sdxl", "diffusion")
    real_words = ("human", "real", "photo", "genuine", "authentic")
    mapping = {}
    for idx, name in id2label.items():
        low = str(name).lower()
        if any(w in low for w in ai_words):
            mapping[int(idx)] = "ai"
        elif any(w in low for w in real_words):
            mapping[int(idx)] = "real"
        else:
            mapping[int(idx)] = None
    return mapping


def load_test_items(test_dir):
    items = []
    for label in ("ai", "real"):
        d = os.path.join(test_dir, label)
        if not os.path.isdir(d):
            raise SystemExit(f"Missing folder: {d}\nRun stage 04 first.")
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith(IMG_EXT):
                items.append((os.path.join(d, fn), label))
    return items


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}")
    log(f"Loading {MODEL_ID} ...")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_ID, cache_dir=CACHE_DIR).to(device).eval()

    id2label = model.config.id2label
    label_map = build_label_map(id2label)
    log(f"Model id2label: {dict(id2label)}")
    log(f"Mapped to folders: {label_map}")
    if any(v is None for v in label_map.values()) or set(label_map.values()) != {"ai", "real"}:
        raise SystemExit(
            f"Could not map model labels to ai/real. id2label={dict(id2label)} "
            f"-> {label_map}. Edit build_label_map().")

    items = load_test_items(TEST_DIR)
    log(f"Test images: {len(items)}")

    rows, correct = [], 0
    conf = {("real", "real"): 0, ("real", "ai"): 0,
            ("ai", "ai"): 0, ("ai", "real"): 0}

    for start in range(0, len(items), BATCH_SIZE):
        batch = items[start:start + BATCH_SIZE]
        imgs = [Image.open(p).convert("RGB") for p, _ in batch]
        inputs = processor(images=imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        preds = logits.argmax(-1).tolist()
        confs = torch.softmax(logits, dim=-1).max(-1).values.tolist()

        for (path, true_label), pred_idx, c in zip(batch, preds, confs):
            pred_label = label_map[int(pred_idx)]
            ok = (pred_label == true_label)
            correct += ok
            conf[(true_label, pred_label)] += 1
            rows.append({"filepath": path, "true": true_label, "pred": pred_label,
                         "correct": int(ok), "confidence": round(float(c), 4)})

        if (start // BATCH_SIZE) % 3 == 0:
            done = min(start + BATCH_SIZE, len(items))
            log(f"  {done}/{len(items)}  running acc={correct/done:.3f}")

    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filepath", "true", "pred", "correct", "confidence"])
        w.writeheader(); w.writerows(rows)

    n = len(items)
    n_real = sum(1 for _, l in items if l == "real")
    n_ai = n - n_real
    real_correct = conf[("real", "real")]
    ai_correct = conf[("ai", "ai")]

    log("=" * 52)
    log("ZERO-SHOT BASELINE RESULTS")
    log(f"  Overall accuracy : {correct/n:.3f}  ({correct}/{n})")
    log(f"  Real recall      : {real_correct/n_real:.3f}  ({real_correct}/{n_real})")
    log(f"  AI   recall      : {ai_correct/n_ai:.3f}  ({ai_correct}/{n_ai})")
    log("  Confusion (true -> pred):")
    log(f"    real->real {conf[('real','real')]:4d}   real->ai {conf[('real','ai')]:4d}")
    log(f"    ai->ai     {conf[('ai','ai')]:4d}   ai->real {conf[('ai','real')]:4d}")
    log(f"  Per-image predictions -> {RESULTS_CSV}")
    log(f"TOTAL RUNTIME: {hms(time.time()-START)}")


if __name__ == "__main__":
    main()
