"""
Stage 06: HEAD-ONLY FINE-TUNE.

Freezes the Organika/sdxl-detector backbone and trains ONLY the classifier head
on the SDXL train split. Picks the best epoch by val loss or accuracy (SELECT_BY),
early-stops on the same signal, then evaluates on test/ with the same metrics as
the baseline.

Reads:  dataset_final/{train,val,test}/{ai,real}/
Writes: finetuned_sdxl_detector/  +  finetune_results.csv
Requires: pip install torch transformers pillow

NOTE (see NOTES.md): this run uses plain CrossEntropyLoss (no label smoothing /
temperature scaling), and early stopping did not fire in the run of record
(hit the epoch ceiling). Describe accordingly in the writeup.
"""

import csv
import os
import random
import time

import config

CACHE_DIR = config.setup_hf_env()  # before torch/transformers import

import torch          # noqa: E402
import torch.nn as nn # noqa: E402
from PIL import Image, ImageFile                                        # noqa: E402
from torch.utils.data import DataLoader, Dataset                        # noqa: E402
from transformers import AutoImageProcessor, AutoModelForImageClassification  # noqa: E402

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATA_ROOT = str(config.FINAL_ROOT)
MODEL_ID = config.MODEL_ID
OUT_DIR = str(config.FINETUNED_DIR)
RESULTS_CSV = str(config.FINETUNE_CSV)

SELECT_BY = "loss"     # "loss" (recommended) or "acc"
MAX_EPOCHS = 100
PATIENCE = 3
BATCH_SIZE = 4
LR = 1e-5
WEIGHT_DECAY = 0.01
SEED = 42
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")

assert SELECT_BY in ("loss", "acc")
random.seed(SEED)
torch.manual_seed(SEED)

START = time.time()
def hms(s):
    s = int(s); m, s = divmod(s, 60); return f"{m}m{s:02d}s" if m else f"{s}s"
def log(msg):
    print(f"[{hms(time.time()-START)}] {msg}", flush=True)


def build_label_map(id2label):
    ai_words = ("art", "artificial", "fake", "ai", "synthetic", "generated",
                "sdxl", "diffusion")
    real_words = ("human", "real", "photo", "genuine", "authentic")
    m = {}
    for idx, name in id2label.items():
        low = str(name).lower()
        if any(w in low for w in ai_words):
            m[int(idx)] = "ai"
        elif any(w in low for w in real_words):
            m[int(idx)] = "real"
        else:
            m[int(idx)] = None
    return m


class ImgDataset(Dataset):
    def __init__(self, split_dir, label2idx, train=False):
        self.items = []
        for label in ("ai", "real"):
            d = os.path.join(split_dir, label)
            if not os.path.isdir(d):
                raise SystemExit(f"Missing: {d}\nRun stage 04 first.")
            for fn in sorted(os.listdir(d)):
                if fn.lower().endswith(IMG_EXT):
                    self.items.append((os.path.join(d, fn), label2idx[label], label))
        self.train = train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, target, label = self.items[i]
        img = Image.open(path).convert("RGB")
        if self.train and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img, target, path, label


def make_collate(processor):
    def collate(batch):
        imgs = [b[0] for b in batch]
        targets = torch.tensor([b[1] for b in batch], dtype=torch.long)
        paths = [b[2] for b in batch]
        labels = [b[3] for b in batch]
        pixel_values = processor(images=imgs, return_tensors="pt")["pixel_values"]
        return pixel_values, targets, paths, labels
    return collate


@torch.no_grad()
def evaluate(model, loader, device, label_map, criterion=None, collect_rows=False):
    model.eval()
    correct, n, total_loss = 0, 0, 0.0
    conf = {("real", "real"): 0, ("real", "ai"): 0, ("ai", "ai"): 0, ("ai", "real"): 0}
    rows = []
    for pixel_values, targets, paths, labels in loader:
        pixel_values = pixel_values.to(device)
        targets = targets.to(device)
        logits = model(pixel_values=pixel_values).logits
        if criterion is not None:
            total_loss += criterion(logits, targets).item() * pixel_values.size(0)
        probs = torch.softmax(logits, dim=-1)
        preds = logits.argmax(-1).tolist()
        pconf = probs.max(-1).values.tolist()
        for pred_idx, true_label, path, c in zip(preds, labels, paths, pconf):
            pred_label = label_map[int(pred_idx)]
            ok = (pred_label == true_label)
            correct += ok; n += 1
            conf[(true_label, pred_label)] += 1
            if collect_rows:
                rows.append({"filepath": path, "true": true_label, "pred": pred_label,
                             "correct": int(ok), "confidence": round(float(c), 4)})
    avg_loss = (total_loss / max(1, n)) if criterion is not None else None
    return correct / max(1, n), avg_loss, conf, rows


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}   |   selecting best epoch by: {SELECT_BY}")

    log(f"Loading {MODEL_ID} ...")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_ID, cache_dir=CACHE_DIR).to(device)

    id2label = model.config.id2label
    label_map = build_label_map(id2label)
    if set(label_map.values()) != {"ai", "real"}:
        raise SystemExit(f"Label map failed: {dict(id2label)} -> {label_map}")
    label2idx = {v: k for k, v in label_map.items()}
    log(f"id2label={dict(id2label)}  ->  label2idx={label2idx}")

    trainable = []
    for name, p in model.named_parameters():
        p.requires_grad = ("classifier" in name)
        if p.requires_grad:
            trainable.append(name)
    if not trainable:
        raise SystemExit("No 'classifier' params found to train.")
    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    log(f"Trainable head params: {n_train_params:,} / {n_total:,} "
        f"({100*n_train_params/n_total:.3f}%)")
    log(f"Training: {trainable}")

    model.eval()  # deterministic backbone; only head params require grad

    collate = make_collate(processor)
    train_ds = ImgDataset(os.path.join(DATA_ROOT, "train"), label2idx, train=True)
    val_ds = ImgDataset(os.path.join(DATA_ROOT, "val"), label2idx)
    test_ds = ImgDataset(os.path.join(DATA_ROOT, "test"), label2idx)
    log(f"train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=2, collate_fn=collate)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    use_amp = (device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    base_acc, base_loss, _, _ = evaluate(model, val_loader, device, label_map, criterion=criterion)
    log(f"Val BEFORE fine-tuning (reference): acc={base_acc:.3f}  loss={base_loss:.4f}")

    best_val_acc, best_val_loss = -1.0, float("inf")
    best_head, best_epoch, stale = None, -1, 0

    for epoch in range(1, MAX_EPOCHS + 1):
        running, seen = 0.0, 0
        model.eval()
        for pixel_values, targets, _, _ in train_loader:
            pixel_values = pixel_values.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(pixel_values=pixel_values).logits
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * pixel_values.size(0)
            seen += pixel_values.size(0)

        val_acc, val_loss, _, _ = evaluate(model, val_loader, device, label_map, criterion=criterion)

        if SELECT_BY == "loss":
            improved = val_loss < best_val_loss - 1e-4
        else:
            improved = val_acc > best_val_acc + 1e-4

        flag = ""
        if improved:
            best_val_acc, best_val_loss, best_epoch = val_acc, val_loss, epoch
            best_head = {n: p.detach().clone()
                         for n, p in model.named_parameters() if p.requires_grad}
            stale = 0
            flag = "  <- best"
        else:
            stale += 1

        log(f"epoch {epoch:2d}/{MAX_EPOCHS}  train_loss={running/seen:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}{flag}")

        if stale >= PATIENCE:
            log(f"Early stop: no {SELECT_BY} improvement for {PATIENCE} epochs.")
            break

    if best_head is not None:
        sd = model.state_dict()
        for n, p in best_head.items():
            sd[n] = p
        model.load_state_dict(sd)
    log(f"Best epoch: {best_epoch}  (val_loss={best_val_loss:.4f}, "
        f"val_acc={best_val_acc:.3f})  [selected by {SELECT_BY}]")

    os.makedirs(OUT_DIR, exist_ok=True)
    model.save_pretrained(OUT_DIR)
    processor.save_pretrained(OUT_DIR)
    log(f"Saved fine-tuned model -> {OUT_DIR}")

    test_acc, test_loss, conf, rows = evaluate(
        model, test_loader, device, label_map, criterion=criterion, collect_rows=True)
    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filepath", "true", "pred", "correct", "confidence"])
        w.writeheader(); w.writerows(rows)

    n_real = sum(1 for _, _, l in test_ds.items if l == "real")
    n_ai = len(test_ds) - n_real
    log("=" * 52)
    log("FINE-TUNED TEST RESULTS (in-distribution / SDXL)")
    log(f"  Overall accuracy : {test_acc:.3f}   (test_loss={test_loss:.4f})")
    log(f"  Real recall      : {conf[('real','real')]/max(1,n_real):.3f}  "
        f"({conf[('real','real')]}/{n_real})")
    log(f"  AI   recall      : {conf[('ai','ai')]/max(1,n_ai):.3f}  "
        f"({conf[('ai','ai')]}/{n_ai})")
    log("  Confusion (true -> pred):")
    log(f"    real->real {conf[('real','real')]:4d}   real->ai {conf[('real','ai')]:4d}")
    log(f"    ai->ai     {conf[('ai','ai')]:4d}   ai->real {conf[('ai','real')]:4d}")
    log(f"  Per-image predictions -> {RESULTS_CSV}")
    log(f"TOTAL RUNTIME: {hms(time.time()-START)}")


if __name__ == "__main__":
    main()
